#include <assert.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <fstream>

#include "../../include/on-device/kernels_od.cuh"
#include "../../include/on-device/kernels_fluidbcs.cuh"
#include "../../include/on-device/initialize_od.cuh"
#include "../../include/on-device/kernels_od_intvar.cuh"
#include "../../include/on-device/kernels_intvarbcs.cuh"

void doSomethingWithTransfer(float *h_fluidvars, int Nx, int Ny, int Nz);

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char* argv[]){
	// Parse inputs - provided by imhdLauncher.py because Python parsing is easiest, and Python launching is easy
	int Nt = atoi(argv[1]);
	int num_bench_iters = atoi(argv[2]);
	int Nx = atoi(argv[3]);
	int Ny = atoi(argv[4]);
	int Nz = atoi(argv[5]);
	float J0 = atof(argv[6]);
	float D = atof(argv[7]);
	float x_min = atof(argv[8]);
	float x_max = atof(argv[9]);
	float y_min = atof(argv[10]);
	float y_max = atof(argv[11]);
	float z_min = atof(argv[12]);
	float z_max = atof(argv[13]);
	float dt = atof(argv[14]);

	int meshblockdims_xthreads = atoi(argv[15]);
	int meshblockdims_ythreads = atoi(argv[16]);
	int meshblockdims_zthreads = atoi(argv[17]);

	int initblockdims_xthreads = atoi(argv[18]);
	int initblockdims_ythreads = atoi(argv[19]);
	int initblockdims_zthreads = atoi(argv[20]);
	
	int intvarblockdims_xthreads = atoi(argv[21]);
	int intvarblockdims_ythreads = atoi(argv[22]);
	int intvarblockdims_zthreads = atoi(argv[23]);

	int fluidblockdims_xthreads = atoi(argv[24]);
	int fluidblockdims_ythreads = atoi(argv[25]);
	int fluidblockdims_zthreads = atoi(argv[26]);

	int SM_mult_x_grid = atoi(argv[27]);
	int SM_mult_y_grid = atoi(argv[28]);
	int SM_mult_z_grid = atoi(argv[29]);

	int SM_mult_x_intvar = atoi(argv[30]);
	int SM_mult_y_intvar = atoi(argv[31]);
	int SM_mult_z_intvar = atoi(argv[32]);

	float dx = (x_max - x_min) / (Nx - 1);
	float dy = (y_max - y_min) / (Ny - 1);
	float dz = (z_max - z_min) / (Nz - 1);

	int fluid_threads = fluidblockdims_xthreads * fluidblockdims_ythreads * fluidblockdims_zthreads;
	std::cout << "fluid_threads=" << fluid_threads << std::endl;
	std::cout << "fluid_xthreads=" << fluidblockdims_xthreads << std::endl;
	std::cout << "fluid_ythreads=" << fluidblockdims_ythreads << std::endl;
	std::cout << "fluid_zthreads=" << fluidblockdims_zthreads << std::endl;

	int intvar_threads = intvarblockdims_xthreads * intvarblockdims_ythreads * intvarblockdims_zthreads;
	std::cout << "intvar_threads=" << intvar_threads << std::endl;

	size_t fluid_var_size = sizeof(float) * Nx * Ny * Nz;
	size_t fluid_data_size = 8 * fluid_var_size;
	
	float* h_fluidvars;

	h_fluidvars = (float*)malloc(fluid_data_size);

	// Initialize device data 
	int deviceId;
	int numberOfSMs;

	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	float *fluidvars, *intvars;
	float *x_grid, *y_grid, *z_grid;

	// Get that data on device global memory 
	checkCuda(cudaMalloc(&fluidvars, 8 * fluid_var_size));
	checkCuda(cudaMalloc(&intvars, 8 * fluid_var_size));

	checkCuda(cudaMalloc(&x_grid, sizeof(float) * Nx));
	checkCuda(cudaMalloc(&y_grid, sizeof(float) * Ny));
	checkCuda(cudaMalloc(&z_grid, sizeof(float) * Nz));

   // Execution grid configurations
	dim3 exec_grid_dims(numberOfSMs, numberOfSMs, numberOfSMs);
	dim3 exec_grid_dims_grid(SM_mult_x_grid * numberOfSMs, SM_mult_y_grid * numberOfSMs, SM_mult_z_grid * numberOfSMs);
	dim3 exec_grid_dims_intvar(SM_mult_x_intvar * numberOfSMs, SM_mult_y_intvar * numberOfSMs, SM_mult_z_intvar * numberOfSMs);
   	dim3 exec_grid_dims_fluidadvance(numberOfSMs, numberOfSMs, numberOfSMs);

	// Execution grid configurations for the Qint boundary microkernels
	dim3 exec_grid_dims_qintbdry_front(numberOfSMs, numberOfSMs, 1); // can also be used for PBCs
	dim3 exec_grid_dims_qintbdry_leftright(numberOfSMs, 1, numberOfSMs);
	dim3 exec_grid_dims_qintbdry_topbottom(1, numberOfSMs, numberOfSMs);

	dim3 exec_grid_dims_qintbdry_frontright(numberOfSMs, 1, 1);
	dim3 exec_grid_dims_qintbdry_frontbottom(1, numberOfSMs, 1);
	dim3 exec_grid_dims_qintbdry_bottomright(1, 1, numberOfSMs);

	// Threadblock execution configurations
	dim3 mesh_block_dims(meshblockdims_xthreads, meshblockdims_ythreads, meshblockdims_zthreads);
	dim3 init_block_dims(initblockdims_xthreads, initblockdims_ythreads, initblockdims_zthreads);
	dim3 intvar_block_dims(intvarblockdims_xthreads, intvarblockdims_ythreads, intvarblockdims_zthreads);
	dim3 fluid_block_dims(fluidblockdims_xthreads, fluidblockdims_ythreads, fluidblockdims_zthreads);

	// Threadblock execution configurations for the Fluid Advance Microkernels
	dim3 fluidadvance_blockdims(8, 8, 8); // power of two

	// Threadblock execution configurations for the Qint boundary microkernels
	/* Really doubt these need to ever be changed. Maybe 8 -> 10 */
	dim3 qintbdry_front_blockdims(8, 8, 1); // can also be used for PBCs
	dim3 qintbdry_leftright_blockdims(8, 1, 8);
	dim3 qintbdry_topbottom_blockdims(1, 8, 8);
	dim3 qintbdry_frontright_blockdims(1024, 1, 1);
	dim3 qintbdry_frontbottom_blockdims(1, 1024, 1);
	dim3 qintbdry_bottomright_blockdims(1, 1, 1024);

	InitializeGrid<<<exec_grid_dims_grid, mesh_block_dims>>>(x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz, x_grid, y_grid, z_grid, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());

	ScrewPinch<<<exec_grid_dims, init_block_dims>>>(fluidvars, J0, x_grid, y_grid, z_grid, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());

	ComputeIntermediateVariablesNoDiff<<<exec_grid_dims_intvar, intvar_block_dims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());    
	
	// Maybe this should be in a wrapper
	QintBdryFrontNoDiff<<<exec_grid_dims_qintbdry_front, qintbdry_front_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
	QintBdryLeftRightNoDiff<<<exec_grid_dims_qintbdry_leftright, qintbdry_leftright_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
	QintBdryTopBottomNoDiff<<<exec_grid_dims_qintbdry_topbottom, qintbdry_topbottom_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
	QintBdryFrontBottomNoDiff<<<exec_grid_dims_qintbdry_frontbottom, qintbdry_frontbottom_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
	QintBdryFrontRightNoDiff<<<exec_grid_dims_qintbdry_frontright, qintbdry_frontright_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
	QintBdryBottomRightNoDiff<<<exec_grid_dims_qintbdry_bottomright, qintbdry_bottomright_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());    

	QintBdryPBCs<<<exec_grid_dims_qintbdry_front, qintbdry_front_blockdims>>>(fluidvars, intvars, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());    
    
	// Benchmarking
    cudaEvent_t start, stop, start_bcs, stop_bcs, start_intvar, stop_intvar, start_intvar_bcs, stop_intvar_bcs;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start_bcs);
    cudaEventCreate(&stop_bcs);
    cudaEventCreate(&start_intvar);
    cudaEventCreate(&stop_intvar);
	cudaEventCreate(&start_intvar_bcs);
	cudaEventCreate(&stop_intvar_bcs);

	cudaEvent_t start_nodiff, stop_nodiff, start_intvar_nodiff, stop_intvar_nodiff;
	cudaEventCreate(&start_nodiff);
	cudaEventCreate(&stop_nodiff);
	cudaEventCreate(&start_intvar_nodiff);
	cudaEventCreate(&stop_intvar_nodiff);

	cudaEvent_t start_d2h, stop_d2h;
	cudaEventCreate(&start_d2h);
	cudaEventCreate(&stop_d2h);

    float fluid_time = 0.0, bcs_time = 0.0, intvar_time = 0.0, intvar_bcs_time = 0.0;
	float fluid_time_nodiff = 0.0, intvar_time_nodiff = 0.0; 
	float d2h_time = 0.0;

    // For recording the benchmarking data
    std::ofstream bench_file;
    bench_file.open("../data/benchNxNyNz_" + std::to_string(Nx) + std::to_string(Ny) + std::to_string(Nz) + ".csv");

    bench_file << "it, fluid_time_nodiff(ms), fluid_time (ms), bcs_time (ms), intvar_time (ms), intvar_time_nodiff (ms), intvar_bcs_time (ms), d2h_time (ms), fluid_threads, bcs_threads, intvar_threads, intvar_bcs_threads" << std::endl; 

	// Simulation loop
	for (size_t it = 0; it < Nt; it++){
		std::cout << "Starting iteration " << it << std::endl;

		std::cout << "Benchmarking fluid microkernels w/no diffusion" << std::endl; 
        cudaEventRecord(start_nodiff);
		for (size_t il = 0; il < num_bench_iters; il++){
			FluidAdvanceMicroRhoLocalNoDiff<<<exec_grid_dims_fluidadvance, fluidadvance_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
			FluidAdvanceMicroRhoVXLocalNoDiff<<<exec_grid_dims_fluidadvance, fluidadvance_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
			FluidAdvanceMicroRhoVYLocalNoDiff<<<exec_grid_dims_fluidadvance, fluidadvance_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
			FluidAdvanceMicroRhoVZLocalNoDiff<<<exec_grid_dims_fluidadvance, fluidadvance_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
			FluidAdvanceMicroBXLocalNoDiff<<<exec_grid_dims_fluidadvance, fluidadvance_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
			FluidAdvanceMicroBYLocalNoDiff<<<exec_grid_dims_fluidadvance, fluidadvance_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
			FluidAdvanceMicroBZLocalNoDiff<<<exec_grid_dims_fluidadvance, fluidadvance_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
			FluidAdvanceMicroELocalNoDiff<<<exec_grid_dims_fluidadvance, fluidadvance_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
		}
		cudaEventRecord(stop_nodiff);
        cudaEventSynchronize(stop_nodiff);
        cudaEventElapsedTime(&fluid_time_nodiff, start_nodiff, stop_nodiff);

		std::cout << "Benchmarking fluid interior w/diffusion" << std::endl; 
        cudaEventRecord(start);
		for (size_t il = 0; il < num_bench_iters; il++){
			FluidAdvanceLocal<<<exec_grid_dims, fluid_block_dims>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
		}
		cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&fluid_time, start, stop);

		std::cout << "Benchmarking fluid B.Cs" << std::endl; 
        cudaEventRecord(start_bcs);
		for (size_t il = 0; il < num_bench_iters; il++){
			BoundaryConditions<<<exec_grid_dims, fluid_block_dims>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
		}
		cudaEventRecord(stop_bcs);
        cudaEventSynchronize(stop_bcs);
        cudaEventElapsedTime(&bcs_time, start_bcs, stop_bcs);
		
		std::cout << "Benchmarking intermediate vars w/no diffusion" << std::endl;
        cudaEventRecord(start_intvar_nodiff);
		for (size_t il = 0; il < num_bench_iters; il++){
			ComputeIntermediateVariablesNoDiff<<<exec_grid_dims_intvar, intvar_block_dims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
		}
		cudaEventRecord(stop_intvar_nodiff);
        cudaEventSynchronize(stop_intvar_nodiff);
        cudaEventElapsedTime(&intvar_time_nodiff, start_intvar_nodiff, stop_intvar_nodiff);

		std::cout << "Benchmarking intermediate vars w/diffusion" << std::endl;
        cudaEventRecord(start_intvar);
		for (size_t il = 0; il < num_bench_iters; il++){
			ComputeIntermediateVariables<<<exec_grid_dims_intvar, intvar_block_dims>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
		}
		cudaEventRecord(stop_intvar);
        cudaEventSynchronize(stop_intvar);
        cudaEventElapsedTime(&intvar_time, start_intvar, stop_intvar);

		std::cout << "Benchmarking intermediate B.Cs microkernel implementation" << std::endl;
		// std::cout << "Current involves so much thread divergence that it kills performance" << std::endl;
        cudaEventRecord(start_intvar_bcs);
		for (size_t il = 0; il < num_bench_iters; il++){
			// Maybe this should be in a wrapper
			QintBdryFrontNoDiff<<<exec_grid_dims_qintbdry_front, qintbdry_front_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
			QintBdryLeftRightNoDiff<<<exec_grid_dims_qintbdry_leftright, qintbdry_leftright_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
			QintBdryTopBottomNoDiff<<<exec_grid_dims_qintbdry_topbottom, qintbdry_topbottom_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
			QintBdryFrontBottomNoDiff<<<exec_grid_dims_qintbdry_frontbottom, qintbdry_frontbottom_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
			QintBdryFrontRightNoDiff<<<exec_grid_dims_qintbdry_frontright, qintbdry_frontright_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
			QintBdryBottomRightNoDiff<<<exec_grid_dims_qintbdry_bottomright, qintbdry_bottomright_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
			checkCuda(cudaDeviceSynchronize());    

			QintBdryPBCs<<<exec_grid_dims_qintbdry_front, qintbdry_front_blockdims>>>(fluidvars, intvars, Nx, Ny, Nz);
			checkCuda(cudaDeviceSynchronize());    
		}
		cudaEventRecord(stop_intvar_bcs);
        cudaEventSynchronize(stop_intvar_bcs);
        cudaEventElapsedTime(&intvar_bcs_time, start_intvar_bcs, stop_intvar_bcs);

		std::cout << "Benchmarking data transfer" << std::endl;
		cudaEventRecord(start_d2h);
		for (size_t il = 0; il < num_bench_iters; il++){
      		cudaMemcpy(h_fluidvars, fluidvars, fluid_data_size, cudaMemcpyDeviceToHost);
		}
		cudaEventRecord(stop_d2h);
        cudaEventSynchronize(stop_d2h);
		cudaEventElapsedTime(&d2h_time, start_d2h, stop_d2h);

		doSomethingWithTransfer(h_fluidvars, Nx, Ny, Nz); // Don't want transfers optimized away

		bench_file << it << "," << fluid_time_nodiff / num_bench_iters << "," << fluid_time / num_bench_iters << "," << bcs_time / num_bench_iters << "," 
			<< intvar_time / num_bench_iters << "," << intvar_time_nodiff / num_bench_iters << "," << intvar_bcs_time / num_bench_iters << "," 
			<< d2h_time / num_bench_iters << ","
			<< fluid_threads << "," << fluid_threads << "," << intvar_threads << "," << intvar_threads << std::endl;
	}

	/* Free device data */ 
	checkCuda(cudaFree(fluidvars));
	checkCuda(cudaFree(intvars));

	checkCuda(cudaFree(x_grid));
	checkCuda(cudaFree(y_grid));
	checkCuda(cudaFree(z_grid));

    /* Free host data */
    bench_file.close();
	free(h_fluidvars);
	return 0;
}

// Don't want the migration optimized away
void doSomethingWithTransfer(float *h_fluidvars, int Nx, int Ny, int Nz){
	for (int k = 0; k < Nz; k++){
		for (int i = 0; i < Nx; i++){
			for (int j = 0; j < Ny; j++){
				h_fluidvars[Nx * Ny * k + Ny * i + j] = 0.0;
			}
		}
	}
}