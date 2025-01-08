#include <assert.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <fstream>

#include "../../include/on-device/kernels_od.cuh"
#include "../../include/on-device/initialize_od.cuh"
#include "../../include/on-device/kernels_od_intvar.cuh"

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
	int fluidblockdims_ythreads = atoi(argv[35]);
	int fluidblockdims_zthreads = atoi(argv[26]);

	float dx = (x_max - x_min) / (Nx - 1);
	float dy = (y_max - y_min) / (Ny - 1);
	float dz = (z_max - z_min) / (Nz - 1);

	int fluid_threads = fluidblockdims_xthreads * fluidblockdims_ythreads * fluidblockdims_zthreads;
	int intvar_threads = intvarblockdims_xthreads * intvarblockdims_ythreads * intvarblockdims_zthreads;

	// Initialize device data 
	int deviceId;
	int numberOfSMs;

	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	float *fluidvars, *intvars;
	float *x_grid, *y_grid, *z_grid;

	int fluid_data_size = sizeof(float) * Nx * Ny * Nz;

	// Get that data on device global memory 
	checkCuda(cudaMalloc(&fluidvars, 8 * fluid_data_size));
	checkCuda(cudaMalloc(&intvars, 8 * fluid_data_size));

	checkCuda(cudaMalloc(&x_grid, sizeof(float) * Nx));
	checkCuda(cudaMalloc(&y_grid, sizeof(float) * Ny));
	checkCuda(cudaMalloc(&z_grid, sizeof(float) * Nz));

	dim3 exec_grid_dims(numberOfSMs, numberOfSMs, numberOfSMs);
	dim3 mesh_block_dims(meshblockdims_xthreads, meshblockdims_ythreads, meshblockdims_zthreads);
	dim3 init_block_dims(initblockdims_xthreads, initblockdims_ythreads, initblockdims_zthreads);
	dim3 intvar_block_dims(intvarblockdims_xthreads, intvarblockdims_ythreads, intvarblockdims_zthreads);
	dim3 fluid_block_dims(fluidblockdims_xthreads, fluidblockdims_ythreads, fluidblockdims_zthreads);

	InitializeGrid<<<exec_grid_dims, mesh_block_dims>>>(x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz, x_grid, y_grid, z_grid, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());

	ScrewPinch<<<exec_grid_dims, init_block_dims>>>(fluidvars, J0, x_grid, y_grid, z_grid, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());

	InitializeIntvars<<<exec_grid_dims, intvar_block_dims>>>(intvars, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());

	ComputeIntermediateVariables<<<exec_grid_dims, intvar_block_dims>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());    
	
	ComputeIntermediateVariablesBoundary<<<exec_grid_dims, intvar_block_dims>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
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

    float fluid_time = 0.0, bcs_time = 0.0, intvar_time = 0.0, intvar_bcs_time = 0.0; 

    // For recording the benchmarking data
    std::ofstream bench_file;
    bench_file.open("../data/benchNxNyNz_" + std::to_string(Nx) + std::to_string(Ny) + std::to_string(Nz) + ".csv");

    bench_file << "it, fluid_time (ms), bcs_time (ms), intvar_time (ms), intvar_bcs_time (ms), fluid_threads, bcs_threads, intvar_threads, intvar_bcs_threads" << std::endl; 

	// Simulation loop
	for (size_t it = 0; it < Nt; it++){
		std::cout << "Starting iteration " << it << std::endl;

		std::cout << "Benchmarking fluid interior" << std::endl; 
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
		
		std::cout << "Benchmarking intermediate vars" << std::endl;
        cudaEventRecord(start_intvar);
		for (size_t il = 0; il < num_bench_iters; il++){
			ComputeIntermediateVariables<<<exec_grid_dims, intvar_block_dims>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
		}
		cudaEventRecord(stop_intvar);
        cudaEventSynchronize(stop_intvar);
        cudaEventElapsedTime(&intvar_time, start_intvar, stop_intvar);

		std::cout << "Benchmarking intermediate B.Cs" << std::endl;
        cudaEventRecord(start_intvar_bcs);
		for (size_t il = 0; il < num_bench_iters; il++){
			ComputeIntermediateVariablesBoundary<<<exec_grid_dims, intvar_block_dims>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
		}
		cudaEventRecord(stop_intvar_bcs);
        cudaEventSynchronize(stop_intvar_bcs);
        cudaEventElapsedTime(&intvar_bcs_time, start_intvar_bcs, stop_intvar_bcs);

		bench_file << it << "," << fluid_time << "," << bcs_time << "," << intvar_time << "," << intvar_bcs_time << "," 
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
	return 0;
}