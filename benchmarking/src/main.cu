#include <assert.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <fstream>

#include "../../include/kernels_od.cuh"
#include "../../include/initialize_od.cuh"
#include "../../include/kernels_od_intvar.cuh"

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
	int Nx = atoi(argv[2]);
	int Ny = atoi(argv[3]);
	int Nz = atoi(argv[4]);
	int SM_mult_x = atoi(argv[5]);
	int SM_mult_y = atoi(argv[6]);
	int SM_mult_z = atoi(argv[7]);
	int num_threads_per_block_x = atoi(argv[8]);
	int num_threads_per_block_y = atoi(argv[9]);
	int num_threads_per_block_z = atoi(argv[10]);
	float J0 = atof(argv[11]);
	float D = atof(argv[12]);
	float x_min = atof(argv[13]);
	float x_max = atof(argv[14]);
	float y_min = atof(argv[15]);
	float y_max = atof(argv[16]);
	float z_min = atof(argv[17]);
	float z_max = atof(argv[18]);
	float dt = atof(argv[19]);

	float dx = (x_max - x_min) / (Nx - 1);
	float dy = (y_max - y_min) / (Ny - 1);
	float dz = (z_max - z_min) / (Nz - 1);

	/* Initialize device data */
	int deviceId;
	int numberOfSMs;

	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	float *fluidvar, *fluidvar_np1, *intvar;
	float *grid_x, *grid_y, *grid_z;

	int fluid_data_size = sizeof(float) * Nx * Ny * Nz;

	/* MALLOC TO DEVICE */
	checkCuda(cudaMalloc(&fluidvar, 8 * fluid_data_size));
	checkCuda(cudaMalloc(&fluidvar_np1, 8 * fluid_data_size));
	checkCuda(cudaMalloc(&intvar, 8 * fluid_data_size));

	checkCuda(cudaMalloc(&grid_x, sizeof(float) * Nx));
	checkCuda(cudaMalloc(&grid_y, sizeof(float) * Ny));
	checkCuda(cudaMalloc(&grid_z, sizeof(float) * Nz));

	dim3 grid_dimensions(SM_mult_x * numberOfSMs, SM_mult_y * numberOfSMs, SM_mult_z * numberOfSMs);
	
	// dim3 block_dimensions(num_threads_per_block_x, num_threads_per_block_y, num_threads_per_block_z);
	dim3 block_dims_grid(32, 16, 2); // 1024 threads per block
	dim3 block_dims_init(8, 4, 4); // 256 < 923 threads per block
	dim3 block_dims_intvar(8, 8, 2); // 128 < 334 threads per block 
	dim3 block_dims_fluid(8, 8, 2); // 128 < 331 threads per block - based on register requirement of FluidAdvance + BCs kernels

	InitializeGrid<<<grid_dimensions, block_dims_grid>>>(x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz,
															grid_x, grid_y, grid_z, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());

	InitialConditions<<<grid_dimensions, block_dims_init>>>(fluidvar, J0, grid_x, grid_y, grid_z, Nx, Ny, Nz); // Screw-pinch
	InitializeIntAndSwap<<<grid_dimensions, block_dims_init>>>(fluidvar_np1, intvar, Nx, Ny, Nz); // All 0.0
	checkCuda(cudaDeviceSynchronize());

	ComputeIntermediateVariables<<<grid_dimensions, block_dims_intvar>>>(fluidvar_np1, intvar, dt, dx, dy, dz, Nx, Ny, Nz);
	ComputeIntermediateVariablesBoundary<<<grid_dimensions, block_dims_intvar>>>(fluidvar_np1, intvar, dt, dx, dy, dz, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());
    
	// Benchmarking 
    cudaEvent_t start, stop, start_bcs, stop_bcs, start_swap, stop_swap;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start_bcs);
    cudaEventCreate(&stop_bcs);
    cudaEventCreate(&start_swap);
    cudaEventCreate(&stop_swap);

    float fluid_time = 0.0, bcs_time = 0.0, swap_time = 0.0; 

    // For recording the benchmarking data
    std::ofstream bench_file;
    bench_file.open("../data/benchNxNyNz_" + std::to_string(Nx) + std::to_string(Ny) + std::to_string(Nz) + ".csv");

    bench_file << "it, fluid_time (ms), bcs_time (ms), swap_time (ms), numblocks_x, numblocks_y, numblocks_z, numthreadsper_x, numthreadsper_y, numthreadsper_z" << std::endl; 

	/* Simulation loop */
	size_t num_bench_iters = 1; 
	for (size_t it = 0; it < Nt; it++){
		std::cout << "Starting iteration " << it << std::endl;

		std::cout << "Evolving fluid interior and boundary" << std::endl; 
		/* DO 1000 REPS of KERNEL B/W RECORDING */
        cudaEventRecord(start);
		for (size_t il = 0; il < num_bench_iters; il++){
			FluidAdvance<<<grid_dimensions, block_dims_fluid>>>(fluidvar_np1, fluidvar, intvar, D, dt, dx, dy, dz, Nx, Ny, Nz);
		}
		cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&fluid_time, start, stop);

        cudaEventRecord(start_bcs);
        BoundaryConditions<<<grid_dimensions, block_dims_fluid>>>(fluidvar_np1, fluidvar, intvar, D, dt, dx, dy, dz, Nx, Ny, Nz);
		cudaEventRecord(stop_bcs);
        cudaEventSynchronize(stop_bcs);
        cudaEventElapsedTime(&bcs_time, start_bcs, stop_bcs);

        // checkCuda(cudaDeviceSynchronize());
		
		// Transfer future timestep data to current timestep in order to avoid race conditions
		std::cout << "Swapping future timestep to current" << std::endl;
        cudaEventRecord(start_swap);
		SwapSimData<<<grid_dimensions, block_dims_intvar>>>(fluidvar, fluidvar_np1, Nx, Ny, Nz);
		ComputeIntermediateVariables<<<grid_dimensions, block_dims_intvar>>>(fluidvar_np1, intvar, dt, dx, dy, dz, Nx, Ny, Nz);
		ComputeIntermediateVariablesBoundary<<<grid_dimensions, block_dims_intvar>>>(fluidvar_np1, intvar, dt, dx, dy, dz, Nx, Ny, Nz);
		cudaEventRecord(stop_swap);
        cudaEventSynchronize(stop_swap);
        cudaEventElapsedTime(&swap_time, start_swap, stop_swap);
        // checkCuda(cudaDeviceSynchronize());
        bench_file << it << "," << fluid_time << "," << bcs_time << "," << swap_time << "," << SM_mult_x * numberOfSMs << "," << SM_mult_y * numberOfSMs << "," << 
            SM_mult_z * numberOfSMs << "," << num_threads_per_block_x << "," << num_threads_per_block_y << "," << num_threads_per_block_z << std::endl;
	}

	/* Free device data */ 
	checkCuda(cudaFree(fluidvar));
	checkCuda(cudaFree(fluidvar_np1));
	checkCuda(cudaFree(intvar));

	checkCuda(cudaFree(grid_x));
	checkCuda(cudaFree(grid_y));
	checkCuda(cudaFree(grid_z));

    /* Free host data */
    bench_file.close();
	return 0;
}