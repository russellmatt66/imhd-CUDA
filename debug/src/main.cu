#include <assert.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>

#include "../../include/kernels_od.cuh"
#include "../../include/initialize_od.cuh"
#include "../../include/kernels_od_intvar.cuh"
#include "../../include/utils.hpp"

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
    std::vector<float> inputs (19, 0.0);
    parseInputFileDebug(inputs, "./debug.inp");
	
	for (int i = 0; i < inputs.size(); i++){
		std::cout << "inputs[" << i << "] = " << inputs[i] << std::endl; 
	}

    int Nt = int(inputs[0]);
	int Nx = int(inputs[1]);
	int Ny = int(inputs[2]);
	int Nz = int(inputs[3]);
	int SM_mult_x = int(inputs[4]);
	int SM_mult_y = int(inputs[5]);
	int SM_mult_z = int(inputs[6]);
	int num_threads_per_block_x = int(inputs[7]);
	int num_threads_per_block_y = int(inputs[8]);
	int num_threads_per_block_z = int(inputs[9]);
	float J0 = inputs[10];
	float D = inputs[11];
	float x_min = inputs[12];
	float x_max = inputs[13];
	float y_min = inputs[14];
	float y_max = inputs[15];
	float z_min = inputs[16];
	float z_max = inputs[17];
	float dt = inputs[18];

	float dx = (x_max - x_min) / (Nx - 1);
	float dy = (y_max - y_min) / (Ny - 1);
	float dz = (z_max - z_min) / (Nz - 1);

	// Initialize device data
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
	dim3 block_dims_init(8, 4, 4); // 512 < 923 threads per block
	dim3 block_dims_intvar(8, 8, 2); // 256 < 334 threads per block 
	dim3 block_dims_fluid(8, 8, 2); // 256 < 331 threads per block - based on register requirement of FluidAdvance + BCs kernels

	InitializeGrid<<<grid_dimensions, block_dims_grid>>>(x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz,
															grid_x, grid_y, grid_z, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());

	InitialConditions<<<grid_dimensions, block_dims_init>>>(fluidvar, J0, grid_x, grid_y, grid_z, Nx, Ny, Nz); // Screw-pinch
	InitializeIntAndSwap<<<grid_dimensions, block_dims_init>>>(fluidvar_np1, intvar, Nx, Ny, Nz); // All 0.0
	checkCuda(cudaDeviceSynchronize());

	ComputeIntermediateVariables<<<grid_dimensions, block_dims_intvar>>>(fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz);
	ComputeIntermediateVariablesBoundary<<<grid_dimensions, block_dims_intvar>>>(fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());

	/* Simulation loop */
	for (size_t it = 0; it < Nt; it++){
		std::cout << "Starting iteration " << it << std::endl;

		/* Compute interior and boundaries*/
		std::cout << "Evolving fluid interior and boundary" << std::endl; 
		FluidAdvance<<<grid_dimensions, block_dims_fluid>>>(fluidvar_np1, fluidvar, intvar, D, dt, dx, dy, dz, Nx, Ny, Nz);
		BoundaryConditions<<<grid_dimensions, block_dims_fluid>>>(fluidvar_np1, fluidvar, intvar, D, dt, dx, dy, dz, Nx, Ny, Nz);
	
		std::cout << "Writing fluid data to host" << std::endl;
		checkCuda(cudaDeviceSynchronize());
		
		// Transfer future timestep data to current timestep in order to avoid race conditions
		std::cout << "Swapping future timestep to current" << std::endl;
		SwapSimData<<<grid_dimensions, block_dims_intvar>>>(fluidvar, fluidvar_np1, Nx, Ny, Nz);
		ComputeIntermediateVariables<<<grid_dimensions, block_dims_intvar>>>(fluidvar_np1, intvar, dt, dx, dy, dz, Nx, Ny, Nz);
		ComputeIntermediateVariablesBoundary<<<grid_dimensions, block_dims_intvar>>>(fluidvar_np1, intvar, dt, dx, dy, dz, Nx, Ny, Nz);
		checkCuda(cudaDeviceSynchronize());
	}

	/* Free device data */ 
	checkCuda(cudaFree(fluidvar));
	checkCuda(cudaFree(fluidvar_np1));
	checkCuda(cudaFree(intvar));

	checkCuda(cudaFree(grid_x));
	checkCuda(cudaFree(grid_y));
	checkCuda(cudaFree(grid_z));

	return 0;
}