#include <assert.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>

#include "../../include/on-device/kernels_od.cuh"
#include "../../include/on-device/initialize_od.cuh"
#include "../../include/on-device/kernels_od_intvar.cuh"
#include "../../include/on-device/utils/utils.hpp"

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
    parseInputFileDebug(inputs, "./imhd-cuda_profile.inp"); /* Write this again - it disappeared */
	
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

	float *fluidvars, *intvars;
	float *x_grid, *y_grid, *z_grid;

	int fluid_data_size = sizeof(float) * Nx * Ny * Nz;

	// Allocate global device memory
	checkCuda(cudaMalloc(&fluidvars, 8 * fluid_data_size));
	checkCuda(cudaMalloc(&intvars, 8 * fluid_data_size));

	checkCuda(cudaMalloc(&x_grid, sizeof(float) * Nx));
	checkCuda(cudaMalloc(&y_grid, sizeof(float) * Ny));
	checkCuda(cudaMalloc(&z_grid, sizeof(float) * Nz));

	/* FIX THIS LOL */
	dim3 grid_dimensions(SM_mult_x * numberOfSMs, SM_mult_y * numberOfSMs, SM_mult_z * numberOfSMs);
	dim3 block_dims_grid(32, 16, 2); // 1024 threads per block
	dim3 block_dims_init(8, 4, 4); // 256 < 923 threads per block
	dim3 block_dims_intvar(8, 8, 2); // 128 < 334 threads per block 
	dim3 block_dims_fluid(8, 8, 2); // 128 < 331 threads per block - based on register requirement of FluidAdvance + BCs kernels

	// Initialize
	InitializeGrid<<<grid_dimensions, block_dims_grid>>>(x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz,
															x_grid, y_grid, z_grid, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());

	ScrewPinch<<<grid_dimensions, block_dims_init>>>(fluidvars, J0, x_grid, y_grid, z_grid, Nx, Ny, Nz); // Screw-pinch
	checkCuda(cudaDeviceSynchronize());
	
	ComputeIntermediateVariables<<<grid_dimensions, block_dims_intvar>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
    checkCuda(cudaDeviceSynchronize());

	ComputeIntermediateVariablesBoundary<<<grid_dimensions, block_dims_intvar>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());

	// Timestep
    std::cout << "Evolving fluid interior" << std::endl; 
    FluidAdvanceLocal<<<grid_dimensions, block_dims_fluid>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
    checkCuda(cudaDeviceSynchronize());

    std::cout << "Evolving fluid boundaries" << std::endl; 
	BoundaryConditions<<<grid_dimensions, block_dims_fluid>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
    checkCuda(cudaDeviceSynchronize());
    
    ComputeIntermediateVariables<<<grid_dimensions, block_dims_intvar>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
    checkCuda(cudaDeviceSynchronize());

    ComputeIntermediateVariablesBoundary<<<grid_dimensions, block_dims_intvar>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
    checkCuda(cudaDeviceSynchronize());

	// Free device data 
	checkCuda(cudaFree(fluidvars));
	checkCuda(cudaFree(intvars));

	checkCuda(cudaFree(x_grid));
	checkCuda(cudaFree(y_grid));
	checkCuda(cudaFree(z_grid));

	return 0;
}