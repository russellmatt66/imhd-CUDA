#include <assert.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>

#include "../../include/on-device/kernels_od.cuh"
#include "../../include/on-device/kernels_fluidbcs.cuh"
#include "../../include/on-device/initialize_od.cuh"
#include "../../include/on-device/kernels_od_intvar.cuh"
#include "../../include/on-device/kernels_intvarbcs.cuh"
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
    std::vector<float> inputs (25, 0.0);
    parseInputFileDebug(inputs, "./debug.inp"); 
	
	for (int i = 0; i < inputs.size(); i++){
		std::cout << "inputs[" << i << "] = " << inputs[i] << std::endl; 
	}

    int Nt = int(inputs[0]);
	int Nx = int(inputs[1]);
	int Ny = int(inputs[2]);
	int Nz = int(inputs[3]);

	float J0 = inputs[4];
	float D = inputs[5];
	float x_min = inputs[6];
	float x_max = inputs[7];
	float y_min = inputs[8];
	float y_max = inputs[9];
	float z_min = inputs[10];
	float z_max = inputs[11];
	float dt = inputs[12];

	int grid_xthreads=int(inputs[13]);
	int grid_ythreads=int(inputs[14]);
	int grid_zthreads=int(inputs[15]);

	int init_xthreads=int(inputs[16]);
	int init_ythreads=int(inputs[17]);
	int init_zthreads=int(inputs[18]);

	int intvar_xthreads=int(inputs[19]);
	int intvar_ythreads=int(inputs[20]);
	int intvar_zthreads=int(inputs[21]);

	int fluid_xthreads=int(inputs[22]);
	int fluid_ythreads=int(inputs[23]);
	int fluid_zthreads=int(inputs[24]);

	int SM_mult_intvar_x=int(inputs[25]);
	int SM_mult_intvar_y=int(inputs[26]);
	int SM_mult_intvar_z=int(inputs[27]);

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

	// Different execution configurations are needed to varying register pressures
	dim3 exec_grid_dim(numberOfSMs, numberOfSMs, numberOfSMs);
	dim3 exec_grid_dim_intvar(SM_mult_intvar_x * numberOfSMs, SM_mult_intvar_y * numberOfSMs, SM_mult_intvar_z * numberOfSMs);

	dim3 block_dims_grid(grid_xthreads, grid_ythreads, grid_zthreads); // 1024 threads per block
	dim3 block_dims_init(init_xthreads, init_ythreads, init_zthreads); // 256 < 923 threads per block
	dim3 block_dims_intvar(intvar_xthreads, intvar_ythreads, intvar_zthreads); 
	dim3 block_dims_fluid(fluid_xthreads, fluid_ythreads, fluid_zthreads); 

	// Initialize
	InitializeGrid<<<exec_grid_dim, block_dims_grid>>>(x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz,
															x_grid, y_grid, z_grid, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());

	ScrewPinchStride<<<exec_grid_dim, block_dims_init>>>(fluidvars, J0, x_grid, y_grid, z_grid, Nx, Ny, Nz); // Screw-pinch
	checkCuda(cudaDeviceSynchronize());
	
	ComputeIntermediateVariablesStride<<<exec_grid_dim_intvar, block_dims_intvar>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
    checkCuda(cudaDeviceSynchronize());

	ComputeIntermediateVariablesBoundary<<<exec_grid_dim_intvar, block_dims_intvar>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());

	// Timestep
	for (int it = 1; it < Nt; it++){
		std::cout << "Taking timestep " << it << std::endl;
		std::cout << "Evolving fluid interior" << std::endl; 
		FluidAdvanceLocal<<<exec_grid_dim, block_dims_fluid>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
		checkCuda(cudaDeviceSynchronize());

		std::cout << "Evolving fluid boundaries" << std::endl; 
		BoundaryConditions<<<exec_grid_dim, block_dims_fluid>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
		checkCuda(cudaDeviceSynchronize());
		
		ComputeIntermediateVariablesStride<<<exec_grid_dim_intvar, block_dims_intvar>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
		checkCuda(cudaDeviceSynchronize());

		ComputeIntermediateVariablesBoundary<<<exec_grid_dim_intvar, block_dims_intvar>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
		checkCuda(cudaDeviceSynchronize());
	}

	// Free device data 
	checkCuda(cudaFree(fluidvars));
	checkCuda(cudaFree(intvars));

	checkCuda(cudaFree(x_grid));
	checkCuda(cudaFree(y_grid));
	checkCuda(cudaFree(z_grid));

	return 0;
}