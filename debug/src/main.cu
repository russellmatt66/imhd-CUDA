#include <assert.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>

#include "../../include/kernels_od.cuh"
#include "../../include/initialize_od.cuh"
#include "../../include/kernels_od_intvar.cuh"
#include "../../include/utils.hpp"
#include "../../include/utils.cuh"

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
    std::vector<float> inputs (27, 0.0);
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
	int write_rho = inputs[19]; // Data volume gets very large
	int write_rhovx = inputs[20];
	int write_rhovy = inputs[21];
	int write_rhovz = inputs[22];
	int write_Bx = inputs[23];
	int write_By = inputs[24];
	int write_Bz = inputs[25];
	int write_e = inputs[26];

	float dx = (x_max - x_min) / (Nx - 1);
	float dy = (y_max - y_min) / (Ny - 1);
	float dz = (z_max - z_min) / (Nz - 1);

	// Initialize device data
	int deviceId;
	int numberOfSMs;

	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	int* to_write_or_not;
	to_write_or_not = (int*)malloc(8 * sizeof(int));

	for (int i = 0; i < 8; i++){ /* COULD USE AN INT FOR THIS */
		to_write_or_not[i] = atoi(argv[21 + i]);
	}

	float *fluidvar, *fluidvar_np1, *intvar;
	float *grid_x, *grid_y, *grid_z;

	int cube_size = Nx * Ny * Nz;
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

	std::cout << "Initializing screw-pinch" << std::endl;
	InitialConditions<<<grid_dimensions, block_dims_init>>>(fluidvar, J0, grid_x, grid_y, grid_z, Nx, Ny, Nz); // Screw-pinch
	InitializeIntAndSwap<<<grid_dimensions, block_dims_init>>>(fluidvar_np1, intvar, Nx, Ny, Nz); // All 0.0
	checkCuda(cudaDeviceSynchronize());

	std::cout << "Initial computation of intermediate variables" << std::endl;
	ComputeIntermediateVariables<<<grid_dimensions, block_dims_intvar>>>(fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz);
	ComputeIntermediateVariablesBoundary<<<grid_dimensions, block_dims_intvar>>>(fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());

	float *h_rho, *h_rhovx, *h_rhovy, *h_rhovz, *h_Bx, *h_By, *h_Bz, *h_e;

	h_rho = (float*)malloc(fluid_data_size);
	h_rhovx = (float*)malloc(fluid_data_size);
	h_rhovy = (float*)malloc(fluid_data_size);
	h_rhovz = (float*)malloc(fluid_data_size);
	h_Bx = (float*)malloc(fluid_data_size);
	h_By = (float*)malloc(fluid_data_size);
	h_Bz = (float*)malloc(fluid_data_size);
	h_e = (float*)malloc(fluid_data_size);

	for (size_t ih = 0; ih < 8; ih++){
		if (!to_write_or_not[ih]){ // No need for the host memory if it's not being written out
			switch (ih)
			{
			case 0:
				free(h_rho);
				break;
			case 1:
				free(h_rhovx);
				break;
			case 2:
				free(h_rhovy);
				break;			
			case 3:
				free(h_rhovz);
				break;			
			case 4:
				free(h_Bx);
				break;			
			case 5:
				free(h_By);
				break;			
			case 6:
				free(h_Bz);
				break;			
			case 7:
				free(h_e);
				break;			
			default:
				break;
			}
		}
	}

	/* Simulation loop */
	for (size_t it = 0; it < Nt; it++){
		std::cout << "Starting iteration " << it << std::endl;

		/* Compute interior and boundaries*/
		std::cout << "Evolving fluid interior and boundary" << std::endl; 
		FluidAdvance<<<grid_dimensions, block_dims_fluid>>>(fluidvar_np1, fluidvar, intvar, D, dt, dx, dy, dz, Nx, Ny, Nz);
		BoundaryConditions<<<grid_dimensions, block_dims_fluid>>>(fluidvar_np1, fluidvar, intvar, D, dt, dx, dy, dz, Nx, Ny, Nz);
	
		std::cout << "Writing fluid data to host and printing intermediate variables" << std::endl;
		// Data volume scales very fast w/problem size, don't want to always write everything out 
		for (size_t iv = 0; iv < 8; iv++){ 
			if (to_write_or_not[iv]){  
				switch (iv)
				{
				case 0:
					cudaMemcpy(h_rho, fluidvar, fluid_data_size, cudaMemcpyDeviceToHost);
					break;
				case 1:
					// cudaMemcpy(h_rhovx, fluidvar + fluid_data_size, fluid_data_size, cudaMemcpyDeviceToHost);
					cudaMemcpy(h_rhovx, fluidvar + cube_size, fluid_data_size, cudaMemcpyDeviceToHost);
					break;
				case 2:
					// cudaMemcpy(h_rhovy, fluidvar + 2 * fluid_data_size, fluid_data_size, cudaMemcpyDeviceToHost);
					cudaMemcpy(h_rhovy, fluidvar + 2 * cube_size, fluid_data_size, cudaMemcpyDeviceToHost);
					break;
				case 3:
					// cudaMemcpy(h_rhovz, fluidvar + 3 * fluid_data_size, fluid_data_size, cudaMemcpyDeviceToHost);
					cudaMemcpy(h_rhovz, fluidvar + 3 * cube_size, fluid_data_size, cudaMemcpyDeviceToHost);
					break;			
				case 4:
					// cudaMemcpy(h_Bx, fluidvar + 4 * fluid_data_size, fluid_data_size, cudaMemcpyDeviceToHost);
					cudaMemcpy(h_Bx, fluidvar + 4 * cube_size, fluid_data_size, cudaMemcpyDeviceToHost);
					break;				
				case 5:
					// cudaMemcpy(h_By, fluidvar + 5 * fluid_data_size, fluid_data_size, cudaMemcpyDeviceToHost);
					cudaMemcpy(h_By, fluidvar + 5 * cube_size, fluid_data_size, cudaMemcpyDeviceToHost);
					break;				
				case 6:
					// cudaMemcpy(h_Bz, fluidvar + 6 * fluid_data_size, fluid_data_size, cudaMemcpyDeviceToHost);
					cudaMemcpy(h_Bz, fluidvar + 6 * cube_size, fluid_data_size, cudaMemcpyDeviceToHost);
					break;				
				case 7:
					// cudaMemcpy(h_e, fluidvar + 7 * fluid_data_size, fluid_data_size, cudaMemcpyDeviceToHost);
					cudaMemcpy(h_e, fluidvar + 7 * cube_size, fluid_data_size, cudaMemcpyDeviceToHost);
					break;				
				default:
					break;
				}
			}
		}
		if (it == 2){
			PrintIntvar<<<grid_dimensions, block_dims_intvar>>>(intvar, fluidvar, Nx, Ny, Nz);
		}
		checkCuda(cudaDeviceSynchronize());
		
		// Transfer future timestep data to current timestep in order to avoid race conditions
		std::cout << "Swapping future timestep to current, and printing fluidvars" << std::endl;
		SwapSimData<<<grid_dimensions, block_dims_intvar>>>(fluidvar, fluidvar_np1, Nx, Ny, Nz);
		ComputeIntermediateVariables<<<grid_dimensions, block_dims_intvar>>>(fluidvar_np1, intvar, dt, dx, dy, dz, Nx, Ny, Nz);
		ComputeIntermediateVariablesBoundary<<<grid_dimensions, block_dims_intvar>>>(fluidvar_np1, intvar, dt, dx, dy, dz, Nx, Ny, Nz);
		
		if (it == 2) { 		
			PrintFluidvar<<<grid_dimensions, block_dims_fluid>>>(fluidvar, Nx, Ny, Nz);
		}
		checkCuda(cudaDeviceSynchronize());
	}

	/* Free device data */ 
	checkCuda(cudaFree(fluidvar));
	checkCuda(cudaFree(fluidvar_np1));
	checkCuda(cudaFree(intvar));

	checkCuda(cudaFree(grid_x));
	checkCuda(cudaFree(grid_y));
	checkCuda(cudaFree(grid_z));

	/* Free host data */
	for (size_t ih = 0; ih < 8; ih++){
		if (to_write_or_not[ih]){ // Don't forget to free the rest of the host buffers 
			switch (ih)
			{
			case 0:
				free(h_rho);
				break;
			case 1:
				free(h_rhovx);
				break;
			case 2:
				free(h_rhovy);
				break;			
			case 3:
				free(h_rhovz);
				break;			
			case 4:
				free(h_Bx);
				break;			
			case 5:
				free(h_By);
				break;			
			case 6:
				free(h_Bz);
				break;			
			case 7:
				free(h_e);
				break;			
			default:
				break;
			}
		}
	}
	free(to_write_or_not);
	return 0;
}