#include <assert.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>

#include "../include/kernels_od.cuh"
#include "../include/initialize_od.cuh"
#include "../include/gds.cuh"
#include "../include/utils.hpp"

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
	int write_rho = atoi(argv[20]); // Data volume gets very large
	int write_rhovx = atoi(argv[21]);
	int write_rhovy = atoi(argv[22]);
	int write_rhovz = atoi(argv[23]);
	int write_Bx = atoi(argv[24]);
	int write_By = atoi(argv[25]);
	int write_Bz = atoi(argv[26]);
	int write_e = atoi(argv[27]);

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

	for (int i = 0; i < 8; i++){ /* COULD USE A CHAR FOR THIS */
		to_write_or_not[i] = atoi(argv[20 + i]);
	}

	float *rho, *rhov_x, *rhov_y, *rhov_z, *Bx, *By, *Bz, *e;
	float *rho_np1, *rhovx_np1, *rhovy_np1, *rhovz_np1, *Bx_np1, *By_np1, *Bz_np1, *e_np1;
	float *rho_int, *rhovx_int, *rhovy_int, *rhovz_int, *Bx_int, *By_int, *Bz_int, *e_int;
	float *grid_x, *grid_y, *grid_z;

	int fluid_data_size = sizeof(float) * Nx * Ny * Nz;

	/* MALLOC TO DEVICE */
	checkCuda(cudaMalloc(&rho, fluid_data_size));
	checkCuda(cudaMalloc(&rhov_x, fluid_data_size));
	checkCuda(cudaMalloc(&rhov_y, fluid_data_size));
	checkCuda(cudaMalloc(&rhov_z, fluid_data_size));
	checkCuda(cudaMalloc(&Bx, fluid_data_size));
	checkCuda(cudaMalloc(&By, fluid_data_size));
	checkCuda(cudaMalloc(&Bz, fluid_data_size));
	checkCuda(cudaMalloc(&e, fluid_data_size));

	checkCuda(cudaMalloc(&rho_np1, fluid_data_size));
	checkCuda(cudaMalloc(&rhovx_np1, fluid_data_size));
	checkCuda(cudaMalloc(&rhovy_np1, fluid_data_size));
	checkCuda(cudaMalloc(&rhovz_np1, fluid_data_size));
	checkCuda(cudaMalloc(&Bx_np1, fluid_data_size));
	checkCuda(cudaMalloc(&By_np1, fluid_data_size));
	checkCuda(cudaMalloc(&Bz_np1, fluid_data_size));
	checkCuda(cudaMalloc(&e_np1, fluid_data_size));

	checkCuda(cudaMalloc(&rho_int, fluid_data_size));
	checkCuda(cudaMalloc(&rhovx_int, fluid_data_size));
	checkCuda(cudaMalloc(&rhovy_int, fluid_data_size));
	checkCuda(cudaMalloc(&rhovz_int, fluid_data_size));
	checkCuda(cudaMalloc(&Bx_int, fluid_data_size));
	checkCuda(cudaMalloc(&By_int, fluid_data_size));
	checkCuda(cudaMalloc(&Bz_int, fluid_data_size));
	checkCuda(cudaMalloc(&e_int, fluid_data_size));

	checkCuda(cudaMalloc(&grid_x, sizeof(float) * Nx));
	checkCuda(cudaMalloc(&grid_y, sizeof(float) * Ny));
	checkCuda(cudaMalloc(&grid_z, sizeof(float) * Nz));

	dim3 grid_dimensions(SM_mult_x * numberOfSMs, SM_mult_y * numberOfSMs, SM_mult_z * numberOfSMs);
	dim3 block_dimensions(num_threads_per_block_x, num_threads_per_block_y, num_threads_per_block_z);

	InitializeGrid<<<grid_dimensions, block_dimensions>>>(x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz,
															grid_x, grid_y, grid_z, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());

	InitialConditions<<<grid_dimensions, block_dimensions>>>(rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, 
																J0, grid_x, grid_y, grid_z, Nx, Ny, Nz); // Screw-pinch
	InitializeIntAndSwap<<<grid_dimensions, block_dimensions>>>(rho_np1, rhovx_np1, rhovy_np1, rhovz_np1, Bx_np1, By_np1, Bz_np1, e_np1,
																rho_int, rhovx_int, rhovy_int, rhovz_int, Bx_int, By_int, Bz_int, e_int, 
																Nx, Ny, Nz); // All 0.0
	checkCuda(cudaDeviceSynchronize());

    // Prepare host data for writing out
	std::vector<std::string> fluid_data_files (8); // 8 is the number of threads I'm going with
    std::string base_file = "../data/rho/";
    for (size_t i = 0; i < fluid_data_files.size(); i++){
        fluid_data_files[i] = base_file + std::to_string(i) + ".csv";
    }   

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
		FluidAdvance<<<grid_dimensions, block_dimensions>>>(rho_np1, rhovx_np1, rhovy_np1, rhovz_np1, Bx_np1, By_np1, Bz_np1, e_np1, 
																rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, 
																rho_int, rhovx_int, rhovy_int, rhovz_int, Bx_int, By_int, Bz_int, e_int, 
																D, dt, dx, dy, dz, Nx, Ny, Nz);
		BoundaryConditions<<<grid_dimensions, block_dimensions>>>(rho_np1, rhovx_np1, rhovy_np1, rhovz_np1, Bx_np1, By_np1, Bz_np1, e_np1,
																	rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, 
																	rho_int, rhovx_int, rhovy_int, rhovz_int, Bx_int, By_int, Bz_int, e_int, 
																	D, dt, dx, dy, dz, Nx, Ny, Nz);
	
		std::cout << "Writing fluid data to host" << std::endl;
		// Data volume scales very fast w/problem size, don't want to always write everything out 
		for (size_t iv = 0; iv < 8; iv++){ 
			if (to_write_or_not[iv]){  
				switch (iv)
				{
				case 0:
					cudaMemcpy(h_rho, rho, fluid_data_size, cudaMemcpyDeviceToHost);
					break;
				case 1:
					cudaMemcpy(h_rhovx, rhov_x, fluid_data_size, cudaMemcpyDeviceToHost);
					break;
				case 2:
					cudaMemcpy(h_rhovy, rhov_y, fluid_data_size, cudaMemcpyDeviceToHost);
					break;
				case 3:
					cudaMemcpy(h_rhovz, rhov_z, fluid_data_size, cudaMemcpyDeviceToHost);
					break;			
				case 4:
					cudaMemcpy(h_Bx, Bx, fluid_data_size, cudaMemcpyDeviceToHost);
					break;				
				case 5:
					cudaMemcpy(h_By, By, fluid_data_size, cudaMemcpyDeviceToHost);
					break;				
				case 6:
					cudaMemcpy(h_Bz, Bz, fluid_data_size, cudaMemcpyDeviceToHost);
					break;				
				case 7:
					cudaMemcpy(h_e, e, fluid_data_size, cudaMemcpyDeviceToHost);
					break;				
				default:
					break;
				}
			}
		}
		checkCuda(cudaDeviceSynchronize());
		
		// Transfer future timestep data to current timestep in order to avoid race conditions
		std::cout << "Swapping future timestep to current" << std::endl;
		SwapSimData<<<grid_dimensions, block_dimensions>>>(rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, 
															rho_np1, rhovx_np1, rhovy_np1, rhovz_np1, Bx_np1, By_np1, Bz_np1, e_np1,
															Nx, Ny, Nz);

		// Split the Device2Host and Host2Storage writes up to reduce synchro barriers
		std::cout << "Writing host data to storage" << std::endl; 
		for (size_t iv = 0; iv < 8; iv++){ 
			if (to_write_or_not[iv]){ 
				base_file = getNewBaseDataLoc(iv);
				for (size_t i = 0; i < fluid_data_files.size(); i++){
					fluid_data_files[i] = base_file + std::to_string(i) + ".csv";
				}  
				switch (iv)
				{
				case 0:
					writeFluidVars(fluid_data_files, h_rho, Nx, Ny, Nz);					
					break;
				case 1:
					writeFluidVars(fluid_data_files, h_rhovx, Nx, Ny, Nz);					
					break;
				case 2:
					writeFluidVars(fluid_data_files, h_rhovy, Nx, Ny, Nz);					
					break;
				case 3:
					writeFluidVars(fluid_data_files, h_rhovz, Nx, Ny, Nz);					
					break;			
				case 4:
					writeFluidVars(fluid_data_files, h_Bx, Nx, Ny, Nz);					
					break;				
				case 5:
					writeFluidVars(fluid_data_files, h_By, Nx, Ny, Nz);					
					break;				
				case 6:
					writeFluidVars(fluid_data_files, h_Bz, Nx, Ny, Nz);					
					break;				
				case 7:
					writeFluidVars(fluid_data_files, h_e, Nx, Ny, Nz);					
					break;				
				default:
					break;
				}
			}
		}
		checkCuda(cudaDeviceSynchronize());
	}

	/* Free device data */ 
	checkCuda(cudaFree(rho));
	checkCuda(cudaFree(rhov_x));
	checkCuda(cudaFree(rhov_y));
	checkCuda(cudaFree(rhov_z));
	checkCuda(cudaFree(Bx));
	checkCuda(cudaFree(By));
	checkCuda(cudaFree(Bz));
	checkCuda(cudaFree(e));

	checkCuda(cudaFree(rho_np1));
	checkCuda(cudaFree(rhovx_np1));
	checkCuda(cudaFree(rhovy_np1));
	checkCuda(cudaFree(rhovz_np1));
	checkCuda(cudaFree(Bx_np1));
	checkCuda(cudaFree(By_np1));
	checkCuda(cudaFree(Bz_np1));
	checkCuda(cudaFree(e_np1));

	checkCuda(cudaFree(rho_int));
	checkCuda(cudaFree(rhovx_int));
	checkCuda(cudaFree(rhovy_int));
	checkCuda(cudaFree(rhovz_int));
	checkCuda(cudaFree(Bx_int));
	checkCuda(cudaFree(By_int));
	checkCuda(cudaFree(Bz_int));
	checkCuda(cudaFree(e_int));

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