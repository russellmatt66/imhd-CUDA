#include <assert.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>

#include "../include/kernels_od.cuh"
#include "../include/initialize_od.cuh"
#include "../include/gds.cuh"

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

	/* Simulation loop */
	for (size_t it = 0; it < Nt; it++){
		std::cout << "Starting iteration " << it << std::endl;

		/* Write data out - use GPUDirect Storage (GDS) */
		std::cout << "Writing data out with GDS" << std::endl;
		writeFluidDataGDS(rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx * Ny * Nz, it);

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
		checkCuda(cudaDeviceSynchronize());
		
		/* Transfer future timestep data to current timestep in order to avoid race conditions */
		std::cout << "Swapping future timestep to current" << std::endl;
		SwapSimData<<<grid_dimensions, block_dimensions>>>(rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, 
															rho_np1, rhovx_np1, rhovy_np1, rhovz_np1, Bx_np1, By_np1, Bz_np1, e_np1,
															Nx, Ny, Nz);
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

	return 0;
}