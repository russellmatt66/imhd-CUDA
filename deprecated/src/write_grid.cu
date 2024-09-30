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

/* Don't have the space to store, and then write, entire grid out during main simulation */
int main(int argc, char* argv[]){
    /* Just write the grid out */
	int Nx = atoi(argv[1]);
	int Ny = atoi(argv[2]);
	int Nz = atoi(argv[3]);
	int SM_mult_x = atoi(argv[4]);
	int SM_mult_y = atoi(argv[5]);
	int SM_mult_z = atoi(argv[6]);
	int num_threads_per_block_x = atoi(argv[7]);
	int num_threads_per_block_y = atoi(argv[8]);
	int num_threads_per_block_z = atoi(argv[9]);	
    float x_min = atof(argv[10]);
	float x_max = atof(argv[11]);
	float y_min = atof(argv[12]);
	float y_max = atof(argv[13]);
	float z_min = atof(argv[14]);
	float z_max = atof(argv[15]);

    std::cout << "Grid dimensions" << std::endl;
    std::cout << "Nx = " << Nx << std::endl;
    std::cout << "Ny = " << Ny << std::endl;
    std::cout << "Nz = " << Nz << std::endl;
    std::cout << std::endl;

    std::cout << "Grid boundaries" << std::endl;
    std::cout << "[x_min, x_max]: [" << x_min << "," << x_max << "]" << std::endl;
    std::cout << "[y_min, y_max]: [" << y_min << "," << y_max << "]" << std::endl;  
    std::cout << "[z_min, z_max]: [" << z_min << "," << z_max << "]" << std::endl;  
    std::cout << std::endl;

    float dx = (x_max - x_min) / (Nx - 1);
	float dy = (y_max - y_min) / (Ny - 1);
	float dz = (z_max - z_min) / (Nz - 1);

    std::cout << "Grid spacing" << std::endl;
    std::cout << "dx = " << dx << std::endl;
    std::cout << "dy = " << dy << std::endl;
    std::cout << "dz = " << dz << std::endl;
    std::cout << std::endl;

	/* Initialize device data */
	int deviceId;
	int numberOfSMs;

	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    
    float *grid_x, *grid_y, *grid_z; // *grid_buffer;

	// int fluid_data_size = sizeof(float) * Nx * Ny * Nz;

    checkCuda(cudaMalloc(&grid_x, sizeof(float) * Nx));
	checkCuda(cudaMalloc(&grid_y, sizeof(float) * Ny));
	checkCuda(cudaMalloc(&grid_z, sizeof(float) * Nz));
    // checkCuda(cudaMalloc(&grid_buffer, 3 * fluid_data_size));

	dim3 grid_dimensions(SM_mult_x * numberOfSMs, SM_mult_y * numberOfSMs, SM_mult_z * numberOfSMs);
	dim3 block_dimensions(num_threads_per_block_x, num_threads_per_block_y, num_threads_per_block_z);

    std::cout << "Initializing the grid" << std::endl;
	InitializeGrid<<<grid_dimensions, block_dimensions>>>(x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz,
															grid_x, grid_y, grid_z, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());
    std::cout << "Grid initialization kernel complete" << std::endl;

    // Device2Host: >32 MB data exceeds max pinned memory size for GDS transfer
    float *h_gridx, *h_gridy, *h_gridz;

    h_gridx = (float*)malloc(sizeof(float) * Nx);
    h_gridy = (float*)malloc(sizeof(float) * Ny);
    h_gridz = (float*)malloc(sizeof(float) * Nz);

    std::cout << "Transferring grid basis from device to host" << std::endl;
    checkCuda(cudaMemcpy(h_gridx, grid_x, sizeof(float) * Nx, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_gridy, grid_y, sizeof(float) * Ny, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_gridz, grid_z, sizeof(float) * Nz, cudaMemcpyDeviceToHost));
    std::cout << "Device2Host grid basis transfer complete" << std::endl;

    std::vector<std::string> grid_files (8); // 8 is the number of threads I'm going with
    std::string base_file = "../data/grid/grid_data";
    for (size_t i = 0; i < grid_files.size(); i++){
        grid_files[i] = base_file + std::to_string(i) + ".csv";
    }   

    std::cout << "Writing grid data to file" << std::endl;
    writeGrid(grid_files, h_gridx, h_gridy, h_gridz, Nx, Ny, Nz);
    std::cout << "Grid data writing to file complete" << std::endl;

    // GDS - max pinned memory size problem
    // WriteGridBuffer<<<grid_dimensions, block_dimensions>>>(grid_buffer, grid_x, grid_y, grid_z, Nx, Ny, Nz);
    // writeGridGDS("../data/grid.dat", grid_buffer, Nx, Ny, Nz);

    checkCuda(cudaFree(grid_x));
	checkCuda(cudaFree(grid_y));
	checkCuda(cudaFree(grid_z));      
    // checkCuda(cudaFree(grid_buffer));  

    free(h_gridx);
    free(h_gridy);
    free(h_gridz);
    return 0;
}