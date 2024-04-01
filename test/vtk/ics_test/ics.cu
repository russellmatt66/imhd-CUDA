/* 
Visualize the initial conditions
(1) Compute ICs
(2) Output them for visualization
*/

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "../../../include/initialize_od.cuh"
#include "../../../include/gds.cuh"


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
   int Nx = atoi(argv[1]);
   int Ny = atoi(argv[2]);
   int Nz = atoi(argv[3]);

   int deviceId;
   int numberOfSMs;

   cudaGetDevice(&deviceId);
   cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

   int num_blocks = 2 * numberOfSMs;
   int num_threads_per_block = 1024;

   dim3 grid_dimensions(num_blocks, num_blocks, num_blocks);
   dim3 block_dimensions(num_threads_per_block, num_threads_per_block, num_threads_per_block);

   float *rho, *rhov_x, *rhov_y, *rhov_z, *Bx, *By, *Bz, *e;
   float *x_grid, *y_grid, *z_grid;

   size_t fluid_data_dimension = Nx*Ny*Nz;
   size_t fluid_data_size = sizeof(float)*fluid_data_dimension;

   checkCuda(cudaMalloc(&rho, fluid_data_size));
   checkCuda(cudaMalloc(&rhov_x, fluid_data_size));
   checkCuda(cudaMalloc(&rhov_y, fluid_data_size));
   checkCuda(cudaMalloc(&rhov_z, fluid_data_size));
   checkCuda(cudaMalloc(&Bx, fluid_data_size));
   checkCuda(cudaMalloc(&By, fluid_data_size));
   checkCuda(cudaMalloc(&Bz, fluid_data_size));
   checkCuda(cudaMalloc(&e, fluid_data_size));

   // checkCuda(cudaMalloc(&x_grid, fluid_data_size));
   // checkCuda(cudaMalloc(&y_grid, fluid_data_size));
   // checkCuda(cudaMalloc(&z_grid, fluid_data_size));
   checkCuda(cudaMalloc(&x_grid, sizeof(float)*Nx));
   checkCuda(cudaMalloc(&y_grid, sizeof(float)*Ny));
   checkCuda(cudaMalloc(&z_grid, sizeof(float)*Nz));

   float x_min = -M_PI;
   float x_max = M_PI;
   float y_min = -M_PI;
   float y_max = M_PI;
   float z_min = -M_PI;
   float z_max = M_PI;

   float dx = (x_max - x_min) / (Nx - 1.0);
   float dy = (y_max - y_min) / (Ny - 1.0);
   float dz = (z_max - z_min) / (Nz - 1.0);

   InitializeGrid<<<grid_dimensions, block_dimensions>>>(x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz, x_grid, y_grid, z_grid, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());

   InitialConditions<<<grid_dimensions, block_dimensions>>>(rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, 1.0, x_grid, y_grid, z_grid, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());

   writeGridGDS("xyz_grid.dat", x_grid, y_grid, z_grid, Nx, Ny, Nz);
   
   std::cout << "Writing rho data out" << std::endl;
   writeDataGDS("rho_ics.dat", rho, fluid_data_dimension);

   cudaFree(rho);
   cudaFree(rhov_x);
   cudaFree(rhov_y);
   cudaFree(rhov_z);
   cudaFree(Bx);
   cudaFree(By);
   cudaFree(Bz);
   cudaFree(e);
   cudaFree(x_grid);
   cudaFree(y_grid);
   cudaFree(z_grid);
   return 0;
}