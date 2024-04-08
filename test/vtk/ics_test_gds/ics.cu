/* 
Visualize the initial conditions
(1) Compute ICs
(2) Output them for visualization
*/

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "../../../include/initialize_od.cuh"
#include "../../../include/gds.cuh"
#include "../../../include/utils.cuh"

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

   size_t freePinned, totalPinned;

   std::cout << "Beginning of program" << std::endl;
   checkCuda(cudaMemGetInfo(&freePinned, &totalPinned));
   std::cout << "Total pinned memory: " << totalPinned / (1024 * 1024) << " MB" << std::endl;
   std::cout << "Free pinned memory: " << freePinned / (1024 * 1024) << " MB" << std::endl;

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
   float *grid_data;

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
   
   checkCuda(cudaMalloc(&grid_data, fluid_data_size * 3));

   float x_min = -M_PI;
   float x_max = M_PI;
   float y_min = -M_PI;
   float y_max = M_PI;
   float z_min = -M_PI;
   float z_max = M_PI;

   float dx = (x_max - x_min) / (Nx - 1.0);
   float dy = (y_max - y_min) / (Ny - 1.0);
   float dz = (z_max - z_min) / (Nz - 1.0);

   std::cout << "Domain boundaries are: " << std::endl;
   std::cout << "[x_min, x_max] = [" << x_min << ", " << x_max << "]" << std::endl;
   std::cout << "[y_min, y_max] = [" << y_min << ", " << y_max << "]" << std::endl; 
   std::cout << "[z_min, z_max] = [" << z_min << ", " << z_max << "]" << std::endl; 

   std::cout << "Spacing is: " << std::endl;
   std::cout << "dx = " << dx << std::endl;
   std::cout << "dy = " << dy << std::endl;
   std::cout << "dz = " << dz << std::endl;

   InitializeGrid<<<grid_dimensions, block_dimensions>>>(x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz, x_grid, y_grid, z_grid, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());

   // // Print some values of the grid out
   // size_t stride_x = Nx / 8, stride_y = Ny / 8, stride_z = Nz / 8;
   // for (int k = 0; k < Nz; k += stride_z){
   //    for (int i = 0; i < Nx; i += stride_x){
   //       for (int j = 0; j < Ny; j += stride_y){
   //          std::cout << "(x, y, z) = (" << x_grid[i] << ", " << y_grid[j] << ", " << z_grid[k] << ")" << std::endl;  
   //       }
   //    }
   // } 

   InitialConditions<<<grid_dimensions, block_dimensions>>>(rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, 1.0, x_grid, y_grid, z_grid, Nx, Ny, Nz);
   WriteGridBuffer<<<grid_dimensions, block_dimensions>>>(grid_data, x_grid, y_grid, z_grid, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());

   std::cout << "Right before writing grid data to storage" << std::endl;
   checkCuda(cudaMemGetInfo(&freePinned, &totalPinned));
   std::cout << "Total pinned memory: " << totalPinned / (1024 * 1024) << " MB" << std::endl;
   std::cout << "Free pinned memory: " << freePinned / (1024 * 1024) << " MB" << std::endl;

   std::cout << "Writing grid data out" << std::endl;
   std::cout << "Size of grid data is " << fluid_data_size * 3 / (1024 * 1024) << " MB" << std::endl;

   /*
      grid_data looks like 
      x0 y0 z0 x0 y1 z0 x0 y2 z0 ... x0 yNy-1 z0 x1 y0 z0 x1 y1 z0 ... xNx-1 yNy-1 z0 x0 y0 z1 x0 y1 z1 ... xNx-1 yNy-1 zNz-1
   */  
   writeGridGDS("xyz_grid.dat", grid_data, Nx, Ny, Nz); 
   
   std::cout << "Right before writing fluid data to storage" << std::endl;
   checkCuda(cudaMemGetInfo(&freePinned, &totalPinned));
   std::cout << "Total pinned memory: " << totalPinned / (1024 * 1024) << " MB" << std::endl;
   std::cout << "Free pinned memory: " << freePinned / (1024 * 1024) << " MB" << std::endl;
   
   std::cout << "Writing rho data out" << std::endl;
   std::cout << "Size of rho data is " << fluid_data_size / (1024 * 1024) << " MB" << std::endl;
   // data looks like d000 d010 d020 ... d0,Ny-1,0 d100 d110 ... dNx-1,Ny-1,0 d001 d011 ... dNx-1,Ny-1,Nz-1
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
   cudaFree(grid_data);
   return 0;
}