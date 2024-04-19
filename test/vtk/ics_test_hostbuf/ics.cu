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
#include <string>
#include <vector>

#include "../../../include/initialize_od.cuh"
// #include "../../../include/gds.cuh"
#include "../../../include/utils.hpp"

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

   dim3 grid_dimensions(num_blocks, num_blocks, num_blocks);
   dim3 block_dimensions(32, 16, 2);

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
   checkCuda(cudaPeekAtLastError());
   checkCuda(cudaDeviceSynchronize());

   float *h_xgrid, *h_ygrid, *h_zgrid;

   h_xgrid = (float*)malloc(sizeof(float)*Nx);
   h_ygrid = (float*)malloc(sizeof(float)*Ny);
   h_zgrid = (float*)malloc(sizeof(float)*Nz); 

   cudaMemcpy(h_xgrid, x_grid, sizeof(float)*Nx, cudaMemcpyDeviceToHost);
   cudaMemcpy(h_ygrid, y_grid, sizeof(float)*Ny, cudaMemcpyDeviceToHost);
   cudaMemcpy(h_zgrid, z_grid, sizeof(float)*Nz, cudaMemcpyDeviceToHost);
   checkCuda(cudaPeekAtLastError());
   checkCuda(cudaDeviceSynchronize());

   std::vector<std::string> grid_files (8); // 8 is the number of threads I'm going with
   std::string base_file = "./data/grid/grid_data";
   for (size_t i = 0; i < grid_files.size(); i++){
      grid_files[i] = base_file + std::to_string(i) + ".csv";
   }   

   std::cout << "Right before writing grid data to storage" << std::endl;
   checkCuda(cudaMemGetInfo(&freePinned, &totalPinned));
   std::cout << "Total pinned memory: " << totalPinned / (1024 * 1024) << " MB" << std::endl;
   std::cout << "Free pinned memory: " << freePinned / (1024 * 1024) << " MB" << std::endl;

   std::cout << "Writing grid data out" << std::endl;
   std::cout << "Size of grid data is " << fluid_data_size * 3 / (1024 * 1024) << " MB" << std::endl;
   
   writeGrid(grid_files, h_xgrid, h_ygrid, h_zgrid, Nx, Ny, Nz);

   InitialConditions<<<grid_dimensions, block_dimensions>>>(rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, 1.0, x_grid, y_grid, z_grid, Nx, Ny, Nz);
   checkCuda(cudaPeekAtLastError());
   // WriteGridBuffer<<<grid_dimensions, block_dimensions>>>(grid_data, x_grid, y_grid, z_grid, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());

   float *h_rho, *h_rhovz;

   h_rho = (float*)malloc(sizeof(float)*Nx*Ny*Nz);
   h_rhovz = (float*)malloc(sizeof(float)*Nx*Ny*Nz);

   cudaMemcpy(h_rho, rho, sizeof(float)*Nx*Ny*Nz, cudaMemcpyDeviceToHost);
   cudaMemcpy(h_rhovz, rhov_z, sizeof(float)*Nx*Ny*Nz, cudaMemcpyDeviceToHost);
   checkCuda(cudaPeekAtLastError());
   checkCuda(cudaDeviceSynchronize());

   std::vector<std::string> fluidvar_files (8); // 8 is the number of threads I'm going with
   base_file = "./data/fluidvars/fluid_data";
   for (size_t i = 0; i < grid_files.size(); i++){
      fluidvar_files[i] = base_file + std::to_string(i) + ".csv";
   }  

   writeFluidVars(fluidvar_files, h_rho, h_rhovz, Nx, Ny, Nz);

   std::cout << "Right before writing fluid data to storage" << std::endl;
   checkCuda(cudaMemGetInfo(&freePinned, &totalPinned));
   std::cout << "Total pinned memory: " << totalPinned / (1024 * 1024) << " MB" << std::endl;
   std::cout << "Free pinned memory: " << freePinned / (1024 * 1024) << " MB" << std::endl;
   
   std::cout << "Writing rho data out" << std::endl;
   std::cout << "Size of rho data is " << fluid_data_size / (1024 * 1024) << " MB" << std::endl;


   // Free device data
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

   // Free host data
   free(h_xgrid);
   free(h_ygrid);
   free(h_zgrid);
   free(h_rho);
   free(h_rhovz);
   return 0;
}