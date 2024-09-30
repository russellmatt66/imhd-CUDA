#include <assert.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>

#include "../include/kernels_od.cuh"
#include "../include/kernels_od_intvar.cuh"
#include "../include/initialize_od.cuh"
#include "../include/gds.cuh"
#include "../include/utils.hpp"

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
/* This REALLY needs to be a library somewhere */
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* REFACTOR THIS ACCORDING TO REFACTORED LIBRARIES */
int main(int argc, char* argv[]){
	/* 
   PARSE ARGUMENTS 
   */
   int Nt = atoi(argv[1]);
   int Nx = atoi(argv[2]);
   int Ny = atoi(argv[3]);
   int Nz = atoi(argv[4]);
   float J0 = atof(argv[5]); // amplitude of the current
	float D = atof(argv[6]); // coefficient of numerical diffusion
	float x_min = atof(argv[7]);
	float x_max = atof(argv[8]);
	float y_min = atof(argv[9]);
	float y_max = atof(argv[10]);
	float z_min = atof(argv[11]);
	float z_max = atof(argv[12]);
	float dt = atof(argv[13]);

   /* 
   CUDA BOILERPLATE 
   */
   int deviceId;
   int numberOfSMs;

   cudaGetDevice(&deviceId);
   cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

   /* SPECIFY EXECUTION CONFIGURATIONS */
   dim3 exec_grid_dims(numberOfSMs, numberOfSMs, numberOfSMs);
   dim3 mesh_block_dims();
   dim3 init_block_dims();
   dim3 intvar_block_dims();
   dim3 fluid_block_dims();

   /* 
   INITIALIZE SIMULATION DATA
   */
   size_t cube_size = Nx * Ny * Nz;
   size_t fluidvar_size = sizeof(float) * cube_size;
   size_t fluid_data_size = 8 * fluidvar_size;

   float *fluidvars, *fluidvars_np1, *fluidvars_int;

   checkCuda(cudaMalloc(&fluidvars, fluid_data_size));
   checkCuda(cudaMalloc(&fluidvars_np1, fluid_data_size));
   checkCuda(cudaMalloc(&fluidvars_int, fluid_data_size));

   float *x_grid, *y_grid, *z_grid;

   checkCuda(cudaMalloc(&x_grid, sizeof(float) * Nx));
   checkCuda(cudaMalloc(&y_grid, sizeof(float) * Ny));
   checkCuda(cudaMalloc(&z_grid, sizeof(float) * Nz));

   float dx = (x_max - x_min) / (Nx - 1);
	float dy = (y_max - y_min) / (Ny - 1);
	float dz = (z_max - z_min) / (Nz - 1);

   /* CALL INITIALIZATION KERNELS */

   /* WRITE INITIAL DATA OUT */

   /* COMPUTE STABILITY CRITERION */

   /* SIMULATION LOOP */
   for (int it = 0; it < Nt; it++){
      /* UPDATE fluidvars_np1 USING fluidvars */
      /* COPY fluidvars TO HOST */
      checkCuda(cudaDeviceSynchronize());

      /* COMPUTE fluidvars_int */
      /* WRITE fluidvars_np1 INTO fluidvars */
      /* WRITE .h5 file TO STORAGE USING fluidvars_np1 */   
      checkCuda(cudaDeviceSynchronize());
      
   }

   /* FREE EVERYTHING */
   checkCuda(cudaFree(fluidvars));
   checkCuda(cudaFree(fluidvars_np1));
   checkCuda(cudaFree(fluidvars_int));
   checkCuda(cudaFree(x_grid));
   checkCuda(cudaFree(y_grid));
   checkCuda(cudaFree(z_grid));
   return 0;
}