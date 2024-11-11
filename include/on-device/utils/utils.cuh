#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void PrintGrid(const float* x_grid, const float* y_grid, const float* z_grid, const size_t Nx, const size_t Ny, const size_t Nz);
__global__ void PrintIntvar(const float* intvar, const float* fluidvar, const size_t Nx, const size_t Ny, const size_t Nz);
__global__ void PrintFluidvar(const float* fluidvar, const size_t Nx, const size_t Ny, const size_t Nz);

#endif