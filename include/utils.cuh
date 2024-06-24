#ifndef UTILS_CUH
#define UTILS_CUH

__global__ void PrintGrid(const float* x_grid, const float* y_grid, const float* z_grid, const size_t Nx, const size_t Ny, const size_t Nz);
__global__ void PrintIntvar(const float* intvar, const float* fluidvar, const size_t Nx, const size_t Ny, const size_t Nz);
__global__ void PrintFluidvar(const float* fluidvar, const size_t Nx, const size_t Ny, const size_t Nz);

#endif