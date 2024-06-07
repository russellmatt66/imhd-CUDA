#ifndef DIFFUSION_CUH
#define DIFFUSION_CUH

__device__ float numericalDiffusion(const int i, const int j, const int k, const float* fluid_var, 
    const float D, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

__device__ float numericalDiffusionFront(const int i, const int j, const float* fluid_var, 
    const float D, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

__device__ float numericalDiffusionBack(const int i, const int j, const float* fluid_var, 
    const float D, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

#endif