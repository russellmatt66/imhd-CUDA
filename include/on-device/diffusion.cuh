#ifndef DIFFUSION_CUH
#define DIFFUSION_CUH

__device__ float numericalDiffusionLocal(const float fluidvar, 
    const float fluidvar_ip1, const float fluidvar_jp1, const float fluidvar_kp1, 
    const float fluidvar_im1, const float fluidvar_jm1, const float fluidvar_km1,
    const float D, const float dx, const float dy, const float dz);

// Thrash the cache
__device__ float numericalDiffusion(const int i, const int j, const int k, const float* fluid_var, 
    const float D, const float dx, const float dy, const float dz, 
    const int ivf, const int Nx, const int Ny, const int Nz);

__device__ float numericalDiffusionFront(const int i, const int j, const float* fluid_var, 
    const float D, const float dx, const float dy, const float dz, 
    const int ivf, const int Nx, const int Ny, const int Nz);

__device__ float numericalDiffusionBack(const int i, const int j, const float* fluid_var, 
    const float D, const float dx, const float dy, const float dz, 
    const int ivf, const int Nx, const int Ny, const int Nz);

#endif