#include "diffusion.cuh"

__device__ float numericalDiffusion(const int i, const int j, const int k, const float* fluid_var, 
    const float D, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
    {
    /* IMPLEMENT 2ND ORDER CENTRAL DIFFERENCES */
    }

__device__ float numericalDiffusionFront(const int i, const int j, const float* fluid_var, 
    const float D, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
    {
    
    /* IMPLEMENT 2ND ORDER CENTRAL DIFFERENCES WITH PERIODICITY */

    }

__device__ float numericalDiffusionBack(const int i, const int j, const float* fluid_var, 
    const float D, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
    {
    
    /* IMPLEMENT 2ND ORDER CENTRAL DIFFERENCES WITH PERIODICITY */

    }