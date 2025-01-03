#ifndef KERNELS_OD_INTVAR_CUH
#define KERNELS_OD_INTVAR_CUH

// Non-thrashing megakernels
__global__ void ComputeIntVarsLocal(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz, const float D,
    const int Nx, const int Ny, const int Nz);

// Cache thrashing megakernels 
__global__ void ComputeIntermediateVariables(const float* fluidvar, float* intvar,
    const float D, const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void ComputeIntermediateVariablesBoundary(const float* fluidvar, float* intvar,
    const float D, const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

// Standard functions for calculating the intermediate variables
__device__ float intRho(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVX(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVY(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVZ(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBx(const int i, const int j, const int k,
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

__device__ float intBy(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

__device__ float intBz(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

__device__ float intE(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

// Standard functions for calculating the intermediate variables on the i = Nx-1, j = 0, k \in [1,Nz-2] line
__device__ float intRhoBottomLeft(const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVXBottomLeft(const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVYBottomLeft(const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVZBottomLeft(const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBXBottomLeft(const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBYBottomLeft(const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBZBottomLeft(const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intEBottomLeft(const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);


#endif