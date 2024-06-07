#ifndef KERNELS_OD_CUH
#define KERNELS_OD_CUH

// Lax-Wendroff scheme
/*
CUDA discourages the use of complex class structures, and race conditions that exist as a consequence of asynchronous thread execution necessitate that the fluid data be 
partitioned into two sets:

(1) The set of fluid variables at the current timestep (const)
(2) The set of fluid variables for the future timestep (*_np1)

(1) will be held static while data is populated into (2). Then, data will be transferred from (2) -> (1), and the process repeated. 
*/ 

// physical constants
#define gamma 5.0 / 3.0

// Global kernels
__global__ void SwapSimData(float* fluidvar, const float* fluidvar_np1, const int Nx, const int Ny, const int Nz); 

__global__ void FluidAdvance(float* fluidvar_np1, const float* fluidvar, const float* intvar, 
     const float D, const float dt, const float dx, const float dy, const float dz, 
     const int Nx, const int Ny, const int Nz);

__global__ void BoundaryConditions(float* fluidvar_np1, const float* fluidvar, const float* intvar, 
     const float D, const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

// Device kernels
__device__ float LaxWendroffAdvRho(const int i, const int j, const int k, 
     const float* fluidvar, const float* intvar, 
     const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvRhoVX(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvRhoVY(const int i, const int j, const int k, 
     const float* fluidvar, const float* intvar,  
     const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvRhoVZ(const int i, const int j, const int k,
     const float* fluidvar, const float* intvar, 
     const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvBX(const int i, const int j, const int k,
     const float* fluidvar, const float* intvar, 
     const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvBY(const int i, const int j, const int k,
     const float* fluidvar, const float* intvar, 
     const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvBZ(const int i, const int j, const int k,
     const float* fluidvar, const float* intvar, 
     const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvE(const int i, const int j, const int k,
     const float* fluidvar, const float* intvar, 
     const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

// Helper Functions
__device__ float B_sq(int i, int j, int k, const float* fluidvar, const int Nx, const int Ny, const int Nz); // B / \sqrt{\mu_{0}} -> B 
__device__ float p(int i, int j, int k, const float* fluidvar, const float B_sq, const float KE, const int Nx, const int Ny, const int Nz);
__device__ float KE(int i, int j, int k, const float* fluidvar, const int Nx, const int Ny, const int Nz); // \rho * \vec{u}\cdot\vec{u} * 0.5
__device__ float B_dot_u(int i, int j, int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);

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