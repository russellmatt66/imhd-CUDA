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
#define gamma (5.0 / 3.0)

struct KernelConfig { // variables needed to launch kernrels
    dim3 gridDim, blockDim;
    float D; // numerical diffusivity
    float dt, dx, dy, dz; 
    int Nx, Ny, Nz;
};

// Global kernels and launchers
// Megakernels
__global__ void FluidAdvance(float* fluidvar, const float* intvar, 
     const float D, const float dt, const float dx, const float dy, const float dz, 
     const int Nx, const int Ny, const int Nz);

__global__ void FluidAdvanceLocal(float* fluidvar, const float* intvar, 
    const float D, const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void FluidAdvanceLocalNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

void LaunchFluidAdvanceLocalNoDiff(float* fluidvar, const float* intvar, const KernelConfig& kcfg);

__global__ void BoundaryConditions(float* fluidvar, const float* intvar, 
     const float D, const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

// Microkernels
__global__ void FluidAdvanceMicroRhoLocalNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void FluidAdvanceMicroRhoVXLocalNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void FluidAdvanceMicroRhoVYLocalNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void FluidAdvanceMicroRhoVZLocalNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void FluidAdvanceMicroBXLocalNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void FluidAdvanceMicroBYLocalNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void FluidAdvanceMicroBZLocalNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void FluidAdvanceMicroELocalNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

// Device kernels
__device__ float LaxWendroffAdvRho(const int i, const int j, const int k, 
     const float* fluidvar, const float* intvar, 
     const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvRhoLocal(const float rho, const float rho_int, 
    const float rhovx_int, const float rhovx_int_im1,
    const float rhovy_int, const float rhovy_int_jm1,
    const float rhovz_int, const float rhovz_int_km1,
    const float dt, const float dx, const float dy, const float dz);

__device__ float LaxWendroffAdvRhoVX(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvRhoVXLocal(const float rho_int, const float rho_int_im1, const float rho_int_jm1, const float rho_int_km1,
    const float rhovx, const float rhovx_int, const float rhovx_int_im1, const float rhovx_int_jm1, const float rhovx_int_km1,
    const float rhovy_int, const float rhovy_int_jm1, 
    const float rhovz_int, const float rhovz_int_km1,
    const float Bx_int, const float Bx_int_im1, const float Bx_int_jm1, const float Bx_int_km1,
    const float By_int, const float By_int_jm1, 
    const float Bz_int, const float Bz_int_km1, 
    const float p_ijk, const float p_im1jk, 
    const float Bsq_ijk, const float Bsq_im1jk, 
    const float dt, const float dx, const float dy, const float dz);

__device__ float LaxWendroffAdvRhoVY(const int i, const int j, const int k, 
     const float* fluidvar, const float* intvar,  
     const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvRhoVYLocal(const float rho_int, const float rho_int_im1, const float rho_int_jm1, const float rho_int_km1,
    const float rhovx_int, const float rhovx_int_im1,
    const float rhovy, const float rhovy_int, const float rhovy_int_im1, const float rhovy_int_jm1, const float rhovy_int_km1,
    const float rhovz_int, const float rhovz_int_km1,
    const float Bx_int, const float Bx_int_im1, 
    const float By_int, const float By_int_im1, const float By_int_jm1, const float By_int_km1,
    const float Bz_int, const float Bz_int_km1, 
    const float Bsq_ijk, const float Bsq_ijm1k,
    const float p_ijk, const float p_ijm1k,
    const float dt, const float dx, const float dy, const float dz);

__device__ float LaxWendroffAdvRhoVZ(const int i, const int j, const int k,
     const float* fluidvar, const float* intvar, 
     const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvRhoVZLocal(const float rho_int, const float rho_int_im1, const float rho_int_jm1, const float rho_int_km1,
    const float rhovx_int, const float rhovx_int_im1,  
    const float rhovy_int, const float rhovy_int_jm1, 
    const float rhovz, const float rhovz_int, const float rhovz_int_im1, const float rhovz_int_jm1, const float rhovz_int_km1, 
    const float Bx_int, const float Bx_int_im1,
    const float By_int, const float By_int_jm1, 
    const float Bz_int, const float Bz_int_im1, const float Bz_int_jm1, const float Bz_int_km1,  
    const float p_ijk, const float p_ijkm1, 
    const float Bsq_ijk, const float Bsq_ijkm1,
    const float dt, const float dx, const float dy, const float dz);

__device__ float LaxWendroffAdvBX(const int i, const int j, const int k,
     const float* fluidvar, const float* intvar, 
     const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvBXLocal(const float rho_int, const float rho_int_jm1, const float rho_int_km1, 
    const float rhovx_int, const float rhovx_int_jm1, const float rhovx_int_km1,  
    const float rhovy_int, const float rhovy_int_jm1, 
    const float rhovz_int, const float rhovz_int_km1, 
    const float Bx, const float Bx_int, const float Bx_int_jm1, const float Bx_int_km1,
    const float By_int, const float By_int_jm1, 
    const float Bz_int, const float Bz_int_km1, 
    const float dt, const float dx, const float dy, const float dz);

__device__ float LaxWendroffAdvBY(const int i, const int j, const int k,
     const float* fluidvar, const float* intvar, 
     const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvBYLocal(const float rho_int, const float rho_int_im1, const float rho_int_km1,  
    const float rhovx_int, const float rhovx_int_im1,  
    const float rhovy_int, const float rhovy_int_im1, const float rhovy_int_km1, 
    const float rhovz_int, const float rhovz_int_km1, 
    const float Bx_int, const float Bx_int_im1,
    const float By, const float By_int, const float By_int_im1, const float By_int_km1, 
    const float Bz_int, const float Bz_int_km1, 
    const float dt, const float dx, const float dy, const float dz);

__device__ float LaxWendroffAdvBZ(const int i, const int j, const int k,
     const float* fluidvar, const float* intvar, 
     const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvBZLocal(const float rho_int, const float rho_int_im1, const float rho_int_jm1, 
    const float rhovx_int, const float rhovx_int_im1,
    const float rhovy_int, const float rhovy_int_jm1,    
    const float rhovz_int, const float rhovz_int_im1, const float rhovz_int_jm1, 
    const float Bx_int, const float Bx_int_im1,  
    const float By_int, const float By_int_jm1, 
    const float Bz, const float Bz_int, const float Bz_int_im1, const float Bz_int_jm1, 
    const float dt, const float dx, const float dy, const float dz);

__device__ float LaxWendroffAdvE(const int i, const int j, const int k,
     const float* fluidvar, const float* intvar, 
     const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvELocal(const float rho_int, const float rho_int_im1, const float rho_int_jm1, const float rho_int_km1, 
    const float rhovx_int, const float rhovx_int_im1,
    const float rhovy_int, const float rhovy_int_jm1, 
    const float rhovz_int, const float rhovz_int_km1, 
    const float Bx_int, const float Bx_int_im1, 
    const float By_int, const float By_int_jm1, 
    const float Bz_int, const float Bz_int_km1, 
    const float e, const float e_int, const float e_int_im1, const float e_int_jm1, const float e_int_km1, 
    const float p_ijk, const float p_im1jk, const float p_ijm1k, const float p_ijkm1, 
    const float Bsq_ijk, const float Bsq_im1jk, const float Bsq_ijm1k, const float Bsq_ijkm1, 
    const float Bdotu_ijk, const float Bdotu_im1jk, const float Bdotu_ijm1k, const float Bdotu_ijkm1, 
    const float dt, const float dx, const float dy, const float dz);
     
#endif