#ifndef KERNELS_OD_INTVAR_CUH
#define KERNELS_OD_INTVAR_CUH

struct IVKernelConfig {
    dim3 gridDim, blockDim;
    float D;
    float dt, dx, dy, dz;
    int Nx, Ny, Nz;
};

// Kernel Launchers
void LaunchIntvarAdvanceNoDiff(const float* fluidvars, float* intvars, IVKernelConfig ivkcfg);
void LaunchIntvarAdvanceStrideNoDiff(const float* fluidvars, float* intvars, IVKernelConfig ivkcfg);

// MEGAKERNELS
// Non-thrashing megakernels
__global__ void ComputeIntVarsSHMEMNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

// Cache-thrashing megakernels 
__global__ void ComputeIntermediateVariablesStride(const float* fluidvar, float* intvar,
    const float D, const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void ComputeIntermediateVariablesStrideNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void ComputeIntermediateVariablesNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

// MICROKERNELS
__global__ void ComputeIntRhoMicroLocalNoDiff(const float* fluidvar, float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void ComputeIntRhoVXMicroLocalNoDiff(const float* fluidvar, float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void ComputeIntRhoVYMicroLocalNoDiff(const float* fluidvar, float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void ComputeIntRhoVZMicroLocalNoDiff(const float* fluidvar, float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void ComputeIntBXMicroLocalNoDiff(const float* fluidvar, float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void ComputeIntBYMicroLocalNoDiff(const float* fluidvar, float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void ComputeIntBZMicroLocalNoDiff(const float* fluidvar, float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);
    
__global__ void ComputeIntEMicroLocalNoDiff(const float* fluidvar, float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

// DEVICE KERNELS
// Standard, cache-thrashing functions for calculating the intermediate variables
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

__device__ float intBX(const int i, const int j, const int k,
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

__device__ float intBY(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

__device__ float intBZ(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

__device__ float intE(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

// High register pressure predictor steps
__device__ float intRhoLocal(const float rho, 
    const float rhovx, const float rhovy, const float rhovz,
    const float rhovx_ip1, const float rhovy_jp1, const float rhovz_kp1,
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVXLocal(const float rho, const float rho_ip1, const float rho_jp1, const float rho_kp1,
    const float rhovx, const float rhovx_ip1, const float rhovx_jp1, const float rhovx_kp1, 
    const float rhovy, const float rhovy_jp1, 
    const float rhovz, const float rhovz_kp1,
    const float Bx, const float Bx_ip1, const float Bx_jp1, const float Bx_kp1,
    const float By, const float By_jp1, 
    const float Bz, const float Bz_kp1,
    const float p_ip1jk, const float p_ijk,
    const float Bsq_ip1jk, const float Bsq_ijk,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVYLocal(const float rho, const float rho_ip1, const float rho_jp1, const float rho_kp1,
    const float rhovx, const float rhovx_ip1, 
    const float rhovy, const float rhovy_ip1, const float rhovy_jp1, const float rhovy_kp1,
    const float rhovz, const float rhovz_kp1,
    const float Bx, const float Bx_ip1,
    const float By, const float By_ip1, const float By_jp1, const float By_kp1,
    const float Bz, const float Bz_kp1,
    const float p_ijk, const float p_ijp1k,
    const float Bsq_ijk, const float Bsq_ijp1k,  
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVZLocal(const float rho, const float rho_ip1, const float rho_jp1, const float rho_kp1,
    const float rhovx, const float rhovx_ip1, 
    const float rhovy, const float rhovy_jp1, 
    const float rhovz, const float rhovz_ip1, const float rhovz_jp1, const float rhovz_kp1,
    const float Bx, const float Bx_ip1, 
    const float By, const float By_jp1,
    const float Bz, const float Bz_ip1, const float Bz_jp1, const float Bz_kp1,
    const float p_ijk, const float p_ijkp1, 
    const float Bsq_ijk, const float Bsq_ijkp1,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBXLocal(const float rho, const float rho_jp1, const float rho_kp1,
    const float rhovx, const float rhovx_jp1, const float rhovx_kp1, 
    const float rhovy, const float rhovy_jp1,
    const float rhovz, const float rhovz_kp1, 
    const float Bx, const float Bx_jp1, const float Bx_kp1,
    const float By, const float By_jp1, 
    const float Bz, const float Bz_kp1,
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

__device__ float intBYLocal(const float rho, const float rho_ip1, const float rho_kp1,
    const float rhovx, const float rhovx_ip1, 
    const float rhovy, const float rhovy_ip1, const float rhovy_kp1, 
    const float rhovz, const float rhovz_kp1, 
    const float Bx, const float Bx_ip1, 
    const float By, const float By_ip1, const float By_kp1,
    const float Bz, const float Bz_kp1,  
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

__device__ float intBZLocal(const float rho, const float rho_ip1, const float rho_jp1, 
    const float rhovx, const float rhovx_ip1, 
    const float rhovy, const float rhovy_jp1,
    const float rhovz, const float rhovz_ip1, const float rhovz_jp1, 
    const float Bx, const float Bx_ip1, 
    const float By, const float By_jp1,
    const float Bz, const float Bz_ip1, const float Bz_jp1, 
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

__device__ float intELocal(const float rho, const float rho_ip1, const float rho_jp1, const float rho_kp1,
    const float rhovx, const float rhovx_ip1, 
    const float rhovy, const float rhovy_jp1, 
    const float rhovz, const float rhovz_kp1,
    const float Bx, const float Bx_ip1, 
    const float By, const float By_jp1, 
    const float Bz, const float Bz_kp1, 
    const float e, const float e_ip1, const float e_jp1, const float e_kp1,
    const float p_ijk, const float p_ip1jk, const float p_ijp1k, const float p_ijkp1,
    const float Bsq_ijk, const float Bsq_ip1jk, const float Bsq_ijp1k, const float Bsq_ijkp1, 
    const float Bdotu_ijk, const float Bdotu_ip1jk, const float Bdotu_ijp1k, const float Bdotu_ijkp1, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

// Standard functions for calculating the intermediate variables on the Front Face (I)
// i \in [1, Nx-1]
// j \in [1, Ny-1]
// k = 0
__device__ float intRhoFront(const int i, const int j, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVXFront(const int i, const int j, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVYFront(const int i, const int j, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVZFront(const int i, const int j, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBXFront(const int i, const int j, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBYFront(const int i, const int j, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBZFront(const int i, const int j, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intEFront(const int i, const int j, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

// Standard functions for calculating the intermediate variables on the Top Face (III)
// i = 0
// j \in [1, Ny-1]
// k \in [1, Nz-2]
__device__ float intRhoTop(const int j, const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVXTop(const int j, const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVYTop(const int j, const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVZTop(const int j, const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBXTop(const int j, const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBYTop(const int j, const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBZTop(const int j, const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intETop(const int j, const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

// Standard functions for calculating the intermediate variables on the Left Face (IV)
// i \in [1, Nx-1]
// j = 0
// k \in [1, Nz-2]
__device__ float intRhoLeft(const int i, const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVXLeft(const int i, const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVYLeft(const int i, const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVZLeft(const int i, const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBXLeft(const int i, const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBYLeft(const int i, const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBZLeft(const int i, const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intELeft(const int i, const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

// Standard functions for calculating the intermediate variables on the Front Top Line (i)
// i = 0 
// j \in [1,Ny-1]
// k = 0
__device__ float intRhoFrontTop(const int j,
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVXFrontTop(const int j, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVYFrontTop(const int j, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVZFrontTop(const int j, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBXFrontTop(const int j, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBYFrontTop(const int j, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBZFrontTop(const int j, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intEFrontTop(const int j, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

// Standard functions for calculating the intermediate variables on the Front Left Line (ii)
// i \in [1, Nx-1]
// j = 0
// k = 0
__device__ float intRhoFrontLeft(const int i,
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVXFrontLeft(const int i, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVYFrontLeft(const int i, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVZFrontLeft(const int i, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBXFrontLeft(const int i, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBYFrontLeft(const int i, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBZFrontLeft(const int i, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intEFrontLeft(const int i, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

// Standard functions for calculating the intermediate variables on the Top Left Line (iii)
// i = 0 
// j = 0
// k \in [1,Nz-1] 
__device__ float intRhoTopLeft(const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVXTopLeft(const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVYTopLeft(const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVZTopLeft(const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBXTopLeft(const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBYTopLeft(const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBZTopLeft(const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intETopLeft(const int k, 
    const float *fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

// Standard functions for calculating the intermediate variables on the Bottom Left Line (vi)
// i = Nx-1
// j = 0
// k \in [1,Nz-2] line
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