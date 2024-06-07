#ifndef KERNELS_HD_CUH
#define KERNELS_HD_CUH

#define gamma = 5.0 / 3.0

/* Host-device declarations - see README.md for explanation */
// Lax-Wendroff scheme
__global__ void FluidAdvanceRho(float* rho_np1, const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, const float* Bx, 
    const float* By, const float* Bz, const float* e, const int N);

__global__ void FluidAdvanceRhoVX(float* rhovx_np1, const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, const float* Bx, 
    const float* By, const float* Bz, const float* e, const int N);

__global__ void FluidAdvanceRhoVY(float* rhovy_np1, const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, const float* Bx, 
    const float* By, const float* Bz, const float* e, const int N);

__global__ void FluidAdvanceRhoVZ(float* rhovz_np1, const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, const float* Bx, 
    const float* By, const float* Bz, const float* e, const int N);

__global__ void FluidAdvanceBX(float* Bx_np1, const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, const float* Bx, const float* By, 
    const float* Bz, const float* e, const int N);

__global__ void FluidAdvanceBY(float* By_np1, const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, const float* Bx, const float* By, 
    const float* Bz, const float* e, const int N);

__global__ void FluidAdvanceBZ(float* Bz_np1, const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, const float* Bx, const float* By, 
    const float* Bz, const float* e, const int N);

__global__ void FluidAdvanceE(float* E_np1, const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, const float* Bx, const float* By, 
    const float* Bz, const float* e, const int N);

// Flux functions
// These should all be consts - will fix with Python at the right time
__device__ float XFluxRho(int i, int j, int k, float* rho, float* rhov_x, float* rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float YFluxRho(int i, int j, int k, float* rho, float* rhov_x, float* rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float ZFluxRho(int i, int j, int k, float* rho, float* rhov_x, float* rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);

__device__ float XFluxRhoVX(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float YFluxRhoVX(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float ZFluxRhoVX(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);

__device__ float XFluxRhoVY(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float YFluxRhoVY(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float ZFluxRhoVY(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);

__device__ float XFluxRhoVZ(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float YFluxRhoVZ(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float ZFluxRhoVZ(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);

__device__ float XFluxBX(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float YFluxBX(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float ZFluxBX(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);

__device__ float XFluxBY(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float YFluxBY(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float ZFluxBY(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);

__device__ float XFluxBZ(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float YFluxBZ(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float ZFluxBZ(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);

__device__ float XFluxE(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float YFluxE(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float ZFluxE(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);

// Intermediate flux functions
// These should all be consts - will fix with Python at the right time
__device__ float INTXFluxRho(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float INTYFluxRho(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float INTZFluxRho(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);

__device__ float INTXFluxRhoVX(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float INTYFluxRhoVX(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float INTZFluxRhoVX(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);

__device__ float INTXFluxRhoVY(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float INTYFluxRhoVY(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float INTZFluxRhoVY(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);

__device__ float INTXFluxRhoVZ(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float INTYFluxRhoVZ(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float INTZFluxRhoVZ(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);

__device__ float INTXFluxBX(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float INTYFluxBX(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float INTZFluxBX(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);

__device__ float INTXFluxBY(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float INTYFluxBY(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float INTZFluxBY(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);

__device__ float INTXFluxBZ(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float INTYFluxBZ(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float INTZFluxBZ(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);

__device__ float INTXFluxE(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float INTYFluxE(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
__device__ float INTZFluxE(int i, int j, int k, float* rho, float* rhov_x, float *rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e, int N);
#endif