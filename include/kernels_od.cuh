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

/* DONT FORGET NUMERICAL DIFFUSION */
__global__ void FluidAdvance(float* rho_np1, float* rhovx_np1, float* rhovy_np1, float* rhovz_np1, float* Bx_np1, float* By_np1, float* Bz_np1, float* e_np1,
     const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, const float* Bx, const float* By, const float* Bz, const float* e, 
     const float D, const int Nx, const int Ny, const int Nz);

/* DONT FORGET NUMERICAL DIFFUSION */
__global__ void BoundaryConditions(float* rho_np1, float* rhovx_np1, float* rhovy_np1, float* rhovz_np1, float* Bx_np1, float* By_np1, float* Bz_np1, float* e_np1,
     const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz);
     
// Flux functions
// These should all be consts - fixing as I go
__device__ float XFluxRho(const int i, const int j, const int k, const float* rhov_x, const int Nx, const int Ny, const int Nz);
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
// These should all be consts - fixing as I go
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