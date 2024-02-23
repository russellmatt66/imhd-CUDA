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
#define q_e 1.6 * pow(10,-19) // [C]
#define m 1.67 * pow(10, -27) // [kg]

/* DONT FORGET NUMERICAL DIFFUSION */
__global__ void FluidAdvance(float* rho_np1, float* rhovx_np1, float* rhovy_np1, float* rhovz_np1, float* Bx_np1, float* By_np1, float* Bz_np1, float* e_np1,
     const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, const float* Bx, const float* By, const float* Bz, const float* e, 
     const float D, const float dt, const float dx, const float dy, const float dz, 
     const int Nx, const int Ny, const int Nz);

/* DONT FORGET NUMERICAL DIFFUSION */
__global__ void BoundaryConditions(
     float* rho_np1, float* rhovx_np1, float* rhovy_np1, float* rhovz_np1, 
     float* Bx_np1, float* By_np1, float* Bz_np1, float* e_np1,
     const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, 
     const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz);

// Flux functions - Overloaded b/c of intermediate variable calculation
/* TODO: IMPLEMENT OVERLOADED FUNCTIONS */
// Rho - Not overloaded b/c minimal verbosity
__device__ float XFluxRho(const int i, const int j, const int k, const float* rhov_x, const int Nx, const int Ny, const int Nz);
__device__ float YFluxRho(const int i, const int j, const int k, const float* rhov_y, const int Nx, const int Ny, const int Nz);
__device__ float ZFluxRho(const int i, const int j, const int k, const float* rhov_z, const int Nx, const int Ny, const int Nz);

// RhoVX
__device__ float XFluxRhoVX(const float rho, const float rhov_x, const float Bx, const float B_sq, const float p);
__device__ float XFluxRhoVX(const int i, const int j, const int k, 
     const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
     const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz);

__device__ float YFluxRhoVX(const float rho, const float rhov_x, const float rhov_y, const float Bx, const float By);
__device__ float YFluxRhoVX(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, 
     const float* Bx, const float* By,
     const int Nx, const int Ny, const int Nz);

__device__ float ZFluxRhoVX(const float rho, const float rhov_x, const float rhov_z, const float Bx, const float Bz);
__device__ float ZFluxRhoVX(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_z, 
     const float* Bx, const float* Bz,
     const int Nx, const int Ny, const int Nz);

// RhoVY
__device__ float XFluxRhoVY(const float rho, const float rhov_x, const float rhov_y, const float Bx, const float By);
__device__ float XFluxRhoVY(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, 
     const float* Bx, const float* By,
     const int Nx, const int Ny, const int Nz);

__device__ float YFluxRhoVY(const float rho, const float rhov_y, const float By, const float B_sq, const float p);
__device__ float YFluxRhoVY(const int i, const int j, const int k, 
     const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
     const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz);

__device__ float ZFluxRhoVY(const float rho, const float rhov_y, const float rhov_z, const float By, const float Bz);
__device__ float ZFluxRhoVY(const int i, const int j, const int k,
     const float* rho, const float* rhov_y, const float* rhov_z, 
     const float* By, const float* Bz,
     const int Nx, const int Ny, const int Nz);

// RhoVZ
__device__ float XFluxRhoVZ(const float rho, const float rhov_x, const float rhov_z, const float Bx, const float Bz);
__device__ float XFluxRhoVZ(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_z, 
     const float* Bx, const float* Bz,
     const int Nx, const int Ny, const int Nz);

__device__ float YFluxRhoVZ(const float rho, const float rhov_y, const float rhov_z, const float By, const float Bz);
__device__ float YFluxRhoVZ(const int i, const int j, const int k,
     const float* rho, const float* rhov_y, const float* rhov_z,
     const float* By, const float* Bz,
     const int Nx, const int Ny, const int Nz);

__device__ float ZFluxRhoVZ(const float rho, const float rhov_z, const float Bz, const float B_sq, const float p);
__device__ float ZFluxRhoVZ(const int i, const int j, const int k, 
     const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
     const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz);

// Bx
__device__ float XFluxBX();

__device__ float YFluxBX(const float rho, const float rhov_x, const float rhov_y, const float Bx, const float By);
__device__ float YFluxBX(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, 
     const float* Bx, const float* By,
     const int Nx, const int Ny, const int Nz);

__device__ float ZFluxBX(const float rho, const float rhov_x, const float rhov_z, const float Bx, const float Bz);
__device__ float ZFluxBX(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_z, 
     const float* Bx, const float* Bz,
     const int Nx, const int Ny, const int Nz);

// By
__device__ float XFluxBY(const float rho, const float rhov_x, const float rhov_y, const float Bx, const float By);
__device__ float XFluxBY(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, 
     const float* Bx, const float* By,
     const int Nx, const int Ny, const int Nz);

__device__ float YFluxBY();

__device__ float ZFluxBY(const float rho, const float rhov_y, const float rhov_z, const float By, const float Bz);
__device__ float ZFluxBY(const int i, const int j, const int k,
     const float* rho, const float* rhov_y, const float* rhov_z, 
     const float* By, const float* Bz,
     const int Nx, const int Ny, const int Nz);

// Bz
__device__ float XFluxBZ(const float rho, const float rhov_x, const float rhov_z, const float Bx, const float Bz);
__device__ float XFluxBZ(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_z, 
     const float* Bx, const float* Bz,
     const int Nx, const int Ny, const int Nz);

__device__ float YFluxBZ(const float rho, const float rhov_y, const float rhov_z, const float By, const float Bz);
__device__ float YFluxBZ(const int i, const int j, const int k,
     const float* rho, const float* rhov_y, const float* rhov_z, 
     const float* By, const float* Bz,
     const int Nx, const int Ny, const int Nz);

__device__ float ZFluxBZ();

// energy
__device__ float XFluxE(const float rho, const float rhov_x, const float Bx, const float e, const float p, const float B_sq, const float B_dot_u);
__device__ float XFluxE(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
     const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz);

__device__ float YFluxE(const float rho, const float rhov_y, const float By, const float e, const float p, const float B_sq, const float B_dot_u);
__device__ float YFluxE(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
     const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz);

__device__ float ZFluxE(const float rho, const float rhov_z, const float Bz, const float e, const float p, const float B_sq, const float B_dot_u);
__device__ float ZFluxE(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
     const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz);

// Intermediate flux functions
// These should all be consts - fixing as I go
/* Aren't these just regular flux functions with intermediate variables as arguments? */
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

// Precomputed values
__device__ float B_sq(int i, int j, int k, const float* Bx, const float* By, const float* Bz, 
     const int Nx, const int Ny, const int Nz); // B / \sqrt{\mu_{0}} -> B 

__device__ float p(int i, int j, int k, const float* e, const float B_sq, const float KE, const int Nx, const int Ny, const int Nz);

__device__ float KE(int i, int j, int k, const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z, 
     const int Nx, const int Ny, const int Nz); // \rho * \vec{u}\cdot\vec{u} * 0.5

__device__ float B_dot_u(int i, int j, int k, const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z, 
     const float* Bx, const float* By, const float* Bz, const int Nx, const int Ny, const int Nz);
#endif