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
__global__ void SwapSimData(float* rho, float* rhov_x, float* rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e,
     const float* rho_np1, const float* rhovx_np1, const float* rhovy_np1, const float* rhovz_np1, 
     const float* Bx_np1, const float* By_np1, const float* Bz_np1, const float* e_np1,
     const int Nx, const int Ny, const int Nz); 

__global__ void FluidAdvance(float* rho_np1, float* rhovx_np1, float* rhovy_np1, float* rhovz_np1, float* Bx_np1, float* By_np1, float* Bz_np1, float* e_np1,
     const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, const float* Bx, const float* By, const float* Bz, const float* e, 
     float* rho_int, float* rhovx_int, float* rhovy_int, float* rhovz_int, float* Bx_int, float* By_int, float* Bz_int, float* e_int,
     const float D, const float dt, const float dx, const float dy, const float dz, 
     const int Nx, const int Ny, const int Nz);

/* DONT FORGET NUMERICAL DIFFUSION */
__global__ void BoundaryConditions(float* rho_np1, float* rhovx_np1, float* rhovy_np1, float* rhovz_np1, 
     float* Bx_np1, float* By_np1, float* Bz_np1, float* e_np1,
     const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, 
     const float* Bx, const float* By, const float* Bz, const float* e, 
     float* rho_int, float* rhovx_int, float* rhovy_int, float* rhovz_int, float* Bx_int, float* By_int, float* Bz_int, float* e_int,
     const float D, const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvRho(const int i, const int j, const int k, 
     const float* rho, 
     const float* rho_int, const float* rhovx_int, const float* rhovy_int, const float* rhovz_int,
     const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvRhoVX(const int i, const int j, const int k,
     const float* rhov_x, 
    const float* rho_int, const float* rhovx_int, const float* rhovy_int, const float* rhovz_int,
    const float* Bx_int, const float* By_int, const float* Bz_int, const float* e_int, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvRhoVY(const int i, const int j, const int k,
     const float* rhov_y, 
     const float* rho_int, const float* rhovx_int, const float* rhovy_int, const float* rhovz_int,
    const float* Bx_int, const float* By_int, const float* Bz_int, const float* e_int, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvRhoVZ(const int i, const int j, const int k,
     const float* rhov_z, 
     const float* rho_int, const float* rhovx_int, const float* rhovy_int, const float* rhovz_int, 
     const float* Bx, const float* By, const float* Bz, const float* e, 
     const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvBX(const int i, const int j, const int k,
     const float* Bx,
     const float* rho_int, const float* rhovx_int, const float* rhovy_int, const float* rhovz_int, 
     const float* Bx_int, const float* By_int, const float* Bz_int,
     const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvBY(const int i, const int j, const int k,
     const float* By,
     const float* rho_int, const float* rhovx_int, const float* rhovy_int, const float* rhovz_int, 
     const float* Bx_int, const float* By_int, const float* Bz_int,
     const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvBZ(const int i, const int j, const int k,
     const float* Bz,
     const float* rho_int, const float* rhovx_int, const float* rhovy_int, const float* rhovz_int, 
     const float* Bx_int, const float* By_int, const float* Bz_int,
     const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

__device__ float LaxWendroffAdvE(const int i, const int j, const int k,
     const float* e, 
     const float* rho_int, const float* rhovx_int, const float* rhovy_int, const float* rhovz_int,
     const float* Bx_int, const float* By_int, const float* Bz_int, const float* e_int, 
     const float dt, const float dx, const float dy, const float dz,
     const int Nx, const int Ny, const int Nz);

// FLUX FUNCTIONS - Overloaded b/c of intermediate variable calculation
// Rho - Not overloaded b/c minimal verbosity
__device__ float XFluxRho(const int i, const int j, const int k, const float* rhov_x, const int Nx, const int Ny, const int Nz);
__device__ float YFluxRho(const int i, const int j, const int k, const float* rhov_y, const int Nx, const int Ny, const int Nz);
__device__ float ZFluxRho(const int i, const int j, const int k, const float* rhov_z, const int Nx, const int Ny, const int Nz);

// RhoVX
// __device__ float XFluxRhoVX(const float rho, const float rhov_x, const float Bx, const float B_sq, const float p);
__device__ float XFluxRhoVX(const int i, const int j, const int k, 
     const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
     const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz);

// __device__ float YFluxRhoVX(const float rho, const float rhov_x, const float rhov_y, const float Bx, const float By);
__device__ float YFluxRhoVX(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, 
     const float* Bx, const float* By,
     const int Nx, const int Ny, const int Nz);

// __device__ float ZFluxRhoVX(const float rho, const float rhov_x, const float rhov_z, const float Bx, const float Bz);
__device__ float ZFluxRhoVX(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_z, 
     const float* Bx, const float* Bz,
     const int Nx, const int Ny, const int Nz);

// RhoVY
// __device__ float XFluxRhoVY(const float rho, const float rhov_x, const float rhov_y, const float Bx, const float By);
__device__ float XFluxRhoVY(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, 
     const float* Bx, const float* By,
     const int Nx, const int Ny, const int Nz);

// __device__ float YFluxRhoVY(const float rho, const float rhov_y, const float By, const float B_sq, const float p);
__device__ float YFluxRhoVY(const int i, const int j, const int k, 
     const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
     const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz);

// __device__ float ZFluxRhoVY(const float rho, const float rhov_y, const float rhov_z, const float By, const float Bz);
__device__ float ZFluxRhoVY(const int i, const int j, const int k,
     const float* rho, const float* rhov_y, const float* rhov_z, 
     const float* By, const float* Bz,
     const int Nx, const int Ny, const int Nz);

// RhoVZ
// __device__ float XFluxRhoVZ(const float rho, const float rhov_x, const float rhov_z, const float Bx, const float Bz);
__device__ float XFluxRhoVZ(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_z, 
     const float* Bx, const float* Bz,
     const int Nx, const int Ny, const int Nz);

// __device__ float YFluxRhoVZ(const float rho, const float rhov_y, const float rhov_z, const float By, const float Bz);
__device__ float YFluxRhoVZ(const int i, const int j, const int k,
     const float* rho, const float* rhov_y, const float* rhov_z,
     const float* By, const float* Bz,
     const int Nx, const int Ny, const int Nz);

// __device__ float ZFluxRhoVZ(const float rho, const float rhov_z, const float Bz, const float B_sq, const float p);
__device__ float ZFluxRhoVZ(const int i, const int j, const int k, 
     const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
     const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz);

// Bx
__device__ float XFluxBX();

// __device__ float YFluxBX(const float rho, const float rhov_x, const float rhov_y, const float Bx, const float By);
__device__ float YFluxBX(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, 
     const float* Bx, const float* By,
     const int Nx, const int Ny, const int Nz);

// __device__ float ZFluxBX(const float rho, const float rhov_x, const float rhov_z, const float Bx, const float Bz);
__device__ float ZFluxBX(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_z, 
     const float* Bx, const float* Bz,
     const int Nx, const int Ny, const int Nz);

// By
// __device__ float XFluxBY(const float rho, const float rhov_x, const float rhov_y, const float Bx, const float By);
__device__ float XFluxBY(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, 
     const float* Bx, const float* By,
     const int Nx, const int Ny, const int Nz);

__device__ float YFluxBY();

// __device__ float ZFluxBY(const float rho, const float rhov_y, const float rhov_z, const float By, const float Bz);
__device__ float ZFluxBY(const int i, const int j, const int k,
     const float* rho, const float* rhov_y, const float* rhov_z, 
     const float* By, const float* Bz,
     const int Nx, const int Ny, const int Nz);

// Bz
// __device__ float XFluxBZ(const float rho, const float rhov_x, const float rhov_z, const float Bx, const float Bz);
__device__ float XFluxBZ(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_z, 
     const float* Bx, const float* Bz,
     const int Nx, const int Ny, const int Nz);

// __device__ float YFluxBZ(const float rho, const float rhov_y, const float rhov_z, const float By, const float Bz);
__device__ float YFluxBZ(const int i, const int j, const int k,
     const float* rho, const float* rhov_y, const float* rhov_z, 
     const float* By, const float* Bz,
     const int Nx, const int Ny, const int Nz);

__device__ float ZFluxBZ();

// Energy
// __device__ float XFluxE(const float rho, const float rhov_x, const float Bx, const float e, const float p, const float B_sq, const float B_dot_u);
__device__ float XFluxE(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
     const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz);

// __device__ float YFluxE(const float rho, const float rhov_y, const float By, const float e, const float p, const float B_sq, const float B_dot_u);
__device__ float YFluxE(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
     const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz);

// __device__ float ZFluxE(const float rho, const float rhov_z, const float Bz, const float e, const float p, const float B_sq, const float B_dot_u);
__device__ float ZFluxE(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
     const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz);

// Intermediate Variables
__device__ float intRho(const int i, const int j, const int k, 
    const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVX(const int i, const int j, const int k, 
    const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
    const float* Bx, const float* By, const float* Bz, const float* e, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVY(const int i, const int j, const int k, 
    const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z, 
    const float* Bx, const float* By, const float* Bz, const float* e,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVZ(const int i, const int j, const int k, 
    const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
    const float* Bx, const float* By, const float* Bz, const float* e, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBx(const int i, const int j, const int k,
    const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
    const float* Bx, const float* By, const float* Bz,
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

__device__ float intBy(const int i, const int j, const int k, 
    const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
    const float* Bx, const float* By, const float* Bz, 
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

__device__ float intBz(const int i, const int j, const int k, 
    const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
    const float* Bx, const float* By, const float* Bz,
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

__device__ float intE(const int i, const int j, const int k, 
    const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
    const float* Bx, const float* By, const float* Bz, const float* e, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

// Helper Functions
__device__ float B_sq(int i, int j, int k, const float* Bx, const float* By, const float* Bz, 
     const int Nx, const int Ny, const int Nz); // B / \sqrt{\mu_{0}} -> B 

__device__ float p(int i, int j, int k, 
     const float* e, const float B_sq, const float KE, 
     const int Nx, const int Ny, const int Nz);

__device__ float KE(int i, int j, int k, const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z, 
     const int Nx, const int Ny, const int Nz); // \rho * \vec{u}\cdot\vec{u} * 0.5

__device__ float B_dot_u(int i, int j, int k, const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z, 
     const float* Bx, const float* By, const float* Bz, const int Nx, const int Ny, const int Nz);

__device__ float numericalDiffusion(const int i, const int j, const int k, const float* fluid_var, 
    const float D, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

#endif