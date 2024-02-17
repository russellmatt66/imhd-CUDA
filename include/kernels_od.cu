#ifndef KERNELS_OD_DECL
#define KERNELS_OD_DECL

#include "kernels_od.cuh"

// row-major, column-minor order
#define IDX3D(i, j, k, Nx, Ny, Nz) (k * (Nx * Ny) + i * Nx + j)

/* DONT FORGET NUMERICAL DIFFUSION */
__global__ void FluidAdvance(float* rho_np1, float* rhovx_np1, float* rhovy_np1, float* rhovz_np1, float* Bx_np1, float* By_np1, float* Bz_np1, float* e_np1,
     const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, const float* Bx, const float* By, const float* Bz, const float* e,
     const float D, const int Nx, const int Ny, const int Nz)
    {
        // Execution configuration boilerplate
        int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;
        int xthreads = gridDim.x * blockDim.x;
        int ythreads = gridDim.y * blockDim.y;
        int zthreads = gridDim.z * blockDim.z;

        // Handle B.Cs separately
        for (int i = tidx + 1; i < Nx - 1; i += xthreads){
            for (int j = tidy + 1; j < Ny - 1; j += ythreads){
                for (int k = tidz + 1; k < Nz - 1; k += zthreads){
                    // TODO - Compute fluid fluxes

                    // TODO - Compute intermediate variables

                    // TODO - Compute intermediate fluxes

                    // TODO - Update fluid variables
                }
            }
        } 
        return;
    }

/* 
Boundary Conditions are:
(1) Rigid, Perfectly-Conducting wall
(2) Periodic in z
*/
__global__ void BoundaryConditions(float* rho_np1, float* rhovx_np1, float* rhovy_np1, float* rhovz_np1, float* Bx_np1, float* By_np1, float* Bz_np1, float* e_np1,
     const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz)
     {
        // Execution configuration boilerplate
        int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;
        int xthreads = gridDim.x * blockDim.x;
        int ythreads = gridDim.y * blockDim.y;
        int zthreads = gridDim.z * blockDim.z;

        /* B.Cs on (i, j, k = 0) */

        /* B.Cs on (i, j, k = N-1) */

        /* B.Cs on (i = 0, j, k) */
        
        /* B.Cs on (i = N-1, j, k) */

        /* B.Cs on (i, j = 0, k) */

        /* B.Cs on (i, j = N-1, k) */

        return;
     }

// Flux declarations
__device__ float XFluxRho(const int i, const int j, const int k, const float* rhov_x, const int Nx, const int Ny, const int Nz)
    {
        return rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)];
    }

#endif