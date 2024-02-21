#ifndef KERNELS_OD_DECL
#define KERNELS_OD_DECL

#include <math.h>

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

        // Hoist fluid variables
        float t_rho = 0.0;
        float t_rhov_x = 0.0, t_rhov_y = 0.0, t_rhov_z = 0.0, KE = 0.0;
        float t_Bx = 0.0, t_By = 0.0, t_Bz = 0.0, t_B_sq = 0.0;
        float t_p = 0.0, t_e = 0.0; 

        /* 
        TODO: Declare Fluid Fluxes 
        */

        /* 
        TODO: Declare Intermediate Variables
        */

        /* 
        TODO: Declare Intermediate Fluxes
        */

        // Handle B.Cs separately
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){ // THIS LOOP ORDER IS FOR CONTIGUOUS MEMORY ACCESS
            for (int i = tidx + 1; i < Nx - 1; i += xthreads){ 
                for (int j = tidy + 1; j < Ny - 1; j += ythreads){
                    /* TODO: Precompute p, B^2, \vec{B}\dot\vec{u}, and the necesssary hoisted quantities */
                    t_rho = rho[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_rhov_x = rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_rhov_y = rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_rhov_z = rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)];
                    // t_KE = KE(); 
                    t_Bx = Bx[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_By = By[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_Bz = Bz[IDX3D(i, j, k, Nx, Ny, Nz)];
                    // t_B_sq = B_sq()
                    
                    // t_p = p(i, j, k, e, )
                    
                    /* TODO - Compute fluid fluxes */

                    /* TODO - Compute intermediate variables */

                    /* TODO - Compute intermediate fluxes */

                    /* TODO - Update fluid variables */
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

/* Flux declarations */
// Rho
__device__ float XFluxRho(const int i, const int j, const int k, const float* rhov_x, const int Nx, const int Ny, const int Nz)
    {
        return rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)];
    }
__device__ float YFluxRho(const int i, const int j, const int k, const float* rhov_y, const int Nx, const int Ny, const int Nz)
    {
        return rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)];
    }
__device__ float ZFluxRho(const int i, const int j, const int k, const float* rhov_z, const int Nx, const int Ny, const int Nz)
    {
        return rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)];
    }

// RhoVX
__device__ float XFluxRhoVX(const int i, const int j, const int k, const float* rhov_x, const float* rho, const float* Bx, 
    const float p, const float B)
    {
        /* IMPLEMENT */
    }
#endif