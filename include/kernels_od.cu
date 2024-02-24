#ifndef KERNELS_OD_DECL
#define KERNELS_OD_DECL

#include <math.h>

#include "kernels_od.cuh"

// row-major, column-minor order
#define IDX3D(i, j, k, Nx, Ny, Nz) (k * (Nx * Ny) + i * Nx + j)

__global__ void IntermediateVarsInterior(const float* rho, 
    const float* rhov_x, const float* rhov_y, const float* rhov_z, 
    const float* Bx, const float* By, const float* Bz, const float* e,
    float* rho_int, float* rhovx_int, float* rhovy_int, float* rhovz_int,
    float* Bx_int, float* By_int, float* Bz_int, float* e_int, 
    const float dt, const float dx, const float dy, const float dz,
    const float Nx, const float Ny, const float Nz)
    {
        int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;
        int xthreads = gridDim.x * blockDim.x;
        int ythreads = gridDim.y * blockDim.y;
        int zthreads = gridDim.z * blockDim.z;

        for (int k = tidz + 1; k < Nz; k += zthreads){
            for (int i = tidx + 1; i < Nx; i += xthreads){
                for (int j = tidy + 1; j < Ny; j += ythreads){
                    /* IMPLEMENT INTERMEDIATE VARIABLE CALCULATION */
                }
            }
        }
    }

__global__ void IntermediateVarsBCs(const float* rho, 
    const float* rhov_x, const float* rhov_y, const float* rhov_z, 
    const float* Bx, const float* By, const float* Bz, const float* e,
    float* rho_int, float* rhovx_int, float* rhovy_int, float* rhovz_int,
    float* Bx_int, float* By_int, float* Bz_int, float* e_int, 
    const float dt, const float dx, const float dy, const float dz,
    const float Nx, const float Ny, const float Nz)
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
    }

/* DONT FORGET NUMERICAL DIFFUSION */
__global__ void FluidAdvance(float* rho_np1, float* rhovx_np1, float* rhovy_np1, float* rhovz_np1, float* Bx_np1, float* By_np1, float* Bz_np1, float* e_np1,
    const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, const float* Bx, const float* By, const float* Bz, const float* e, 
    float* rho_int, float* rhovx_int, float* rhovy_int, float* rhovz_int, float* Bx_int, float* By_int, float* Bz_int, float* e_int,
    const float D, const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
    {
        // Execution configuration boilerplate
        int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;
        int xthreads = gridDim.x * blockDim.x;
        int ythreads = gridDim.y * blockDim.y;
        int zthreads = gridDim.z * blockDim.z;
        
        // Handle B.Cs separately
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){ // THIS LOOP ORDER IS FOR CONTIGUOUS MEMORY ACCESS
            for (int i = tidx + 1; i < Nx - 1; i += xthreads){ 
                for (int j = tidy + 1; j < Ny - 1; j += ythreads){
                    /* Update fluid variables on interior */
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
/* Overloaded to reduce memory accesses */

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
__device__ float XFluxRhoVX(const float rho, const float rhov_x, const float Bx, const float B_sq, const float p)
    {
        return (1.0 / rho) * pow(rhov_x, 2) - pow(Bx, 2) + p + B_sq / 2.0;
    }
__device__ float XFluxRhoVX(const int i, const int j, const int k, 
     const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
     const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz)
     {
        float Bsq = B_sq(i, j, k, Bx, By, Bz, Nx, Ny, Nz);
        float ke = KE(i, j, k, rho, rhov_x, rhov_y, rhov_z, Nx, Ny, Nz);
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * pow(rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)], 2) 
            - pow(Bx[IDX3D(i, j, k, Nx, Ny, Nz)], 2) + p(i, j, k, e, Bsq, ke, Nx, Ny, Nz)
            + Bsq / 2.0;
     }


__device__ float YFluxRhoVX(const float rho, const float rhov_x, const float rhov_y, const float Bx, const float By)
    {
        return (1.0 / rho) * rhov_x * rhov_y - Bx * By;
    }
__device__ float YFluxRhoVX(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, 
     const float* Bx, const float* By,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)]
            - Bx[IDX3D(i, j, k, Nx, Ny, Nz)] * By[IDX3D(i, j, k, Nx, Ny, Nz)];
     }

__device__ float ZFluxRhoVX(const float rho, const float rhov_x, const float rhov_z, const float Bx, const float Bz)
    {
        return (1.0 / rho) * rhov_x * rhov_z - Bx * Bz;
    }
__device__ float ZFluxRhoVX(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_z, 
     const float* Bx, const float* Bz,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)]
            - Bx[IDX3D(i, j, k, Nx, Ny, Nz)] * Bz[IDX3D(i, j, k, Nx, Ny, Nz)];
     }

// RhoVY
__device__ float XFluxRhoVY(const float rho, const float rhov_x, const float rhov_y, const float Bx, const float By)
    {
        return (1.0 / rho) * rhov_x * rhov_y - Bx * By;
    }
__device__ float XFluxRhoVY(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, 
     const float* Bx, const float* By,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)]
            - Bx[IDX3D(i, j, k, Nx, Ny, Nz)] * By[IDX3D(i, j, k, Nx, Ny, Nz)];
     }

__device__ float YFluxRhoVY(const float rho, const float rhov_y, const float By, const float B_sq, const float p)
    {
        return (1.0 / rho) * pow(rhov_y, 2) - pow(By, 2) + p + B_sq / 2.0;
    }
__device__ float YFluxRhoVY(const int i, const int j, const int k, 
     const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
     const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz)
     {
        float Bsq = B_sq(i, j, k, Bx, By, Bz, Nx, Ny, Nz);
        float ke = KE(i, j, k, rho, rhov_x, rhov_y, rhov_z, Nx, Ny, Nz);
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * pow(rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)], 2) 
            - pow(By[IDX3D(i, j, k, Nx, Ny, Nz)], 2) + p(i, j, k, e, Bsq, ke, Nx, Ny, Nz)
            + Bsq / 2.0;
     }

__device__ float ZFluxRhoVY(const float rho, const float rhov_y, const float rhov_z, const float By, const float Bz)
    {
        return (1.0 / rho) * rhov_y * rhov_z - By * Bz;
    }
__device__ float ZFluxRhoVY(const int i, const int j, const int k,
     const float* rho, const float* rhov_y, const float* rhov_z, 
     const float* By, const float* Bz,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)]
            - By[IDX3D(i, j, k, Nx, Ny, Nz)] * Bz[IDX3D(i, j, k, Nx, Ny, Nz)];
     }

// RhoVZ
__device__ float XFluxRhoVZ(const float rho, const float rhov_x, const float rhov_z, const float Bx, const float Bz)
    {
        return (1.0 / rho) * rhov_x * rhov_z - Bx * Bz;
    }
__device__ float XFluxRhoVZ(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_z, 
     const float* Bx, const float* Bz,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)]
            - Bx[IDX3D(i, j, k, Nx, Ny, Nz)] * Bz[IDX3D(i, j, k, Nx, Ny, Nz)];
     }

__device__ float YFluxRhoVZ(const float rho, const float rhov_y, const float rhov_z, const float By, const float Bz)
    {
        return (1.0 / rho) * rhov_y * rhov_z - By * Bz;
    }
__device__ float YFluxRhoVZ(const int i, const int j, const int k,
     const float* rho, const float* rhov_y, const float* rhov_z,
     const float* By, const float* Bz,
     const int Nx, const int Ny, const int Nz)
    {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)]
            - By[IDX3D(i, j, k, Nx, Ny, Nz)] * Bz[IDX3D(i, j, k, Nx, Ny, Nz)];
    }

__device__ float ZFluxRhoVZ(const float rho, const float rhov_z, const float Bz, const float B_sq, const float p)
    {
        return (1.0 / rho) * pow(rhov_z, 2) - pow(Bz, 2) + p + B_sq / 2.0;
    }
__device__ float ZFluxRhoVZ(const int i, const int j, const int k, 
     const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
     const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz)
     {
        float Bsq = B_sq(i, j, k, Bx, By, Bz, Nx, Ny, Nz);
        float ke = KE(i, j, k, rho, rhov_x, rhov_y, rhov_z, Nx, Ny, Nz);
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * pow(rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)], 2) 
            - pow(Bz[IDX3D(i, j, k, Nx, Ny, Nz)], 2) + p(i, j, k, e, Bsq, ke, Nx, Ny, Nz)
            + Bsq / 2.0;
     }

// Bx
__device__ float XFluxBX()
    {
        return 0.0;
    }

__device__ float YFluxBX(const float rho, const float rhov_x, const float rhov_y, const float Bx, const float By)
    {
        return (1.0 / rho) * (rhov_x * By - Bx * rhov_y);
    }
__device__ float YFluxBX(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, 
     const float* Bx, const float* By,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) 
            * (rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] * By[IDX3D(i, j, k, Nx, Ny, Nz)] 
            - Bx[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)]);
     }

__device__ float ZFluxBX(const float rho, const float rhov_x, const float rhov_z, const float Bx, const float Bz)
    {
        return (1.0 / rho) * (rhov_x * Bz - Bx * rhov_z);
    }
__device__ float ZFluxBX(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_z, 
     const float* Bx, const float* Bz,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) 
        * (rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] * Bz[IDX3D(i, j, k, Nx, Ny, Nz)] 
        - Bx[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)]);
     }

// By
__device__ float XFluxBY(const float rho, const float rhov_x, const float rhov_y, const float Bx, const float By)
    {
        return (1.0 / rho) * (rhov_y * Bx - By * rhov_x); 
    }
__device__ float XFluxBY(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, 
     const float* Bx, const float* By,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) 
        * (rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)] * Bx[IDX3D(i, j, k, Nx, Ny, Nz)] 
        - By[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)]); 
     }

__device__ float YFluxBY()
    {
        return 0.0;
    }

__device__ float ZFluxBY(const float rho, const float rhov_y, const float rhov_z, const float By, const float Bz)
    {
        return (1.0 / rho) * (rhov_y * Bz - By * rhov_z);
    }
__device__ float ZFluxBY(const int i, const int j, const int k,
     const float* rho, const float* rhov_y, const float* rhov_z, 
     const float* By, const float* Bz,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) 
        * (rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)] * Bz[IDX3D(i, j, k, Nx, Ny, Nz)] 
        - By[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)]);
     }

// Bz
__device__ float XFluxBZ(const float rho, const float rhov_x, const float rhov_z, const float Bx, const float Bz)
    {
        return (1.0 / rho) * (rhov_z * Bx - Bz * rhov_x);
    }
__device__ float XFluxBZ(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_z, 
     const float* Bx, const float* Bz,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) 
        * (rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)] * Bx[IDX3D(i, j, k, Nx, Ny, Nz)] 
        - Bz[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)]);
     }

__device__ float YFluxBZ(const float rho, const float rhov_y, const float rhov_z, const float By, const float Bz)
    {
        return (1.0 / rho) * (rhov_z * By - Bz * rhov_y);
    }
__device__ float YFluxBZ(const int i, const int j, const int k,
     const float* rho, const float* rhov_y, const float* rhov_z, 
     const float* By, const float* Bz,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) 
        * (rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)] * By[IDX3D(i, j, k, Nx, Ny, Nz)] 
        - Bz[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)]);
     }

__device__ float ZFluxBZ()
    {
        return 0.0;
    }

// e
__device__ float XFluxE(const float rho, const float rhov_x, const float Bx, const float e, const float p, const float B_sq, const float B_dot_u)
    {
        return e + p + B_sq * (rhov_x / rho) - B_dot_u * Bx; 
    }
__device__ float XFluxE(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
     const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz)
    {
        float Bsq = B_sq(i, j, k, Bx, By, Bz, Nx, Ny, Nz);
        float ke = KE(i, j, k, rho, rhov_x, rhov_y, rhov_z, Nx, Ny, Nz);
        float pressure = p(i, j, k, e, Bsq, ke, Nx, Ny, Nz);
        float Bdotu = B_dot_u(i, j, k, 
            rho, rhov_x, rhov_y, rhov_z,
            Bx, By, Bz, 
            Nx, Ny, Nz);
        return e[IDX3D(i, j, k, Nx, Ny, Nz)] + pressure + Bsq 
            * (rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) 
            - Bdotu * Bx[IDX3D(i, j, k, Nx, Ny, Nz)]; 
     }


__device__ float YFluxE(const float rho, const float rhov_y, const float By, const float e, const float p, const float B_sq, const float B_dot_u)
    {
        return e + p + B_sq * (rhov_y / rho) - B_dot_u * By; 
    }
__device__ float YFluxE(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
     const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz)
     {
        float Bsq = B_sq(i, j, k, Bx, By, Bz, Nx, Ny, Nz);
        float ke = KE(i, j, k, rho, rhov_x, rhov_y, rhov_z, Nx, Ny, Nz);
        float pressure = p(i, j, k, e, Bsq, ke, Nx, Ny, Nz);
        float Bdotu = B_dot_u(i, j, k, 
            rho, rhov_x, rhov_y, rhov_z,
            Bx, By, Bz, 
            Nx, Ny, Nz);
        return e[IDX3D(i, j, k, Nx, Ny, Nz)] + pressure + Bsq 
            * (rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)] / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) 
            - Bdotu * By[IDX3D(i, j, k, Nx, Ny, Nz)]; 
     }

__device__ float ZFluxE(const float rho, const float rhov_z, const float Bz, const float e, const float p, const float B_sq, const float B_dot_u)
    {
        return e + p + B_sq * (rhov_z / rho) - B_dot_u * Bz; 
    }
__device__ float ZFluxE(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
     const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz)
     {
        float Bsq = B_sq(i, j, k, Bx, By, Bz, Nx, Ny, Nz);
        float ke = KE(i, j, k, rho, rhov_x, rhov_y, rhov_z, Nx, Ny, Nz);
        float pressure = p(i, j, k, e, Bsq, ke, Nx, Ny, Nz);
        float Bdotu = B_dot_u(i, j, k, 
            rho, rhov_x, rhov_y, rhov_z,
            Bx, By, Bz, 
            Nx, Ny, Nz);
        return e[IDX3D(i, j, k, Nx, Ny, Nz)] + pressure + Bsq 
            * (rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)] / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) 
            - Bdotu * Bz[IDX3D(i, j, k, Nx, Ny, Nz)]; 
     }


/* Intermediate Variables */
__device__ float intRho(const int i, const int j, const int k, 
    const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
{
    return rho[IDX3D(i, j, k, Nx, Ny, Nz)] 
        - (dt / dx) * (XFluxRho(i, j, k, rhov_x, Nx, Ny, Nz) - XFluxRho(i-1, j, k, rhov_x, Nx, Ny, Nz))
        - (dt / dy) * (YFluxRho(i, j, k, rhov_y, Nx, Ny, Nz) - YFluxRho(i, j-1, k, rhov_y, Nx, Ny, Nz))
        - (dt / dz) * (ZFluxRho(i, j, k, rhov_z, Nx, Ny, Nz) - ZFluxRho(i, j, k-1, rhov_z, Nx, Ny, Nz));
}

__device__ float intRhoVX(const int i, const int j, const int k, 
    const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
    const float* Bx, const float* By, const float* Bz, const float* e, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
{
    return rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)]
        - (dt / dx) * (XFluxRhoVX(i, j, k, rho, rhov_x, rhov_y, rhov_z, 
                            Bx, By, Bz, e, Nx, Ny, Nz) 
                        - XFluxRhoVX(i-1, j, k, rho, rhov_x, rhov_y, rhov_z, 
                            Bx, By, Bz, e, Nx, Ny, Nz))
        - (dt / dy) * (YFluxRhoVX(i, j, k, rho, rhov_x, rhov_y, Bx, By,
                            Nx, Ny, Nz)
                        - YFluxRhoVX(i, j-1, k, rho, rhov_x, rhov_y, Bx, By,
                            Nx, Ny, Nz))
        - (dt / dz) * (ZFluxRhoVX(i, j, k, rho, rhov_x, rhov_z, Bx, Bz, 
                            Nx, Ny, Nz)
                        - ZFluxRhoVX(i, j, k-1, rho, rhov_x, rhov_z, Bx, Bz, 
                            Nx, Ny, Nz));
}

__device__ float intRhoVY(const int i, const int j, const int k, 
    const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z, 
    const float* Bx, const float* By, const float* Bz, const float* e,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
{
    return rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)]
        - (dt / dx) * (XFluxRhoVY(i, j, k, rho, rhov_x, rhov_y, Bx, By,
                            Nx, Ny, Nz)
                        - XFluxRhoVY(i-1, j, k, rho, rhov_x, rhov_y, Bx, By,
                            Nx, Ny, Nz))
        - (dt / dy) * (YFluxRhoVY(i, j, k, rho, rhov_x, rhov_y, rhov_z,
                            Bx, By, Bz, e, Nx, Ny, Nz)
                        - YFluxRhoVY(i, j-1, k, rho, rhov_x, rhov_y, rhov_z,
                            Bx, By, Bz, e, Nx, Ny, Nz))
        - (dt / dz) * (ZFluxRhoVY(i, j, k, rho, rhov_y, rhov_z, 
                            By, Bz, Nx, Ny, Nz)
                        - ZFluxRhoVY(i, j, k-1, rho, rhov_y, rhov_z, 
                            By, Bz, Nx, Ny, Nz));
}

__device__ float intRhoVZ(const int i, const int j, const int k, 
    const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
    const float* Bx, const float* By, const float* Bz, const float* e, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
{   
    return rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)]
        - (dt / dx) * (XFluxRhoVZ(i, j, k, rho, rhov_x, rhov_z, Bx, Bz,
                            Nx, Ny, Nz)
                        - XFluxRhoVZ(i-1, j, k, rho, rhov_x, rhov_z, Bx, Bz,
                            Nx, Ny, Nz))
        - (dt / dy) * (YFluxRhoVZ(i, j, k, rho, rhov_y, rhov_z, By, Bz, 
                            Nx, Ny, Nz)
                        - YFluxRhoVZ(i, j-1, k, rho, rhov_y, rhov_z, By, Bz, 
                            Nx, Ny, Nz))
        - (dt / dz) * (ZFluxRhoVZ(i, j, k, rho, rhov_x, rhov_y, rhov_z, 
                            Bx, By, Bz, e, Nx, Ny, Nz)
                        - ZFluxRhoVZ(i, j, k-1, rho, rhov_x, rhov_y, rhov_z, 
                            Bx, By, Bz, e, Nx, Ny, Nz));
}

__device__ float intBx(const int i, const int j, const int k,
    const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
    const float* Bx, const float* By, const float* Bz,
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
{
    return Bx[IDX3D(i, j, k, Nx, Ny, Nz)]
        - (dt / dx) * (XFluxBX() - XFluxBX())
        - (dt / dy) * (YFluxBX(i, j, k, rho, rhov_x, rhov_y, Bx, By, 
                            Nx, Ny, Nz)
                        - YFluxBX(i, j-1, k, rho, rhov_x, rhov_y, Bx, By, 
                            Nx, Ny, Nz))
        - (dt / dz) * (ZFluxBX(i, j, k, rho, rhov_x, rhov_z, Bx, Bz, 
                            Nx, Ny, Nz)
                        - ZFluxBX(i, j, k-1, rho, rhov_x, rhov_z, Bx, Bz, 
                            Nx, Ny, Nz));
}

__device__ float intBy(const int i, const int j, const int k, 
    const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
    const float* Bx, const float* By, const float* Bz, 
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
{
    return By[IDX3D(i, j, k, Nx, Ny, Nz)]
        - (dt / dx) * (XFluxBY(i, j, k, rho, rhov_x, rhov_y, Bx, By, 
                            Nx, Ny, Nz) 
                        - XFluxBY(i-1, j, k, rho, rhov_x, rhov_y, Bx, By, 
                            Nx, Ny, Nz))
        - (dt / dy) * (YFluxBY()- YFluxBY())
        - (dt / dz) * (ZFluxBY(i, j, k, rho, rhov_y, rhov_z, By, Bz, 
                            Nx, Ny, Nz)
                        - ZFluxBY(i, j, k-1, rho, rhov_y, rhov_z, By, Bz, 
                            Nx, Ny, Nz));
}

__device__ float intBz(const int i, const int j, const int k, 
    const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
    const float* Bx, const float* By, const float* Bz,
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
{
    return Bz[IDX3D(i, j, k, Nx, Ny, Nz)]
        - (dt / dx) * (XFluxBZ(i, j, k, rho, rhov_x, rhov_z, Bx, Bz, 
                            Nx, Ny, Nz) 
                        - XFluxBZ(i-1, j, k, rho, rhov_x, rhov_z, Bx, Bz, 
                            Nx, Ny, Nz))
        - (dt / dy) * (YFluxBZ(i, j, k, rho, rhov_y, rhov_z, By, Bz, 
                            Nx, Ny, Nz)
                        - YFluxBZ(i, j-1, k, rho, rhov_y, rhov_z, By, Bz,
                            Nx, Ny, Nz))
        - (dt / dz) * (ZFluxBZ() - ZFluxBZ());
}

__device__ float intE(const int i, const int j, const int k, 
    const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
    const float* Bx, const float* By, const float* Bz, const float* e, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
{
    return e[IDX3D(i, j, k, Nx, Ny, Nz)]
        - (dt / dx) * (XFluxE(i, j, k, rho, rhov_x, rhov_y, rhov_z,
                            Bx, By, Bz, e, Nx, Ny, Nz)
                        - XFluxE(i-1, j, k, rho, rhov_x, rhov_y, rhov_z,
                            Bx, By, Bz, e, Nx, Ny, Nz))
        - (dt / dy) * (YFluxE(i, j, k, rho, rhov_x, rhov_y, rhov_z, 
                            Bx, By, Bz, e, Nx, Ny, Nz)
                        - YFluxE(i, j-1, k, rho, rhov_x, rhov_y, rhov_z, 
                            Bx, By, Bz, e, Nx, Ny, Nz))
        - (dt / dz) * (ZFluxE(i, j, k, rho, rhov_x, rhov_y, rhov_z, 
                            Bx, By, Bz, e, Nx, Ny, Nz)
                        - ZFluxE(i, j, k-1, rho, rhov_x, rhov_y, rhov_z, 
                            Bx, By, Bz, e, Nx, Ny, Nz));
}

/* 
Intermediate Flux declarations 
(Aren't these just flux functions w / intermediate variables?)
*/


/* B-squared, etc. */
__device__ float B_sq(const int i, const int j, const int k, const float* Bx, const float* By, const float* Bz, 
    const int Nx, const int Ny, const int Nz)
    {
        return pow(Bx[IDX3D(i, j, k, Nx, Ny, Nz)], 2) + pow(By[IDX3D(i, j, k, Nx, Ny, Nz)], 2) + pow(Bz[IDX3D(i, j, k, Nx, Ny, Nz)], 2);
    }

__device__ float p(const int i, const int j, const int k, const float* e, const float B_sq, const float KE, 
    const int Nx, const int Ny, const int Nz)
    {
        return (gamma - 1.0) * (e[IDX3D(i, j, k, Nx, Ny, Nz)] - KE - B_sq / 2.0);
    }

__device__ float KE(const int i, const int j, const int k, const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
    const int Nx, const int Ny, const int Nz)
    {
        float KE = (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * (
            pow(rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)], 2) + pow(rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)], 2) + pow(rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)], 2));
        return KE;
    }

__device__ float B_dot_u(const int i, const int j, const int k, const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
    const float* Bx, const float* By, const float* Bz, const int Nx, const int Ny, const int Nz)
    {
        float B_dot_u = 0.0;
        B_dot_u = (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * (rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] * Bx[IDX3D(i, j, k, Nx, Ny, Nz)]
            + rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)] * By[IDX3D(i, j, k, Nx, Ny, Nz)] + rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)] * Bz[IDX3D(i, j, k, Nx, Ny, Nz)]);
        return B_dot_u;
    }
#endif