#ifndef KERNELS_OD_DECL
#define KERNELS_OD_DECL

#include <math.h>

#include "kernels_od.cuh"

// row-major, column-minor order
#define IDX3D(i, j, k, Nx, Ny, Nz) (k * (Nx * Ny) + i * Ny + j)

// Advances fluid variables on interior of the grid, and adds a numerical diffusion
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

                    /* Calculate intermediate variables */
                    rho_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intRho(i, j, k, rho, rhov_x, rhov_y, rhov_z,
                                                            dt, dx, dy, dz, Nx, Ny, Nz);
                    rhovx_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intRhoVX(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e,
                                                            dt, dx, dy, dz, Nx, Ny, Nz);
                    rhovy_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intRhoVY(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e,
                                                            dt, dx, dy, dz, Nx, Ny, Nz);
                    rhovz_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intRhoVZ(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e,
                                                            dt, dx, dy, dz, Nx, Ny, Nz);
                    Bx_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intBx(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, 
                                                            dt, dx, dy, dz, Nx, Ny, Nz);
                    By_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intBy(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, 
                                                            dt, dx, dy, dz, Nx, Ny, Nz);
                    Bz_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intBz(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, 
                                                            dt, dx, dy, dz, Nx, Ny, Nz);                
                    e_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intE(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, 
                                                            dt, dx, dy, dz, Nx, Ny, Nz);                

                    /* Update fluid variables on interior */
                    rho_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvRho(i, j, k, rho, 
                                                            rho_int, rhovx_int, rhovy_int, rhovz_int,
                                                            dt, dx, dy, dz, Nx, Ny, Nz)
                                                        + numericalDiffusion(i, j, k, rho, D, Nx, Ny, Nz);

                    rhovx_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvRhoVX(i, j, k, rhov_x, 
                                                                rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, 
                                                                dt, dx, dy, dz, Nx, Ny, Nz)
                                                        + numericalDiffusion(i, j, k, rhov_x, D, Nx, Ny, Nz);

                    rhovy_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvRhoVY(i, j, k, rhov_y, 
                                                                rho_int, rhovx_int, rhovy_int, rhovz_int,
                                                                Bx_int, By_int, Bz_int, e_int, 
                                                                dt, dx, dy, dz, Nx, Ny, Nz)
                                                        + numericalDiffusion(i, j, k, rhov_y, D, Nx, Ny, Nz);

                    rhovz_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvRhoVZ(i, j, k, rhov_z, 
                                                                rho_int, rhovx_int, rhovy_int, rhovz_int,
                                                                Bx_int, By_int, Bz_int, e_int, 
                                                                dt, dx, dy, dz, Nx, Ny, Nz)
                                                        + numericalDiffusion(i, j, k, rhov_z, D, Nx, Ny, Nz);

                    Bx_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvBX(i, j, k, Bx, 
                                                                rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, 
                                                                dt, dx, dy, dz, Nx, Ny, Nz)
                                                        + numericalDiffusion(i, j, k, Bx, D, Nx, Ny, Nz);

                    By_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvBY(i, j, k, By, 
                                                                rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int,
                                                                dt, dx, dy, dz, Nx, Ny, Nz)
                                                        + numericalDiffusion(i, j, k, By, D, Nx, Ny, Nz);

                    Bz_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvBZ(i, j, k, Bz,
                                                                rho_int, rhovx_int, rhovy_int, rhovz_int,
                                                                Bx_int, By_int, Bz_int, 
                                                                dt, dx, dy, dz, Nx, Ny, Nz)
                                                        + numericalDiffusion(i, j, k, Bz, D, Nx, Ny, Nz);

                    e_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvE(i, j, k, e, 
                                                                rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, 
                                                                dt, dx, dy, dz, Nx, Ny, Nz)
                                                        + numericalDiffusion(i, j, k, e, D, Nx, Ny, Nz);

                }
            }
        }
        return;
    }

/* 
Boundary Conditions are:
(1) Rigid, Perfectly-Conducting wall
(2) Periodic in z

Currently O(N) wasted work in this section due to overlap b/w fixed-regions
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

        /* 
        Periodic B.Cs on (i, j, k = 0) 
        FRONT 
        (I)
        */
        for (int i = tidx; i < Nx; i += xthreads){
            for (int j = tidy; j < Ny; j += ythreads){
                /* IMPLEMENT */
            }
        }
        /* 
        Periodic B.Cs on (i, j, k = N-1) 
        BACK
        (VI)
        */
        for (int i = tidx; i < Nx; i += xthreads){
            for (int j = tidy; j < Ny; j += ythreads){
                /* IMPLEMENT */
            }
        }
        
        /* 
        B.Cs on (i = 0, j, k) 
        BOTTOM
        (II)
        */
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){ // Let the PBCs on the front and back face handle the corners
            for (int j = tidy; j < Ny; j += ythreads){
                /* IMPLEMENT */
            }
        }
        
        /* 
        B.Cs on (i = N-1, j, k) 
        TOP
        (IV)
        */
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){
            for (int j = tidy; j < Ny; j += ythreads){
                /* IMPLEMENT */
            }
        }
        
        /* 
        B.Cs on (i, j = 0, k) 
        LEFT
        (V)
        */
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){
            for (int i = tidx; i < Nx; i += xthreads){
                /* IMPLEMENT */
            }
        }
        /* 
        B.Cs on (i, j = N-1, k) 
        RIGHT
        (III)
        */
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){
            for (int i = tidy; i < Nx; i += xthreads){
                /* IMPLEMENT */
            }
        }
        return;
     }

/* LAX WENDROFF ADVANCES */
__device__ float LaxWendroffAdvRho(const int i, const int j, const int k, 
    const float* rho, 
    const float* rho_int, const float* rhovx_int, const float* rhovy_int, const float* rhovz_int, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return 0.5 * (rho[IDX3D(i, j, k, Nx, Ny, Nz)] + rho_int[IDX3D(i, j, k, Nx, Ny, Nz)])
                - 0.5 * (dt / dx) * (XFluxRho(i+1, j, k, rhovx_int, Nx, Ny, Nz) - XFluxRho(i, j, k, rhovx_int, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxRho(i, j+1, k, rhovy_int, Nx, Ny, Nz) - YFluxRho(i, j, k, rhovy_int, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxRho(i, j, k+1, rhovz_int, Nx, Ny, Nz) - ZFluxRho(i, j, k, rhovz_int, Nx, Ny, Nz));
    }

__device__ float LaxWendroffAdvRhoVX(const int i, const int j, const int k,
    const float* rhov_x, 
    const float* rho_int, const float* rhovx_int, const float* rhovy_int, const float* rhovz_int,
    const float* Bx_int, const float* By_int, const float* Bz_int, const float* e_int, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return 0.5 * (rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] + rhovx_int[IDX3D(i, j, k, Nx, Ny, Nz)])
                - 0.5 * (dt / dx) * (XFluxRhoVX(i+1, j, k, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                        Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                    - XFluxRhoVX(i, j, k, rho_int, rhovx_int, rhovy_int, rhovz_int,
                                        Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxRhoVX(i, j+1, k, rho_int, rhovx_int, rhovy_int, Bx_int, By_int, 
                                        Nx, Ny, Nz)
                                    - YFluxRhoVX(i, j, k, rho_int, rhovx_int, rhovy_int, Bx_int, By_int, 
                                        Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxRhoVX(i, j, k+1, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int, 
                                        Nx, Ny, Nz)
                                    - ZFluxRhoVX(i, j, k, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int, 
                                        Nx, Ny, Nz));                                        
    }

__device__ float LaxWendroffAdvRhoVY(const int i, const int j, const int k,
    const float* rhov_y, 
    const float* rho_int, const float* rhovx_int, const float* rhovy_int, const float* rhovz_int,
    const float* Bx_int, const float* By_int, const float* Bz_int, const float* e_int, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return 0.5 * (rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)] + rhovy_int[IDX3D(i, j, k, Nx, Ny, Nz)])
                - 0.5 * (dt / dx) * (XFluxRhoVY(i+1, j, k, rho_int, rhovx_int, rhovy_int, Bx_int, By_int,
                                        Nx, Ny, Nz)
                                    - XFluxRhoVY(i, j, k, rho_int, rhovx_int, rhovy_int, Bx_int, By_int,
                                        Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxRhoVY(i, j+1, k, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                        Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                    - YFluxRhoVY(i, j, k, rho_int, rhovx_int, rhovy_int, rhovz_int,
                                        Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxRhoVY(i, j, k+1, rho_int, rhovy_int, rhovz_int, By_int, Bz_int,
                                        Nx, Ny, Nz));
    }

__device__ float LaxWendroffAdvRhoVZ(const int i, const int j, const int k,
    const float* rhov_z, 
    const float* rho_int, const float* rhovx_int, const float* rhovy_int, const float* rhovz_int,
    const float* Bx_int, const float* By_int, const float* Bz_int, const float* e_int, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return 0.5 * (rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)] + rhovz_int[IDX3D(i, j, k, Nx, Ny, Nz)])
                - 0.5 * (dt / dx) * (XFluxRhoVZ(i+1, j, k, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int, 
                                        Nx, Ny, Nz)
                                    - XFluxRhoVZ(i, j, k, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int,
                                        Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxRhoVZ(i, j+1, k, rho_int, rhovy_int, rhovz_int, By_int, Bz_int,
                                        Nx, Ny, Nz)
                                    - YFluxRhoVZ(i, j, k, rho_int, rhovy_int, rhovz_int, By_int, Bz_int,
                                        Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxRhoVZ(i, j, k+1, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                        Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz));
    }

__device__ float LaxWendroffAdvBX(const int i, const int j, const int k,
    const float* Bx,
    const float* rho_int, const float* rhovx_int, const float* rhovy_int, const float* rhovz_int, 
    const float* Bx_int, const float* By_int, const float* Bz_int,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return 0.5 * (Bx[IDX3D(i, j, k, Nx, Ny, Nz)] + Bx_int[IDX3D(i, j, k, Nx, Ny, Nz)])
                - 0.5 * (dt / dx) * (XFluxBX() - XFluxBX())
                - 0.5 * (dt / dy) * (YFluxBX(i, j+1, k, rho_int, rhovx_int, rhovy_int, Bx_int, By_int, 
                                        Nx, Ny, Nz)
                                    - YFluxBX(i, j, k, rho_int, rhovx_int, rhovy_int, Bx_int, By_int,
                                        Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxBX(i, j, k+1, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int,
                                        Nx, Ny, Nz)
                                    - ZFluxBX(i, j, k, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int,
                                        Nx, Ny, Nz));
    }

__device__ float LaxWendroffAdvBY(const int i, const int j, const int k,
    const float* By,
    const float* rho_int, const float* rhovx_int, const float* rhovy_int, const float* rhovz_int, 
    const float* Bx_int, const float* By_int, const float* Bz_int,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return 0.5 * (By[IDX3D(i, j, k, Nx, Ny, Nz)] + By_int[IDX3D(i, j, k, Nx, Ny, Nz)])
                - 0.5 * (dt / dx) * (XFluxBY(i+1, j, k, rho_int, rhovx_int, rhovy_int, Bx_int, By_int, 
                                        Nx, Ny, Nz)
                                    - XFluxBY(i, j, k, rho_int, rhovx_int, rhovy_int, Bx_int, By_int,
                                        Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxBY() - YFluxBY())
                - 0.5 * (dt / dz) * (ZFluxBY(i, j, k+1, rho_int, rhovy_int, rhovz_int, By_int, Bz_int, 
                                        Nx, Ny, Nz)
                                    - ZFluxBY(i, j, k, rho_int, rhovy_int, rhovz_int, By_int, Bz_int,
                                        Nx, Ny, Nz));
    }

__device__ float LaxWendroffAdvBZ(const int i, const int j, const int k,
    const float* Bz,
    const float* rho_int, const float* rhovx_int, const float* rhovy_int, const float* rhovz_int, 
    const float* Bx_int, const float* By_int, const float* Bz_int,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return 0.5 * (Bz[IDX3D(i, j, k, Nx, Ny, Nz)] + Bz_int[IDX3D(i, j, k, Nx, Ny, Nz)])
                - 0.5 * (dt / dx) * (XFluxBZ(i+1, j, k, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int, 
                                        Nx, Ny, Nz)
                                    - XFluxBZ(i, j, k, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int,
                                        Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxBZ(i, j+1, k, rho_int, rhovy_int, rhovz_int, By_int, Bz_int, 
                                        Nx, Ny, Nz)
                                    - YFluxBZ(i, j, k, rho_int, rhovy_int, rhovz_int, By_int, Bz_int, 
                                        Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxBZ() - ZFluxBZ());
    }

__device__ float LaxWendroffAdvE(const int i, const int j, const int k,
    const float* e, 
    const float* rho_int, const float* rhovx_int, const float* rhovy_int, const float* rhovz_int,
    const float* Bx_int, const float* By_int, const float* Bz_int, const float* e_int, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return 0.5 * (e[IDX3D(i, j, k, Nx, Ny, Nz)] + e_int[IDX3D(i, j, k, Nx, Ny, Nz)])
                - 0.5 * (dt / dx) * (XFluxE(i+1, j, k, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                        Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                    - XFluxE(i, j, k, rho_int, rhovx_int, rhovy_int, rhovz_int,
                                        Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxE(i, j+1, k, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                        Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                    - YFluxE(i, j, k, rho_int, rhovx_int, rhovx_int, rhovz_int, 
                                        Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxE(i, j, k+1, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                        Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz));
    }

/* 
Flux Functions 
Overloaded to reduce memory accesses 
*/
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
// __device__ float XFluxRhoVX(const float rho, const float rhov_x, const float Bx, const float B_sq, const float p)
//     {
//         return (1.0 / rho) * pow(rhov_x, 2) - pow(Bx, 2) + p + B_sq / 2.0;
//     }
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


// __device__ float YFluxRhoVX(const float rho, const float rhov_x, const float rhov_y, const float Bx, const float By)
//     {
//         return (1.0 / rho) * rhov_x * rhov_y - Bx * By;
//     }
__device__ float YFluxRhoVX(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, 
     const float* Bx, const float* By,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)]
            - Bx[IDX3D(i, j, k, Nx, Ny, Nz)] * By[IDX3D(i, j, k, Nx, Ny, Nz)];
     }

// __device__ float ZFluxRhoVX(const float rho, const float rhov_x, const float rhov_z, const float Bx, const float Bz)
//     {
//         return (1.0 / rho) * rhov_x * rhov_z - Bx * Bz;
//     }
__device__ float ZFluxRhoVX(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_z, 
     const float* Bx, const float* Bz,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)]
            - Bx[IDX3D(i, j, k, Nx, Ny, Nz)] * Bz[IDX3D(i, j, k, Nx, Ny, Nz)];
     }

// RhoVY
// __device__ float XFluxRhoVY(const float rho, const float rhov_x, const float rhov_y, const float Bx, const float By)
//     {
//         return (1.0 / rho) * rhov_x * rhov_y - Bx * By;
//     }
__device__ float XFluxRhoVY(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, 
     const float* Bx, const float* By,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)]
            - Bx[IDX3D(i, j, k, Nx, Ny, Nz)] * By[IDX3D(i, j, k, Nx, Ny, Nz)];
     }

// __device__ float YFluxRhoVY(const float rho, const float rhov_y, const float By, const float B_sq, const float p)
//     {
//         return (1.0 / rho) * pow(rhov_y, 2) - pow(By, 2) + p + B_sq / 2.0;
//     }
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

// __device__ float ZFluxRhoVY(const float rho, const float rhov_y, const float rhov_z, const float By, const float Bz)
//     {
//         return (1.0 / rho) * rhov_y * rhov_z - By * Bz;
//     }
__device__ float ZFluxRhoVY(const int i, const int j, const int k,
     const float* rho, const float* rhov_y, const float* rhov_z, 
     const float* By, const float* Bz,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)]
            - By[IDX3D(i, j, k, Nx, Ny, Nz)] * Bz[IDX3D(i, j, k, Nx, Ny, Nz)];
     }

// RhoVZ
// __device__ float XFluxRhoVZ(const float rho, const float rhov_x, const float rhov_z, const float Bx, const float Bz)
//     {
//         return (1.0 / rho) * rhov_x * rhov_z - Bx * Bz;
//     }
__device__ float XFluxRhoVZ(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_z, 
     const float* Bx, const float* Bz,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)]
            - Bx[IDX3D(i, j, k, Nx, Ny, Nz)] * Bz[IDX3D(i, j, k, Nx, Ny, Nz)];
     }

// __device__ float YFluxRhoVZ(const float rho, const float rhov_y, const float rhov_z, const float By, const float Bz)
//     {
//         return (1.0 / rho) * rhov_y * rhov_z - By * Bz;
//     }
__device__ float YFluxRhoVZ(const int i, const int j, const int k,
     const float* rho, const float* rhov_y, const float* rhov_z,
     const float* By, const float* Bz,
     const int Nx, const int Ny, const int Nz)
    {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)]
            - By[IDX3D(i, j, k, Nx, Ny, Nz)] * Bz[IDX3D(i, j, k, Nx, Ny, Nz)];
    }

// __device__ float ZFluxRhoVZ(const float rho, const float rhov_z, const float Bz, const float B_sq, const float p)
//     {
//         return (1.0 / rho) * pow(rhov_z, 2) - pow(Bz, 2) + p + B_sq / 2.0;
//     }
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

// __device__ float YFluxBX(const float rho, const float rhov_x, const float rhov_y, const float Bx, const float By)
//     {
//         return (1.0 / rho) * (rhov_x * By - Bx * rhov_y);
//     }
__device__ float YFluxBX(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, 
     const float* Bx, const float* By,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) 
            * (rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] * By[IDX3D(i, j, k, Nx, Ny, Nz)] 
            - Bx[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)]);
     }

// __device__ float ZFluxBX(const float rho, const float rhov_x, const float rhov_z, const float Bx, const float Bz)
//     {
//         return (1.0 / rho) * (rhov_x * Bz - Bx * rhov_z);
//     }
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
// __device__ float XFluxBY(const float rho, const float rhov_x, const float rhov_y, const float Bx, const float By)
//     {
//         return (1.0 / rho) * (rhov_y * Bx - By * rhov_x); 
//     }
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

// __device__ float ZFluxBY(const float rho, const float rhov_y, const float rhov_z, const float By, const float Bz)
//     {
//         return (1.0 / rho) * (rhov_y * Bz - By * rhov_z);
//     }
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
// __device__ float XFluxBZ(const float rho, const float rhov_x, const float rhov_z, const float Bx, const float Bz)
//     {
//         return (1.0 / rho) * (rhov_z * Bx - Bz * rhov_x);
//     }
__device__ float XFluxBZ(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_z, 
     const float* Bx, const float* Bz,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) 
        * (rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)] * Bx[IDX3D(i, j, k, Nx, Ny, Nz)] 
        - Bz[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)]);
     }

// __device__ float YFluxBZ(const float rho, const float rhov_y, const float rhov_z, const float By, const float Bz)
//     {
//         return (1.0 / rho) * (rhov_z * By - Bz * rhov_y);
//     }
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
// __device__ float XFluxE(const float rho, const float rhov_x, const float Bx, const float e, const float p, const float B_sq, const float B_dot_u)
//     {
//         return e + p + B_sq * (rhov_x / rho) - B_dot_u * Bx; 
//     }
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


// __device__ float YFluxE(const float rho, const float rhov_y, const float By, const float e, const float p, const float B_sq, const float B_dot_u)
//     {
//         return e + p + B_sq * (rhov_y / rho) - B_dot_u * By; 
//     }
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

// __device__ float ZFluxE(const float rho, const float rhov_z, const float Bz, const float e, const float p, const float B_sq, const float B_dot_u)
//     {
//         return e + p + B_sq * (rhov_z / rho) - B_dot_u * Bz; 
//     }
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
Helper Functions
B-squared, pressure, Kinetic Energy, 2nd derivative central difference, etc. 
*/
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

// Adds numerical diffusion to the interior
// 2nd-order central difference
__device__ float numericalDiffusion(const int i, const int j, const int k, const float* fluid_var, 
    const float D, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        float num_diff = 0.0;
        num_diff = D * (
            (1.0 / pow(dx, 2)) * (fluid_var[IDX3D(i+1, j, k, Nx, Ny, Nz)] - 2.0*fluid_var[IDX3D(i, j, k, Nx, Ny, Nz)] + fluid_var[IDX3D(i-1, j, k, Nx, Ny, Nz)])
            + (1.0 / pow(dy, 2)) * (fluid_var[IDX3D(i, j+1, k, Nx, Ny, Nz)] - 2.0*fluid_var[IDX3D(i, j, k, Nx, Ny, Nz)] + fluid_var[IDX3D(i, j-1, k, Nx, Ny, Nz)])
            + (1.0 / pow(dz, 2)) * (fluid_var[IDX3D(i, j, k+1, Nx, Ny, Nz)] - 2.0*fluid_var[IDX3D(i, j, k, Nx, Ny, Nz)] + fluid_var[IDX3D(i, j, k-1, Nx, Ny, Nz)])
            );
        return num_diff;
    }

/* UNCERTAIN IF BELOW IS NECESSARY */
// __global__ void IntermediateVarsInterior(const float* rho, 
//     const float* rhov_x, const float* rhov_y, const float* rhov_z, 
//     const float* Bx, const float* By, const float* Bz, const float* e,
//     float* rho_int, float* rhovx_int, float* rhovy_int, float* rhovz_int,
//     float* Bx_int, float* By_int, float* Bz_int, float* e_int, 
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
//         int tidy = threadIdx.y + blockDim.y * blockIdx.y;
//         int tidz = threadIdx.z + blockDim.z * blockIdx.z;
//         int xthreads = gridDim.x * blockDim.x;
//         int ythreads = gridDim.y * blockDim.y;
//         int zthreads = gridDim.z * blockDim.z;

//         for (int k = tidz + 1; k < Nz; k += zthreads){ // This nested loop order implements contiguous memory access
//             for (int i = tidx + 1; i < Nx; i += xthreads){
//                 for (int j = tidy + 1; j < Ny; j += ythreads){
//                     /* IMPLEMENT INTERMEDIATE VARIABLE CALCULATION */
//                     rho_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intRho(i, j, k, rho, rhov_x, rhov_y, rhov_z,
//                                                             dt, dx, dy, dz, Nx, Ny, Nz);
//                     rhovx_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intRhoVX(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e,
//                                                             dt, dx, dy, dz, Nx, Ny, Nz);
//                     rhovy_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intRhoVY(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e,
//                                                             dt, dx, dy, dz, Nx, Ny, Nz);
//                     rhovz_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intRhoVZ(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e,
//                                                             dt, dx, dy, dz, Nx, Ny, Nz);
//                     Bx_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intBx(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, 
//                                                             dt, dx, dy, dz, Nx, Ny, Nz);
//                     By_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intBy(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, 
//                                                             dt, dx, dy, dz, Nx, Ny, Nz);
//                     Bz_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intBz(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, 
//                                                             dt, dx, dy, dz, Nx, Ny, Nz);                
//                     e_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intE(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, 
//                                                             dt, dx, dy, dz, Nx, Ny, Nz);                
//                 }
//             }
//         }
//     }

// __global__ void IntermediateVarsBoundary(const float* rho, 
//     const float* rhov_x, const float* rhov_y, const float* rhov_z, 
//     const float* Bx, const float* By, const float* Bz, const float* e,
//     float* rho_int, float* rhovx_int, float* rhovy_int, float* rhovz_int,
//     float* Bx_int, float* By_int, float* Bz_int, float* e_int, 
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         // Execution configuration boilerplate
//         int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
//         int tidy = threadIdx.y + blockDim.y * blockIdx.y;
//         int tidz = threadIdx.z + blockDim.z * blockIdx.z;
//         int xthreads = gridDim.x * blockDim.x;
//         int ythreads = gridDim.y * blockDim.y;
//         int zthreads = gridDim.z * blockDim.z;

//         /* B.Cs on (i, j, k = 0) */

//         /* B.Cs on (i, j, k = N-1) */

//         /* B.Cs on (i = 0, j, k) */
        
//         /* B.Cs on (i = N-1, j, k) */

//         /* B.Cs on (i, j = 0, k) */

//         /* B.Cs on (i, j = N-1, k) */
//     }

#endif