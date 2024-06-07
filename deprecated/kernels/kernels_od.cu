#ifndef KERNELS_OD_DECL
#define KERNELS_OD_DECL

#include <math.h>

#include "kernels_od.cuh"

// row-major, column-minor order
#define IDX3D(i, j, k, Nx, Ny, Nz) (k * (Nx * Ny) + i * Ny + j)

/* REQUIRES TOO MANY REGISTERS FOR 1024 THREADS PER THREADBLOCK */
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
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){ // THIS LOOP ORDER IS FOR CONTIGUOUS MEMORY ACCESS, i.e., MEMORY COALESCING
            for (int i = tidx + 1; i < Nx - 1; i += xthreads){ 
                for (int j = tidy + 1; j < Ny - 1; j += ythreads){

                    /* Calculate intermediate variables */
                    rho_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intRho(i, j, k, rho, rhov_x, rhov_y, rhov_z,
                                                            dt, dx, dy, dz, Nx, Ny, Nz); // 12 AO, 7 MR
                    rhovx_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intRhoVX(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e,
                                                            dt, dx, dy, dz, Nx, Ny, Nz); // 84 AO, 43 MR
                    rhovy_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intRhoVY(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e,
                                                            dt, dx, dy, dz, Nx, Ny, Nz); // 84 AO, 43 MR
                    rhovz_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intRhoVZ(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e,
                                                            dt, dx, dy, dz, Nx, Ny, Nz); // 84 AO, 43 MR
                    Bx_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intBx(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, 
                                                            dt, dx, dy, dz, Nx, Ny, Nz); // 32 AO, 21 MR
                    By_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intBy(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, 
                                                            dt, dx, dy, dz, Nx, Ny, Nz); // 32 AO, 21 MR
                    Bz_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intBz(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, 
                                                            dt, dx, dy, dz, Nx, Ny, Nz); // 32 AO, 21 MR                
                    e_int[IDX3D(i, j, k, Nx, Ny, Nz)] = intE(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, 
                                                            dt, dx, dy, dz, Nx, Ny, Nz); // 198 AO, 115 MR                

                    /* Update fluid variables on interior */
                    // 34 AO, 17 MR
                    rho_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvRho(i, j, k, rho, 
                                                            rho_int, rhovx_int, rhovy_int, rhovz_int,
                                                            dt, dx, dy, dz, Nx, Ny, Nz)
                                                        + numericalDiffusion(i, j, k, rho, D, dx, dy, dz, Nx, Ny, Nz);
                    // 106 AO, 53 MR
                    rhovx_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvRhoVX(i, j, k, rhov_x, 
                                                                rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, 
                                                                dt, dx, dy, dz, Nx, Ny, Nz)
                                                        + numericalDiffusion(i, j, k, rhov_x, D, dx, dy, dz, Nx, Ny, Nz);
                    // 106 AO, 53 MR
                    rhovy_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvRhoVY(i, j, k, rhov_y, 
                                                                rho_int, rhovx_int, rhovy_int, rhovz_int,
                                                                Bx_int, By_int, Bz_int, e_int, 
                                                                dt, dx, dy, dz, Nx, Ny, Nz)
                                                        + numericalDiffusion(i, j, k, rhov_y, D, dx, dy, dz, Nx, Ny, Nz);
                    // 106 AO, 53 MR
                    rhovz_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvRhoVZ(i, j, k, rhov_z, 
                                                                rho_int, rhovx_int, rhovy_int, rhovz_int,
                                                                Bx_int, By_int, Bz_int, e_int, 
                                                                dt, dx, dy, dz, Nx, Ny, Nz)
                                                        + numericalDiffusion(i, j, k, rhov_z, D, dx, dy, dz, Nx, Ny, Nz);
                    // 54 AO, 31 MR
                    Bx_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvBX(i, j, k, Bx, 
                                                                rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, 
                                                                dt, dx, dy, dz, Nx, Ny, Nz)
                                                        + numericalDiffusion(i, j, k, Bx, D, dx, dy, dz, Nx, Ny, Nz);
                    // 54 AO, 31 MR
                    By_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvBY(i, j, k, By, 
                                                                rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int,
                                                                dt, dx, dy, dz, Nx, Ny, Nz)
                                                        + numericalDiffusion(i, j, k, By, D, dx, dy, dz, Nx, Ny, Nz);
                    // 54 AO, 31 MR
                    Bz_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvBZ(i, j, k, Bz,
                                                                rho_int, rhovx_int, rhovy_int, rhovz_int,
                                                                Bx_int, By_int, Bz_int, 
                                                                dt, dx, dy, dz, Nx, Ny, Nz)
                                                        + numericalDiffusion(i, j, k, Bz, D, dx, dy, dz, Nx, Ny, Nz);
                    // 220 AO, 125 MR
                    e_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvE(i, j, k, e, 
                                                                rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, 
                                                                dt, dx, dy, dz, Nx, Ny, Nz)
                                                        + numericalDiffusion(i, j, k, e, D, dx, dy, dz, Nx, Ny, Nz);

                }
            }
        }
        return;
    }

/* 
Boundary Conditions are:
(1) Rigid, Perfectly-Conducting wall on top, bottom, and sides
(2) Periodic in z
*/
__global__ void BoundaryConditions(float* rho_np1, float* rhovx_np1, float* rhovy_np1, float* rhovz_np1, 
    float* Bx_np1, float* By_np1, float* Bz_np1, float* e_np1,
    const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, 
    const float* Bx, const float* By, const float* Bz, const float* e, 
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

        /* 
        Periodic B.Cs on (i, j, k = 0) and (i, j, k = N - 1)
        FRONT (I)
        BACK (VI) 
        */
        for (int i = tidx + 1; i < Nx - 1; i += xthreads){
            for (int j = tidy + 1; j < Ny - 1; j += ythreads){
                /* NEED TO SOLVE IMHD EQUATIONS FIRST ON BOTH FACES, CONSIDERING PERIODICITY */
                // (1) Calculate Intermediate Variables at the relevant locations
                // k = 0
                rho_int[IDX3D(i, j, 0, Nx, Ny, Nz)] = rho[IDX3D(i, j, 0, Nx, Ny, Nz)] 
                                                        - (dt / dx) * (XFluxRho(i, j, 0, rhov_x, Nx, Ny, Nz) - XFluxRho(i - 1, j, 0, rhov_x, Nx, Ny, Nz))
                                                        - (dt / dy) * (YFluxRho(i, j, 0, rhov_y, Nx, Ny, Nz) - YFluxRho(i, j - 1, 0, rhov_y, Nx, Ny, Nz))
                                                        - (dt / dz) * (ZFluxRho(i, j, 0, rhov_z, Nx, Ny, Nz) - ZFluxRho(i, j, Nz - 2, rhov_z, Nx, Ny, Nz));
                
                rhovx_int[IDX3D(i, j, 0, Nx, Ny, Nz)] = rhov_x[IDX3D(i, j, 0, Nx, Ny, Nz)]
                                                        - (dt / dx) * (
                                                            XFluxRhoVX(i, j, 0, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                            - XFluxRhoVX(i - 1, j, 0, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                            )
                                                        - (dt / dy) * (
                                                            YFluxRhoVX(i, j, 0, rho, rhov_x, rhov_y, Bx, By, Nx, Ny, Nz)
                                                            - YFluxRhoVX(i, j - 1, 0, rho, rhov_x, rhov_y, Bx, By, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dz) * (
                                                            ZFluxRhoVX(i, j, 0, rho, rhov_x, rhov_z, Bx, Bz, Nx, Ny, Nz)
                                                            - ZFluxRhoVX(i, j, Nz - 2, rho, rhov_x, rhov_z, Bx, Bz, Nx, Ny, Nz)
                                                        );
                
                rhovy_int[IDX3D(i, j, 0, Nx, Ny, Nz)] = rhov_y[IDX3D(i, j, 0, Nx, Ny, Nz)]
                                                        - (dt / dx) * (
                                                            XFluxRhoVY(i, j, 0, rho, rhov_x, rhov_y, Bx, By, Nx, Ny, Nz)
                                                            - XFluxRhoVY(i - 1, j, 0, rho, rhov_x, rhov_y, Bx, By, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dy) * (
                                                            YFluxRhoVY(i, j, 0, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                            - YFluxRhoVY(i, j - 1, 0, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dz) * (
                                                            ZFluxRhoVY(i, j, 0, rho, rhov_y, rhov_z, By, Bz, Nx, Ny, Nz)
                                                            - ZFluxRhoVY(i, j, Nz - 2, rho, rhov_y, rhov_z, By, Bz, Nx, Ny, Nz)
                                                        );

                rhovz_int[IDX3D(i, j, 0, Nx, Ny, Nz)] = rhov_z[IDX3D(i, j, 0, Nx, Ny, Nz)]
                                                        - (dt / dx) * (
                                                            XFluxRhoVZ(i, j, 0, rho, rhov_x, rhov_z, Bx, Bz, Nx, Ny, Nz)
                                                            - XFluxRhoVZ(i - 1, j, 0, rho, rhov_x, rhov_z, Bx, Bz, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dy) * (
                                                            YFluxRhoVZ(i, j, 0, rho, rhov_y, rhov_z, By, Bz, Nx, Ny, Nz)
                                                            - YFluxRhoVZ(i, j - 1, 0, rho, rhov_y, rhov_z, By, Bz, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dz) * (
                                                            ZFluxRhoVZ(i, j, 0, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                            - ZFluxRhoVZ(i, j, Nz - 2, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                        );

                Bx_int[IDX3D(i, j, 0, Nx, Ny, Nz)] = Bx[IDX3D(i, j, 0, Nx, Ny, Nz)]
                                                        - (dt / dx) * (
                                                            XFluxBX() - XFluxBX()
                                                        )
                                                        - (dt / dy) * (
                                                            YFluxBX(i, j, 0, rho, rhov_x, rhov_y, Bx, By, Nx, Ny, Nz)
                                                            - YFluxBX(i, j - 1, 0, rho, rhov_x, rhov_y, Bx, By, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dz) * (
                                                            ZFluxBX(i, j, 0, rho, rhov_x, rhov_z, Bx, Bz, Nx, Ny, Nz)
                                                            - ZFluxBX(i, j, Nz - 2, rho, rhov_x, rhov_z, Bx, Bz, Nx, Ny, Nz)
                                                        );

                By_int[IDX3D(i, j, 0, Nx, Ny, Nz)] = By[IDX3D(i, j, 0, Nx, Ny, Nz)]
                                                        - (dt / dx) * (
                                                            XFluxBY(i, j, 0, rho, rhov_x, rhov_y, Bx, By, Nx, Ny, Nz)
                                                            - XFluxBY(i - 1, j, 0, rho, rhov_x, rhov_y, Bx, By, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dy) * (
                                                            YFluxBY() - YFluxBY()
                                                        )
                                                        - (dt / dz) * (
                                                            ZFluxBY(i, j, 0, rho, rhov_y, rhov_z, By, Bz, Nx, Ny, Nz)
                                                            - ZFluxBY(i, j, Nz - 2, rho, rhov_y, rhov_z, By, Bz, Nx, Ny, Nz)
                                                        );

                Bz_int[IDX3D(i, j, 0, Nx, Ny, Nz)] = Bz[IDX3D(i, j, 0, Nx, Ny, Nz)]
                                                        - (dt / dx) * (
                                                            XFluxBZ(i, j, 0, rho, rhov_x, rhov_z, Bx, Bz, Nx, Ny, Nz)
                                                            - XFluxBZ(i - 1, j, 0, rho, rhov_x, rhov_z, Bx, Bz, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dy) * (
                                                            YFluxBZ(i, j, 0, rho, rhov_y, rhov_z, By, Bz, Nx, Ny, Nz)
                                                            - YFluxBZ(i, j - 1, 0, rho, rhov_y, rhov_z, By, Bz, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dz) * (
                                                            ZFluxBZ() - ZFluxBZ()
                                                        );

                e_int[IDX3D(i, j, 0, Nx, Ny, Nz)] = e[IDX3D(i, j, 0, Nx, Ny, Nz)]
                                                        - (dt / dx) * (
                                                            XFluxE(i, j, 0, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                            - XFluxE(i - 1, j, 0, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dy) * (
                                                            YFluxE(i, j, 0, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                            - YFluxE(i, j - 1, 0, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dz) * (
                                                            ZFluxE(i, j, 0, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                            - ZFluxE(i, j, Nz - 2, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                        );
                
                // k = Nz - 1
                rho_int[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = rho[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] 
                                                        - (dt / dx) * (XFluxRho(i, j, Nz - 1, rhov_x, Nx, Ny, Nz) - XFluxRho(i - 1, j, Nz - 1, rhov_x, Nx, Ny, Nz))
                                                        - (dt / dy) * (YFluxRho(i, j, Nz - 1, rhov_y, Nx, Ny, Nz) - YFluxRho(i, j - 1, Nz - 1, rhov_y, Nx, Ny, Nz))
                                                        - (dt / dz) * (ZFluxRho(i, j, Nz - 1, rhov_z, Nx, Ny, Nz) - ZFluxRho(i, j, Nz - 2, rhov_z, Nx, Ny, Nz));
                
                rhovx_int[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = rhov_x[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)]
                                                        - (dt / dx) * (
                                                            XFluxRhoVX(i, j, Nz - 1, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                            - XFluxRhoVX(i - 1, j, Nz - 1, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                            )
                                                        - (dt / dy) * (
                                                            YFluxRhoVX(i, j, Nz - 1, rho, rhov_x, rhov_y, Bx, By, Nx, Ny, Nz)
                                                            - YFluxRhoVX(i, j - 1, Nz - 1, rho, rhov_x, rhov_y, Bx, By, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dz) * (
                                                            ZFluxRhoVX(i, j, Nz - 1, rho, rhov_x, rhov_z, Bx, Bz, Nx, Ny, Nz)
                                                            - ZFluxRhoVX(i, j, Nz - 2, rho, rhov_x, rhov_z, Bx, Bz, Nx, Ny, Nz)
                                                        );
                
                rhovy_int[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = rhov_y[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)]
                                                        - (dt / dx) * (
                                                            XFluxRhoVY(i, j, Nz - 1, rho, rhov_x, rhov_y, Bx, By, Nx, Ny, Nz)
                                                            - XFluxRhoVY(i - 1, j, Nz - 1, rho, rhov_x, rhov_y, Bx, By, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dy) * (
                                                            YFluxRhoVY(i, j, Nz - 1, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                            - YFluxRhoVY(i, j - 1, Nz - 1, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dz) * (
                                                            ZFluxRhoVY(i, j, Nz - 1, rho, rhov_y, rhov_z, By, Bz, Nx, Ny, Nz)
                                                            - ZFluxRhoVY(i, j, Nz - 2, rho, rhov_y, rhov_z, By, Bz, Nx, Ny, Nz)
                                                        );

                rhovz_int[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = rhov_z[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)]
                                                        - (dt / dx) * (
                                                            XFluxRhoVZ(i, j, Nz - 1, rho, rhov_x, rhov_z, Bx, Bz, Nx, Ny, Nz)
                                                            - XFluxRhoVZ(i - 1, j, Nz - 1, rho, rhov_x, rhov_z, Bx, Bz, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dy) * (
                                                            YFluxRhoVZ(i, j, Nz - 1, rho, rhov_y, rhov_z, By, Bz, Nx, Ny, Nz)
                                                            - YFluxRhoVZ(i, j - 1, Nz - 1, rho, rhov_y, rhov_z, By, Bz, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dz) * (
                                                            ZFluxRhoVZ(i, j, Nz - 1, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                            - ZFluxRhoVZ(i, j, Nz - 2, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                        );

                Bx_int[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = Bx[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)]
                                                        - (dt / dx) * (
                                                            XFluxBX() - XFluxBX()
                                                        )
                                                        - (dt / dy) * (
                                                            YFluxBX(i, j, Nz - 1, rho, rhov_x, rhov_y, Bx, By, Nx, Ny, Nz)
                                                            - YFluxBX(i, j - 1, Nz - 1, rho, rhov_x, rhov_y, Bx, By, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dz) * (
                                                            ZFluxBX(i, j, Nz - 1, rho, rhov_x, rhov_z, Bx, Bz, Nx, Ny, Nz)
                                                            - ZFluxBX(i, j, Nz - 2, rho, rhov_x, rhov_z, Bx, Bz, Nx, Ny, Nz)
                                                        );

                By_int[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = By[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)]
                                                        - (dt / dx) * (
                                                            XFluxBY(i, j, Nz - 1, rho, rhov_x, rhov_y, Bx, By, Nx, Ny, Nz)
                                                            - XFluxBY(i - 1, j, Nz - 1, rho, rhov_x, rhov_y, Bx, By, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dy) * (
                                                            YFluxBY() - YFluxBY()
                                                        )
                                                        - (dt / dz) * (
                                                            ZFluxBY(i, j, Nz - 1, rho, rhov_y, rhov_z, By, Bz, Nx, Ny, Nz)
                                                            - ZFluxBY(i, j, Nz - 2, rho, rhov_y, rhov_z, By, Bz, Nx, Ny, Nz)
                                                        );

                Bz_int[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = Bz[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)]
                                                        - (dt / dx) * (
                                                            XFluxBZ(i, j, Nz - 1, rho, rhov_x, rhov_z, Bx, Bz, Nx, Ny, Nz)
                                                            - XFluxBZ(i - 1, j, Nz - 1, rho, rhov_x, rhov_z, Bx, Bz, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dy) * (
                                                            YFluxBZ(i, j, Nz - 1, rho, rhov_y, rhov_z, By, Bz, Nx, Ny, Nz)
                                                            - YFluxBZ(i, j - 1, Nz - 1, rho, rhov_y, rhov_z, By, Bz, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dz) * (
                                                            ZFluxBZ() - ZFluxBZ()
                                                        );

                e_int[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = e[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)]
                                                        - (dt / dx) * (
                                                            XFluxE(i, j, Nz - 1, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                            - XFluxE(i - 1, j, Nz - 1, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dy) * (
                                                            YFluxE(i, j, Nz - 1, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                            - YFluxE(i, j - 1, Nz - 1, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                        )
                                                        - (dt / dz) * (
                                                            ZFluxE(i, j, Nz - 1, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                            - ZFluxE(i, j, Nz - 2, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, Nx, Ny, Nz)
                                                        );  

                // (2) Calculate the updated state of the plasma at the given location  
                // k = 0
                rho_np1[IDX3D(i, j, 0, Nx, Ny, Nz)] = 0.5 * (rho[IDX3D(i, j, 0, Nx, Ny, Nz)] + rho_int[IDX3D(i, j, 0, Nx, Ny, Nz)])
                                                        - 0.5 * (dt / dx) * (
                                                            XFluxRho(i + 1, j, 0, rhovx_int, Nx, Ny, Nz)
                                                            - XFluxRho(i, j, 0, rhovx_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dy) * (
                                                            YFluxRho(i, j + 1, 0, rhovy_int, Nx, Ny, Nz)
                                                            - YFluxRho(i, j, 0, rhovy_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dz) * (
                                                            ZFluxRho(i, j, 1, rhovz_int, Nx, Ny, Nz)
                                                            - ZFluxRho(i, j, 0, rhovz_int, Nx, Ny, Nz)
                                                        )
                                                        - numericalDiffusionFront(i, j, rho, D, dx, dy, dz, Nx, Ny, Nz); // Periodic boundary conditions
                
                rhovx_np1[IDX3D(i, j, 0, Nx, Ny, Nz)] = 0.5 * (rhov_x[IDX3D(i, j, 0, Nx, Ny, Nz)] + rhovx_int[IDX3D(i, j, 0, Nx, Ny, Nz)])
                                                        - 0.5 * (dt / dx) * (
                                                            XFluxRhoVX(i, j, 0, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                            - XFluxRhoVX(i + 1, j, 0, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dy) * (
                                                            YFluxRhoVX(i, j + 1, 0, rho_int, rhovx_int, rhovy_int, Bx_int, By_int, Nx, Ny, Nz)
                                                            - YFluxRhoVX(i, j, 0, rho_int, rhovx_int, rhovy_int, Bx_int, By_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dz) * (
                                                            ZFluxRhoVX(i, j, 1, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int, Nx, Ny, Nz)
                                                            - ZFluxRhoVX(i, j, 0, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int, Nx, Ny, Nz)
                                                        )
                                                        - numericalDiffusionFront(i, j, rhov_x, D, dx, dy, dz, Nx, Ny, Nz); // Periodic boundary conditions
                
                rhovy_np1[IDX3D(i, j, 0, Nx, Ny, Nz)] = 0.5 * (rhov_y[IDX3D(i, j, 0, Nx, Ny, Nz)] + rhovy_int[IDX3D(i, j, 0, Nx, Ny, Nz)])
                                                        - 0.5 * (dt / dx) * (
                                                            XFluxRhoVY(i + 1, j, 0, rho_int, rhovx_int, rhovy_int, Bx_int, By_int, Nx, Ny, Nz)
                                                            - XFluxRhoVY(i, j, 0, rho_int, rhovx_int, rhovy_int, Bx_int, By_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dy) * (
                                                            YFluxRhoVY(i, j + 1, 0, rho_int, rhovx_int, rhovy_int, rhovz_int,
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                            - YFluxRhoVY(i, j, 0, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dz) * (
                                                            ZFluxRhoVY(i, j, 1, rho_int, rhovy_int, rhovz_int, By_int, Bz_int, Nx, Ny, Nz)
                                                            - ZFluxRhoVY(i, j, 0, rho_int, rhovy_int, rhovz_int, By_int, Bz_int, Nx, Ny, Nz)
                                                        )
                                                        - numericalDiffusionFront(i, j, rhov_y, D, dx, dy, dz, Nx, Ny, Nz); // Periodic boundary conditions

                rhovz_np1[IDX3D(i, j, 0, Nx, Ny, Nz)] = 0.5 * (rhov_z[IDX3D(i, j, 0, Nx, Ny, Nz)] + rhovz_int[IDX3D(i, j, 0, Nx, Ny, Nz)])
                                                        - 0.5 * (dt / dx) * (
                                                            XFluxRhoVZ(i + 1, j, 0, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int, Nx, Ny, Nz)
                                                            - XFluxRhoVZ(i, j, 0, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dy) * (
                                                            YFluxRhoVZ(i, j + 1, 0, rho_int, rhovy_int, rhovz_int, By_int, Bz_int, Nx, Ny, Nz)
                                                            - YFluxRhoVZ(i, j, 0, rho_int, rhovy_int, rhovz_int, By_int, Bz_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dz) * (
                                                            ZFluxRhoVZ(i, j, 1, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                            - ZFluxRhoVZ(i, j, 0, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                        )
                                                        - numericalDiffusionFront(i, j, rhov_z, D, dx, dy, dz, Nx, Ny, Nz); // Periodic boundary conditions

                Bx_np1[IDX3D(i, j, 0, Nx, Ny, Nz)] = 0.5 * (Bx[IDX3D(i, j, 0, Nx, Ny, Nz)] + Bx_int[IDX3D(i, j, 0, Nx, Ny, Nz)])
                                                        - 0.5 * (dt / dx) * (
                                                            XFluxBX() - XFluxBX()
                                                        )
                                                        - 0.5 * (dt / dy) * (
                                                            YFluxBX(i, j + 1, 0, rho_int, rhovx_int, rhovy_int, Bx_int, By_int, Ny, Nx, Nz)
                                                            - YFluxBX(i, j, 0, rho_int, rhovx_int, rhovy_int, Bx_int, By_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dz) * (
                                                            ZFluxBX(i, j, 1, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int, Nx, Ny, Nz)
                                                            - ZFluxBX(i, j, 0, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int, Nx, Ny, Nz)
                                                        )
                                                        - numericalDiffusionFront(i, j, Bx, D, dx, dy, dz, Nx, Ny, Nz); // Periodic boundary conditions

                By_np1[IDX3D(i, j, 0, Nx, Ny, Nz)] = 0.5 * (By[IDX3D(i, j, 0, Nx, Ny, Nz)] + By_int[IDX3D(i, j, 0, Nx, Ny, Nz)])
                                                        - 0.5 * (dt / dx) * (
                                                            XFluxBY(i + 1, j, 0, rho_int, rhovx_int, rhovy_int, Bx_int, By_int, Nx, Ny, Nz)
                                                            - XFluxBY(i, j, 0, rho_int, rhovx_int, rhovy_int, Bx_int, By_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dy) * (
                                                            YFluxBY() - YFluxBY()
                                                        )
                                                        - 0.5 * (dt / dz) * (
                                                            ZFluxBY(i, j, 1, rho_int, rhovy_int, rhovz_int, By_int, Bz_int, Nx, Ny, Nz)
                                                            - ZFluxBY(i, j, 0, rho_int, rhovy_int, rhovz_int, By_int, Bz_int, Nx, Ny, Nz)
                                                        )
                                                        - numericalDiffusionFront(i, j, By, D, dx, dy, dz, Nx, Ny, Nz); // Periodic boundary conditions

                Bz_np1[IDX3D(i, j, 0, Nx, Ny, Nz)] = 0.5 * (Bz[IDX3D(i, j, 0, Nx, Ny, Nz)] + Bz_int[IDX3D(i, j, 0, Nx, Ny, Nz)]) 
                                                        - 0.5 * (dt / dx) * (
                                                            XFluxBZ(i + 1, j, 0, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int, Nx, Ny, Nz)
                                                            - XFluxBZ(i, j, 0, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dy) * (
                                                            YFluxBZ(i, j + 1, 0, rho_int, rhovy_int, rhovz_int, By_int, Bz_int, Nx, Ny, Nz)
                                                            - YFluxBZ(i, j, 0, rho_int, rhovy_int, rhovz_int, By_int, Bz_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dz) * (
                                                            ZFluxBZ() - ZFluxBZ()
                                                        )
                                                        - numericalDiffusionFront(i, j, Bz, D, dx, dy, dz, Nx, Ny, Nz); // Periodic boundary conditions

                e_np1[IDX3D(i, j, 0, Nx, Ny, Nz)] = 0.5 * (e[IDX3D(i, j, 0, Nx, Ny, Nz)] + e_int[IDX3D(i, j, 0, Nx, Ny, Nz)])
                                                        - 0.5 * (dt / dx) * (
                                                            XFluxE(i + 1, j, 0, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                            - XFluxE(i, j, 0, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dy) * (
                                                            YFluxE(i, j + 1, 0, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                            - YFluxE(i, j, 0, rho_int, rhovx_int, rhovy_int, rhovz_int,
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dz) * (
                                                            ZFluxE(i, j, 1, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                            - ZFluxE(i, j, 0, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                        )
                                                        - numericalDiffusionFront(i, j, e, D, dx, dy, dz, Nx, Ny, Nz); // Periodic boundary conditions;

                // k = Nz - 1
                rho_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = 0.5 * (rho[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] + rho_int[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)])
                                                        - 0.5 * (dt / dx) * (
                                                            XFluxRho(i + 1, j, Nz - 1, rhovx_int, Nx, Ny, Nz)
                                                            - XFluxRho(i, j, Nz - 1, rhovx_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dy) * (
                                                            YFluxRho(i, j + 1, Nz - 1, rhovy_int, Nx, Ny, Nz)
                                                            - YFluxRho(i, j, Nz - 1, rhovy_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dz) * (
                                                            ZFluxRho(i, j, 1, rhovz_int, Nx, Ny, Nz)
                                                            - ZFluxRho(i, j, Nz - 1, rhovz_int, Nx, Ny, Nz)
                                                        )
                                                        - numericalDiffusionBack(i, j, rho, D, dx, dy, dz, Nx, Ny, Nz); // Periodic boundary conditions
                
                rhovx_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = 0.5 * (rhov_x[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] + rhovx_int[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)])
                                                        - 0.5 * (dt / dx) * (
                                                            XFluxRhoVX(i, j, Nz - 1, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                            - XFluxRhoVX(i + 1, j, Nz - 1, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dy) * (
                                                            YFluxRhoVX(i, j + 1, Nz - 1, rho_int, rhovx_int, rhovy_int, Bx_int, By_int, Nx, Ny, Nz)
                                                            - YFluxRhoVX(i, j, Nz -1, rho_int, rhovx_int, rhovy_int, Bx_int, By_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dz) * (
                                                            ZFluxRhoVX(i, j, 1, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int, Nx, Ny, Nz)
                                                            - ZFluxRhoVX(i, j, Nz - 1, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int, Nx, Ny, Nz)
                                                        )
                                                        - numericalDiffusionBack(i, j, rhov_x, D, dx, dy, dz, Nx, Ny, Nz); // Periodic boundary conditions
                
                rhovy_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = 0.5 * (rhov_y[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] + rhovy_int[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)])
                                                        - 0.5 * (dt / dx) * (
                                                            XFluxRhoVY(i + 1, j, Nz - 1, rho_int, rhovx_int, rhovy_int, Bx_int, By_int, Nx, Ny, Nz)
                                                            - XFluxRhoVY(i, j, Nz - 1, rho_int, rhovx_int, rhovy_int, Bx_int, By_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dy) * (
                                                            YFluxRhoVY(i, j + 1, Nz - 1, rho_int, rhovx_int, rhovy_int, rhovz_int,
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                            - YFluxRhoVY(i, j, Nz - 1, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dz) * (
                                                            ZFluxRhoVY(i, j, 1, rho_int, rhovy_int, rhovz_int, By_int, Bz_int, Nx, Ny, Nz)
                                                            - ZFluxRhoVY(i, j, Nz - 1, rho_int, rhovy_int, rhovz_int, By_int, Bz_int, Nx, Ny, Nz)
                                                        )
                                                        - numericalDiffusionBack(i, j, rhov_y, D, dx, dy, dz, Nx, Ny, Nz); // Periodic boundary conditions

                rhovz_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = 0.5 * (rhov_z[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] + rhovz_int[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)])
                                                        - 0.5 * (dt / dx) * (
                                                            XFluxRhoVZ(i + 1, j, Nz - 1, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int, Nx, Ny, Nz)
                                                            - XFluxRhoVZ(i, j, Nz - 1, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dy) * (
                                                            YFluxRhoVZ(i, j + 1, Nz - 1, rho_int, rhovy_int, rhovz_int, By_int, Bz_int, Nx, Ny, Nz)
                                                            - YFluxRhoVZ(i, j, Nz - 1, rho_int, rhovy_int, rhovz_int, By_int, Bz_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dz) * (
                                                            ZFluxRhoVZ(i, j, 1, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                            - ZFluxRhoVZ(i, j, Nz - 1, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                        )
                                                        - numericalDiffusionBack(i, j, rhov_z, D, dx, dy, dz, Nx, Ny, Nz); // Periodic boundary conditions


                Bx_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = 0.5 * (Bx[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] + Bx_int[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)])
                                                        - 0.5 * (dt / dx) * (
                                                            XFluxBX() - XFluxBX()
                                                        )
                                                        - 0.5 * (dt / dy) * (
                                                            YFluxBX(i, j + 1, Nz - 1, rho_int, rhovx_int, rhovy_int, Bx_int, By_int, Ny, Nx, Nz)
                                                            - YFluxBX(i, j, Nz - 1, rho_int, rhovx_int, rhovy_int, Bx_int, By_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dz) * (
                                                            ZFluxBX(i, j, 1, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int, Nx, Ny, Nz)
                                                            - ZFluxBX(i, j, Nz - 1, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int, Nx, Ny, Nz)
                                                        )
                                                        - numericalDiffusionBack(i, j, Bx, D, dx, dy, dz, Nx, Ny, Nz); // Periodic boundary conditions

                By_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = 0.5 * (By[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] + By_int[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)])
                                                        - 0.5 * (dt / dx) * (
                                                            XFluxBY(i + 1, j, Nz - 1, rho_int, rhovx_int, rhovy_int, Bx_int, By_int, Nx, Ny, Nz)
                                                            - XFluxBY(i, j, Nz - 1, rho_int, rhovx_int, rhovy_int, Bx_int, By_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dy) * (
                                                            YFluxBY() - YFluxBY()
                                                        )
                                                        - 0.5 * (dt / dz) * (
                                                            ZFluxBY(i, j, 1, rho_int, rhovy_int, rhovz_int, By_int, Bz_int, Nx, Ny, Nz)
                                                            - ZFluxBY(i, j, Nz - 1, rho_int, rhovy_int, rhovz_int, By_int, Bz_int, Nx, Ny, Nz)
                                                        )
                                                        - numericalDiffusionBack(i, j, By, D, dx, dy, dz, Nx, Ny, Nz); // Periodic boundary conditions

                Bz_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = 0.5 * (Bz[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] + Bz_int[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)])
                                                        - 0.5 * (dt / dx) * (
                                                            XFluxBZ(i + 1, j, Nz - 1, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int, Nx, Ny, Nz)
                                                            - XFluxBZ(i, j, Nz - 1, rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dy) * (
                                                            YFluxBZ(i, j + 1, Nz - 1, rho_int, rhovy_int, rhovz_int, By_int, Bz_int, Nx, Ny, Nz)
                                                            - YFluxBZ(i, j, Nz - 1, rho_int, rhovy_int, rhovz_int, By_int, Bz_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dz) * (
                                                            ZFluxBZ() - ZFluxBZ()
                                                        )
                                                        - numericalDiffusionBack(i, j, Bz, D, dx, dy, dz, Nx, Ny, Nz); // Periodic boundary conditions

                e_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = 0.5 * (e[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] + e_int[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)])
                                                        - 0.5 * (dt / dx) * (
                                                            XFluxE(i + 1, j, Nz - 1, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                            - XFluxE(i, j, Nz - 1, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dy) * (
                                                            YFluxE(i, j + 1, Nz - 1, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                            - YFluxE(i, j, Nz - 1, rho_int, rhovx_int, rhovy_int, rhovz_int,
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                        )
                                                        - 0.5 * (dt / dz) * (
                                                            ZFluxE(i, j, 1, rho_int, rhovx_int, rhovy_int, rhovz_int, 
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                            - ZFluxE(i, j, Nz - 1, rho_int, rhovx_int, rhovy_int, rhovz_int,
                                                                Bx_int, By_int, Bz_int, e_int, Nx, Ny, Nz)
                                                        )
                                                        - numericalDiffusionBack(i, j, e, D, dx, dy, dz, Nx, Ny, Nz); // Periodic boundary conditions 

                /* THEN, ACCUMULATE THE RESULTS ONTO ONE FACE, MAP AROUND TO THE OTHER, AND CONTINUE */
                rho_np1[IDX3D(i, j, 0, Nx, Ny, Nz)] += rho_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)];
                rhovx_np1[IDX3D(i, j, 0, Nx, Ny, Nz)] += rhovx_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)];
                rhovy_np1[IDX3D(i, j, 0, Nx, Ny, Nz)] += rhovy_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)];
                rhovz_np1[IDX3D(i, j, 0, Nx, Ny, Nz)] += rhovz_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)];
                Bx_np1[IDX3D(i, j, 0, Nx, Ny, Nz)] += Bx_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)];
                By_np1[IDX3D(i, j, 0, Nx, Ny, Nz)] += By_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)];
                Bz_np1[IDX3D(i, j, 0, Nx, Ny, Nz)] += Bz_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)];
                e_np1[IDX3D(i, j, 0, Nx, Ny, Nz)] += e_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)];

                rho_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = rho_np1[IDX3D(i, j, 0, Nx, Ny, Nz)];
                rhovx_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = rhovx_np1[IDX3D(i, j, 0, Nx, Ny, Nz)];
                rhovy_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = rhovy_np1[IDX3D(i, j, 0, Nx, Ny, Nz)];
                rhovz_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = rhovz_np1[IDX3D(i, j, 0, Nx, Ny, Nz)];
                Bx_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = Bx_np1[IDX3D(i, j, 0, Nx, Ny, Nz)];
                By_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = By_np1[IDX3D(i, j, 0, Nx, Ny, Nz)];
                Bz_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = Bz_np1[IDX3D(i, j, 0, Nx, Ny, Nz)];
                e_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] = e_np1[IDX3D(i, j, 0, Nx, Ny, Nz)];
            }
        }
        
        /* 
        B.Cs on (i = 0, j, k) 
        BOTTOM
        (II)
        */
        for (int k = tidz; k < Nz; k += zthreads){ 
            for (int j = tidy; j < Ny; j += ythreads){
                /* IMPLEMENT */
                rho_np1[IDX3D(0, j, k, Nx, Ny, Nz)] = 0.0001; /* Magic vacuum number, hopefully don't need to change */
                rhovx_np1[IDX3D(0, j, k, Nx, Ny, Nz)] = 0.0; // Rigid wall
                rhovy_np1[IDX3D(0, j, k, Nx, Ny, Nz)] = 0.0;
                rhovz_np1[IDX3D(0, j, k, Nx, Ny, Nz)] = 0.0;
                Bx_np1[IDX3D(0, j, k, Nx, Ny, Nz)] = 0.0; // Perfectly-conducting wall
                By_np1[IDX3D(0, j, k, Nx, Ny, Nz)] = 0.0;
                Bz_np1[IDX3D(0, j, k, Nx, Ny, Nz)] = 0.0;
                e_np1[IDX3D(0, j, k, Nx, Ny, Nz)] = p(0, j, k, e, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
            }
        }
        
        /* 
        B.Cs on (i = Nx-1, j, k) 
        TOP
        (IV)
        */
        for (int k = tidz; k < Nz; k += zthreads){
            for (int j = tidy; j < Ny; j += ythreads){
                /* IMPLEMENT */
                rho_np1[IDX3D(Nx - 1, j, k, Nx, Ny, Nz)] = 0.0001; /* Magic vacuum number, hopefully don't need to change */
                rhovx_np1[IDX3D(Nx - 1, j, k, Nx, Ny, Nz)] = 0.0; // Rigid wall
                rhovy_np1[IDX3D(Nx - 1, j, k, Nx, Ny, Nz)] = 0.0;
                rhovz_np1[IDX3D(Nx - 1, j, k, Nx, Ny, Nz)] = 0.0;
                Bx_np1[IDX3D(Nx - 1, j, k, Nx, Ny, Nz)] = 0.0; // Perfectly-conducting wall
                By_np1[IDX3D(Nx - 1, j, k, Nx, Ny, Nz)] = 0.0;
                Bz_np1[IDX3D(Nx - 1, j, k, Nx, Ny, Nz)] = 0.0;
                e_np1[IDX3D(Nx- 1, j, k, Nx, Ny, Nz)] = p(Nx - 1, j, k, e, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
            }
        }
        
        /* 
        B.Cs on (i, j = 0, k) 
        LEFT
        (V)
        */
        for (int k = tidz; k < Nz; k += zthreads){
            for (int i = tidx + 1; i < Nx - 1; i += xthreads){
                /* IMPLEMENT */
                rho_np1[IDX3D(i, 0, k, Nx, Ny, Nz)] = 0.0001; /* Magic vacuum number, hopefully don't need to change */
                rhovx_np1[IDX3D(i, 0, k, Nx, Ny, Nz)] = 0.0; // Rigid wall
                rhovy_np1[IDX3D(i, 0, k, Nx, Ny, Nz)] = 0.0;
                rhovz_np1[IDX3D(i, 0, k, Nx, Ny, Nz)] = 0.0;
                Bx_np1[IDX3D(i, 0, k, Nx, Ny, Nz)] = 0.0; // Perfectly-conducting wall
                By_np1[IDX3D(i, 0, k, Nx, Ny, Nz)] = 0.0;
                Bz_np1[IDX3D(i, 0, k, Nx, Ny, Nz)] = 0.0;
                e_np1[IDX3D(i, 0, k, Nx, Ny, Nz)] = p(i, 0, k, e, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
            }
        }
        /* 
        B.Cs on (i, j = N-1, k) 
        RIGHT
        (III)
        */
        for (int k = tidz; k < Nz; k += zthreads){
            for (int i = tidy + 1; i < Nx - 1; i += xthreads){
                /* IMPLEMENT */
                rho_np1[IDX3D(i, Ny - 1, k, Nx, Ny, Nz)] = 0.0001; /* Magic vacuum number, hopefully don't need to change */
                rhovx_np1[IDX3D(i, Ny - 1, k, Nx, Ny, Nz)] = 0.0; // Rigid wall
                rhovy_np1[IDX3D(i, Ny - 1, k, Nx, Ny, Nz)] = 0.0;
                rhovz_np1[IDX3D(i, Ny - 1, k, Nx, Ny, Nz)] = 0.0;
                Bx_np1[IDX3D(i, Ny - 1, k, Nx, Ny, Nz)] = 0.0; // Perfectly-conducting wall
                By_np1[IDX3D(i, Ny - 1, k, Nx, Ny, Nz)] = 0.0;
                Bz_np1[IDX3D(i, Ny - 1, k, Nx, Ny, Nz)] = 0.0;
                e_np1[IDX3D(i, Ny - 1, k, Nx, Ny, Nz)] = p(i, Ny - 1, k, e, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
            }
        }
        return;
    }

// Swap buffer to avoid race conditions
__global__ void SwapSimData(float* rho, float* rhov_x, float* rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e,
    const float* rho_np1, const float* rhovx_np1, const float* rhovy_np1, const float* rhovz_np1, 
    const float* Bx_np1, const float* By_np1, const float* Bz_np1, const float* e_np1,
    const int Nx, const int Ny, const int Nz)
    {
        int tidx = threadIdx.x + blockDim.x * blockIdx.x;
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;

        int xthreads = blockDim.x * gridDim.x;
        int ythreads = blockDim.y * gridDim.y;
        int zthreads = blockDim.z * gridDim.z;

        for (int k = tidz; k < Nz; k += zthreads){
            for (int i = tidx; i < Nx; i += xthreads){
                for (int j = tidy; j < Ny; j += ythreads){
                    rho[IDX3D(i, j, k, Nx, Ny, Nz)] = rho_np1[IDX3D(i, j, k, Nx, Ny, Nz)];
                    rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] = rhovx_np1[IDX3D(i, j, k, Nx, Ny, Nz)];
                    rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)] = rhovy_np1[IDX3D(i, j, k, Nx, Ny, Nz)];
                    rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)] = rhovz_np1[IDX3D(i, j, k, Nx, Ny, Nz)];
                    Bx[IDX3D(i, j, k, Nx, Ny, Nz)] = Bx_np1[IDX3D(i, j, k, Nx, Ny, Nz)];
                    By[IDX3D(i, j, k, Nx, Ny, Nz)] = By_np1[IDX3D(i, j, k, Nx, Ny, Nz)];
                    Bz[IDX3D(i, j, k, Nx, Ny, Nz)] = Bz_np1[IDX3D(i, j, k, Nx, Ny, Nz)];
                    e[IDX3D(i, j, k, Nx, Ny, Nz)] = e_np1[IDX3D(i, j, k, Nx, Ny, Nz)];
                }
            }
        }
        return;
    } 

/* LAX WENDROFF ADVANCES */
// 15 AO, 8 MR
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

// 87 AO, 45 MR
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

// 87 AO, 45 MR
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

// 87 AO, 45 MR
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

// 35 AO, 22 MR
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

// 35 AO, 22 MR
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

// 35 AO, 22 MR
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

// 201 AO, 116 MR
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
// 1 MR
__device__ float XFluxRho(const int i, const int j, const int k, const float* rhov_x, const int Nx, const int Ny, const int Nz)
    {
        return rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)];
    }
// 1 MR
__device__ float YFluxRho(const int i, const int j, const int k, const float* rhov_y, const int Nx, const int Ny, const int Nz)
    {
        return rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)];
    }
// 1 MR
__device__ float ZFluxRho(const int i, const int j, const int k, const float* rhov_z, const int Nx, const int Ny, const int Nz)
    {
        return rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)];
    }

// RhoVX
// 3 MR
// 6 AO
// 5 FC = (2 + 5 + 8 + 5) AO + (3 + 4 + 1) MR 
// 26 AO, 11 MR
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

// 5 AO, 5 MR
__device__ float YFluxRhoVX(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, 
     const float* Bx, const float* By,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)]
            - Bx[IDX3D(i, j, k, Nx, Ny, Nz)] * By[IDX3D(i, j, k, Nx, Ny, Nz)];
     }

// 5 AO, 5 MR
__device__ float ZFluxRhoVX(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_z, 
     const float* Bx, const float* Bz,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)]
            - Bx[IDX3D(i, j, k, Nx, Ny, Nz)] * Bz[IDX3D(i, j, k, Nx, Ny, Nz)];
     }

// RhoVY
// 5 AO, 5 MR
__device__ float XFluxRhoVY(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, 
     const float* Bx, const float* By,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)]
            - Bx[IDX3D(i, j, k, Nx, Ny, Nz)] * By[IDX3D(i, j, k, Nx, Ny, Nz)];
     }

// 3 MR
// 6 AO
// 5 FC = (2 + 5 + 8 + 5) AO + (3 + 4 + 1) MR 
// 26 AO, 11 MR
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

// 5 AO, 5 MR
__device__ float ZFluxRhoVY(const int i, const int j, const int k,
     const float* rho, const float* rhov_y, const float* rhov_z, 
     const float* By, const float* Bz,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)]
            - By[IDX3D(i, j, k, Nx, Ny, Nz)] * Bz[IDX3D(i, j, k, Nx, Ny, Nz)];
     }

// RhoVZ
// 5 AO, 5 MR
__device__ float XFluxRhoVZ(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_z, 
     const float* Bx, const float* Bz,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)]
            - Bx[IDX3D(i, j, k, Nx, Ny, Nz)] * Bz[IDX3D(i, j, k, Nx, Ny, Nz)];
     }

// 5 AO, 5 MR
__device__ float YFluxRhoVZ(const int i, const int j, const int k,
     const float* rho, const float* rhov_y, const float* rhov_z,
     const float* By, const float* Bz,
     const int Nx, const int Ny, const int Nz)
    {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)]
            - By[IDX3D(i, j, k, Nx, Ny, Nz)] * Bz[IDX3D(i, j, k, Nx, Ny, Nz)];
    }

// 3 MR
// 6 AO
// 5 FC = (2 + 5 + 8 + 5) AO + (3 + 4 + 1) MR 
// 26 AO, 11 MR
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
// 0 AO, 0 MR
__device__ float XFluxBX()
    {
        return 0.0;
    }

// 5 AO, 5 MR
__device__ float YFluxBX(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, 
     const float* Bx, const float* By,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) 
            * (rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] * By[IDX3D(i, j, k, Nx, Ny, Nz)] 
            - Bx[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)]);
     }

// 5 AO, 5 MR
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
// 5 AO, 5 MR
__device__ float XFluxBY(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_y, 
     const float* Bx, const float* By,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) 
        * (rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)] * Bx[IDX3D(i, j, k, Nx, Ny, Nz)] 
        - By[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)]); 
     }

// 0 AO, 0 MR
__device__ float YFluxBY()
    {
        return 0.0;
    }

// 5 AO, 5 MR
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
// 5 AO, 5 MR
__device__ float XFluxBZ(const int i, const int j, const int k,
     const float* rho, const float* rhov_x, const float* rhov_z, 
     const float* Bx, const float* Bz,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) 
        * (rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)] * Bx[IDX3D(i, j, k, Nx, Ny, Nz)] 
        - Bz[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)]);
     }

// 5 AO, 5 MR
__device__ float YFluxBZ(const int i, const int j, const int k,
     const float* rho, const float* rhov_y, const float* rhov_z, 
     const float* By, const float* Bz,
     const int Nx, const int Ny, const int Nz)
     {
        return (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) 
        * (rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)] * By[IDX3D(i, j, k, Nx, Ny, Nz)] 
        - Bz[IDX3D(i, j, k, Nx, Ny, Nz)] * rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)]);
     }

// 0 AO, 0 MR
__device__ float ZFluxBZ()
    {
        return 0.0;
    }

// e
// 4 MR
// 6 AO
// 4 FC = (5 + 8 + 5 + 7) AO + (3 + 4 + 1 + 7) MR
// 31 AO, 19 MR
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

// 4 MR
// 6 AO
// 4 FC = (5 + 8 + 5 + 7) AO + (3 + 4 + 1 + 7) MR
// 31 AO, 19 MR
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

// 4 MR
// 6 AO
// 4 FC = (5 + 8 + 5 + 7) AO + (3 + 4 + 1 + 7) MR
// 31 AO, 19 MR
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
// 1 MR
// 12 AO
// 6 FC = 6 MR
// 12 AO, 7 MR
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

// 1 MR
// 12 AO
// 6 FC = (2 * 26 + 2 * 5 + 2 * 5) AO + (2 * 11 + 2 * 5 + 2 * 5) MR
// 84 AO, 43 MR
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

// 1 MR
// 12 AO
// 6 FC = (2 * 5 + 2 * 26 + 2 * 5) AO + (2 * 5 + 2 * 11 + 2 * 5) MR
// 84 AO, 43 MR
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

// 1 MR
// 12 AO
// 6 FC = (2 * 5 + 2 * 5  + 2 * 26) AO + (2 * 5 + 2 * 5 + 2 * 11) MR
// 84 AO, 43 MR
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

// 1 MR 
// 12 AO
// 6 FC = 20 AO + 20 MR
// 32 AO, 21 MR
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

// 1 MR 
// 12 AO
// 6 FC = 20 AO + 20 MR
// 32 AO, 21 MR
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

// 1 MR 
// 12 AO
// 6 FC = 20 AO + 20 MR
// 32 AO, 21 MR
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

// 1 MR 
// 12 AO
// 6 FC = 6 * 31 AO + 6 * 19 MR
// 186 AO, 114 MR
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
// 3 MR
// 5 AO
__device__ float B_sq(const int i, const int j, const int k, const float* Bx, const float* By, const float* Bz, 
    const int Nx, const int Ny, const int Nz)
    {
        return pow(Bx[IDX3D(i, j, k, Nx, Ny, Nz)], 2) + pow(By[IDX3D(i, j, k, Nx, Ny, Nz)], 2) + pow(Bz[IDX3D(i, j, k, Nx, Ny, Nz)], 2);
    }

// 1 MR
// 5 AO
__device__ float p(const int i, const int j, const int k, 
    const float* e, const float B_sq, const float KE, 
    const int Nx, const int Ny, const int Nz)
    {
        return (gamma - 1.0) * (e[IDX3D(i, j, k, Nx, Ny, Nz)] - KE - B_sq / 2.0);
    }

// 4 MR
// 8 AO
__device__ float KE(const int i, const int j, const int k, const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
    const int Nx, const int Ny, const int Nz)
    {
        float KE = (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * (
            pow(rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)], 2) + pow(rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)], 2) + pow(rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)], 2));
        return KE;
    }

// 7 MR
// 7 AO
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
// 19 AO, 9 MR
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

// Implements numerical diffusion on the front plane of the simulation grid (k = 0)
// Periodic boundary conditions are the reason
// 2nd-order central difference
// 19 AO, 9 MR
__device__ float numericalDiffusionFront(const int i, const int j, const float* fluid_var, 
    const float D, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
    {
        float num_diff = 0.0;
        num_diff = D * (
            (1.0 / pow(dx, 2)) * (fluid_var[IDX3D(i+1, j, 0, Nx, Ny, Nz)] - 2.0*fluid_var[IDX3D(i, j, 0, Nx, Ny, Nz)] + fluid_var[IDX3D(i-1, j, 0, Nx, Ny, Nz)])
            + (1.0 / pow(dy, 2)) * (fluid_var[IDX3D(i, j+1, 0, Nx, Ny, Nz)] - 2.0*fluid_var[IDX3D(i, j, 0, Nx, Ny, Nz)] + fluid_var[IDX3D(i, j-1, 0, Nx, Ny, Nz)])
            + (1.0 / pow(dz, 2)) * (fluid_var[IDX3D(i, j, 1, Nx, Ny, Nz)] - 2.0*fluid_var[IDX3D(i, j, 0, Nx, Ny, Nz)] + fluid_var[IDX3D(i, j, Nz - 2, Nx, Ny, Nz)])
            );
        return num_diff;
    }

// Implements numerical diffusion on the back plane of the simulation grid (k = Nz - 1)
// Periodic boundary conditions are the reason
// 2nd-order central difference
// 19 AO, 9 MR
__device__ float numericalDiffusionBack(const int i, const int j, const float* fluid_var, 
    const float D, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
    {
        float num_diff = 0.0;
        num_diff = D * (
            (1.0 / pow(dx, 2)) * (fluid_var[IDX3D(i+1, j, Nz - 1, Nx, Ny, Nz)] - 2.0*fluid_var[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] + fluid_var[IDX3D(i-1, j, Nz - 1, Nx, Ny, Nz)])
            + (1.0 / pow(dy, 2)) * (fluid_var[IDX3D(i, j+1, Nz - 1, Nx, Ny, Nz)] - 2.0*fluid_var[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] + fluid_var[IDX3D(i, j-1, Nz - 1, Nx, Ny, Nz)])
            + (1.0 / pow(dz, 2)) * (fluid_var[IDX3D(i, j, 1, Nx, Ny, Nz)] - 2.0*fluid_var[IDX3D(i, j, Nz - 1, Nx, Ny, Nz)] + fluid_var[IDX3D(i, j, Nz - 2, Nx, Ny, Nz)])
            );
        return num_diff;
    }

#endif