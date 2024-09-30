#include <math.h>
#include <stdio.h>

#include "kernels_od.cuh"
#include "kernels_od_fluxes.cuh"
#include "helper_functions.cuh"
#include "diffusion.cuh"

// row-major, column-minor order
#define IDX3D(i, j, k, Nx, Ny, Nz) ((k) * (Nx * Ny) + (i) * Ny + j) // parentheses are necessary to avoid calculating `i - 1 * Ny` or `k - 1 * (Nx * Ny)`

/* 
Needed to be refactored to compactify all data into three arrays:
(1) float *fluidvar
(2) float *intvar
(3) float *fluidvar_np1

Furthermore, there was a race condition amongst the intermediate variable calculation. 

To solve this, the intermediate variable's computation is moved outside of the fluid advance loop, and into its own kernel. 
They are precomputed after the initial conditions, and then compute in the loop while writing data out in order to add minimal synchronization barriers.   

Here is what the storage pattern looks like:
fluidvar -> [rho_{000}, rho_{010}, rho_{020}, ..., rho_{0,Ny-1,0}, rho_{100}, ..., rho_{Nx-1,Ny-1,Nz-1}, rhov_x_{000}, rhov_x_{010}, ... , e_{Nx-1,Ny-1,Nz-1}]
*/

// Global kernels
// 40 registers per thread
__global__ void SwapSimData(float* fluidvar, const float* fluidvar_np1, const int Nx, const int Ny, const int Nz)
    {
    int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    int tidz = threadIdx.z + blockDim.z * blockIdx.z;

    int xthreads = blockDim.x * gridDim.x;
    int ythreads = blockDim.y * gridDim.y;
    int zthreads = blockDim.z * gridDim.z;

    int cube_size = Nx * Ny * Nz;
    for (int k = tidz; k < Nz; k += zthreads){
        for (int i = tidx; i < Nx; i += xthreads){
            for (int j = tidy; j < Ny; j += ythreads){
                fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz)]; // rho
                fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size]; // rhov_x
                fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size]; // rhov_y
                fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size]; // rhov_z
                fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size]; // Bx
                fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size]; // By 
                fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size]; // Bz
                fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size]; // e
            }
        }
    }
    return;
    }

// 66 registers per thread
__global__ void FluidAdvance(float* fluidvar_np1, const float* fluidvar, const float* intvar, 
    const float D, const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
    {
    int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    int tidz = threadIdx.z + blockDim.z * blockIdx.z;

    int xthreads = blockDim.x * gridDim.x;
    int ythreads = blockDim.y * gridDim.y;
    int zthreads = blockDim.z * gridDim.z;

    int cube_size = Nx * Ny * Nz;

    for (int k = tidz + 1; k < Nz - 1; k += zthreads){
        for (int i = tidx + 1; i < Nx - 1; i += xthreads){
            for (int j = tidy + 1; j < Ny - 1; j += ythreads){
                fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvRho(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                            + numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 0, Nx, Ny, Nz); // rho
                fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] =  LaxWendroffAdvRhoVX(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                            + numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 1, Nx, Ny, Nz); // rhov_x
                fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] =  LaxWendroffAdvRhoVY(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                            + numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 2, Nx, Ny, Nz); // rhov_y
                fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] =  LaxWendroffAdvRhoVZ(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                            + numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 3, Nx, Ny, Nz); // rhov_z
                fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] =  LaxWendroffAdvBX(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                            + numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 4, Nx, Ny, Nz); // Bx
                fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] =  LaxWendroffAdvBY(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                            + numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 5, Nx, Ny, Nz); // By 
                fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] =  LaxWendroffAdvBZ(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                            + numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 6, Nx, Ny, Nz); // Bz
                fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] =  LaxWendroffAdvE(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                            + numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 7, Nx, Ny, Nz); // e
                
                // if (isnan(fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size])){
                //     float LW_val = LaxWendroffAdvRhoVZ(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz);
                //     float dfn_val = numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 3, Nx, Ny, Nz);
                //     // printf("For (%d, %d, %d) the value of LWAdvance is %5.4f, and numericalDiffusion is %5.4f\n", i, j, k, LW_val, dfn_val);
                //     if (isnan(LW_val) && !isnan(dfn_val)){
                //         printf("LaxWendroffAdvRhoVZ is a problem, for (%d, %d, %d) the value of rhovz is %5.4f, intvar: %5.4f, XFRVZ: %5.4f, YFRVZ: %5.4f, ZFRVZ: %5.4f, XFRVZip1: %5.4f, YFRVZjp1: %5.4f, ZFRVZkp1: %5.4f\n", 
                //             i, j, k, fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size], intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size], 
                //             XFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz), YFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz), ZFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz), 
                //             XFluxRhoVZ(i + 1, j, k, intvar, Nx, Ny, Nz), YFluxRhoVZ(i, j + 1, k, intvar, Nx, Ny, Nz), ZFluxRhoVZ(i, j, k + 1, intvar, Nx, Ny, Nz));
                //     }
                // } 
            }
        }
    }
    return;
    }

// 120 registers per thread
__global__ void BoundaryConditions(volatile float* fluidvar_np1, const float* fluidvar, const float* intvar, 
    const float D, const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
    int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    int tidz = threadIdx.z + blockDim.z * blockIdx.z;

    int xthreads = blockDim.x * gridDim.x;
    int ythreads = blockDim.y * gridDim.y;
    int zthreads = blockDim.z * gridDim.z;

    int cube_size = Nx * Ny * Nz;

    /* IMPLEMENT PBCs */
    // k = 0 and k = Nz - 1
    int k = 0;
    for (int i = tidx + 1; i < Nx - 1; i += xthreads){
        for (int j = tidy + 1; j < Ny - 1; j += ythreads){
            k = 0;
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] + intvar[IDX3D(i, j, k, Nx, Ny, Nz)])
                                                        - 0.5 * (dt / dx) * (XFluxRho(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRho(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxRho(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRho(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxRho(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxRho(i, j, k, intvar, Nx, Ny, Nz));
                                                        + numericalDiffusionFront(i, j, fluidvar, D, dx, dy, dz, 0, Nx, Ny, Nz);

            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxRhoVX(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxRhoVX(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxRhoVX(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz));
                                                        + numericalDiffusionFront(i, j, fluidvar, D, dx, dy, dz, 1, Nx, Ny, Nz);
            
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxRhoVY(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxRhoVY(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxRhoVY(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz));
                                                        + numericalDiffusionFront(i, j, fluidvar, D, dx, dy, dz, 2, Nx, Ny, Nz);

            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxRhoVZ(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxRhoVZ(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxRhoVZ(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz));
                                                        + numericalDiffusionFront(i, j, fluidvar, D, dx, dy, dz, 3, Nx, Ny, Nz);

            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxBX(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBX(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxBX(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBX(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxBX(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxBX(i, j, k, intvar, Nx, Ny, Nz));
                                                        + numericalDiffusionFront(i, j, fluidvar, D, dx, dy, dz, 4, Nx, Ny, Nz);

            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxBY(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBY(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxBY(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBY(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxBY(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxBY(i, j, k, intvar, Nx, Ny, Nz));
                                                        + numericalDiffusionFront(i, j, fluidvar, D, dx, dy, dz, 5, Nx, Ny, Nz);

            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxBZ(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxBZ(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxBZ(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxBZ(i, j, k, intvar, Nx, Ny, Nz));
                                                        + numericalDiffusionFront(i, j, fluidvar, D, dx, dy, dz, 6, Nx, Ny, Nz);

            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxE(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxE(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxE(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxBZ(i, j, k, intvar, Nx, Ny, Nz));
                                                        + numericalDiffusionFront(i, j, fluidvar, D, dx, dy, dz, 7, Nx, Ny, Nz);

            k = Nz - 1;
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] + intvar[IDX3D(i, j, k, Nx, Ny, Nz)])
                                                        - 0.5 * (dt / dx) * (XFluxRho(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRho(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxRho(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRho(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxRho(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxRho(i, j, k, intvar, Nx, Ny, Nz));
                                                        + numericalDiffusionBack(i, j, fluidvar, D, dx, dy, dz, 0, Nx, Ny, Nz);

            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxRhoVX(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxRhoVX(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxRhoVX(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz));
                                                        + numericalDiffusionBack(i, j, fluidvar, D, dx, dy, dz, 1, Nx, Ny, Nz);
            
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxRhoVY(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxRhoVY(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxRhoVY(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz));
                                                        + numericalDiffusionBack(i, j, fluidvar, D, dx, dy, dz, 2, Nx, Ny, Nz);

            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxRhoVZ(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxRhoVZ(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxRhoVZ(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz));
                                                        + numericalDiffusionBack(i, j, fluidvar, D, dx, dy, dz, 3, Nx, Ny, Nz);

            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxBX(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBX(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxBX(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBX(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxBX(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxBX(i, j, k, intvar, Nx, Ny, Nz));
                                                        + numericalDiffusionBack(i, j, fluidvar, D, dx, dy, dz, 4, Nx, Ny, Nz);

            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxBY(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBY(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxBY(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBY(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxBY(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxBY(i, j, k, intvar, Nx, Ny, Nz));
                                                        + numericalDiffusionBack(i, j, fluidvar, D, dx, dy, dz, 5, Nx, Ny, Nz);

            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxBZ(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxBZ(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxBZ(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxBZ(i, j, k, intvar, Nx, Ny, Nz));
                                                        + numericalDiffusionBack(i, j, fluidvar, D, dx, dy, dz, 6, Nx, Ny, Nz);

            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxE(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxE(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxE(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxBZ(i, j, k, intvar, Nx, Ny, Nz));
                                                        + numericalDiffusionBack(i, j, fluidvar, D, dx, dy, dz, 7, Nx, Ny, Nz);
            
            // THEN, ACCUMULATE THE RESULTS ONTO ONE FACE, MAP AROUND TO THE OTHER, AND CONTINUE
            for (int ivf = 0; ivf < 8; ivf++){
                fluidvar_np1[IDX3D(i, j, 0, Nx, Ny, Nz) + ivf * cube_size] += fluidvar_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz) + ivf * cube_size];
                fluidvar_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz) + ivf * cube_size] = fluidvar_np1[IDX3D(i, j, 0, Nx, Ny, Nz) + ivf * cube_size];
            }
        }
    }

    /* 
    B.Cs on BOTTOM (II) 
    (i = 0, j, k) 
    and
    B.Cs on TOP (IV)
    (i = Nx-1, j, k) 
    */
    int i = 0; 
    for (int k = tidz; k < Nz; k += zthreads){ 
        for (int j = tidy; j < Ny; j += ythreads){
            i = 0;
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0; /* Magic vacuum number */
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);

            i = Nx - 1;
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0; /* Magic vacuum number */
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
        }
    }

    /* 
    B.Cs on LEFT (V)
    (i, j = 0, k) 
    and
    B.Cs on RIGHT (III)
    (i, j = N-1, k) 
    */
    int j = 0;
    for (int k = tidz; k < Nz; k += zthreads){
        for (int i = tidx + 1; i < Nx - 1; i += xthreads){
            j = 0;
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0; /* Magic vacuum number */
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
            
            j = Ny - 1;
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0; /* Magic vacuum number */
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
            fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
        }
    }

    return;
    }

// Device kernels
__device__ float LaxWendroffAdvRho(const int i, const int j, const int k, 
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] + intvar[IDX3D(i, j, k, Nx, Ny, Nz)])
        - 0.5 * (dt / dx) * (XFluxRho(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxRho(i, j, k, intvar, Nx, Ny, Nz))
        - 0.5 * (dt / dy) * (YFluxRho(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxRho(i, j, k, intvar, Nx, Ny, Nz))
        - 0.5 * (dt / dz) * (ZFluxRho(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxRho(i, j, k, intvar, Nx, Ny, Nz));
    }

__device__ float LaxWendroffAdvRhoVX(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size])
                - 0.5 * (dt / dx) * (XFluxRhoVX(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxRhoVX(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxRhoVX(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz));  
    }

__device__ float LaxWendroffAdvRhoVY(const int i, const int j, const int k, 
    const float* fluidvar, const float* intvar,  
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size])
                - 0.5 * (dt / dx) * (XFluxRhoVY(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxRhoVY(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxRhoVY(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz)); 
    }

__device__ float LaxWendroffAdvRhoVZ(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size])
                - 0.5 * (dt / dx) * (XFluxRhoVZ(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxRhoVZ(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxRhoVZ(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz)); 
    }

__device__ float LaxWendroffAdvBX(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size])
                - 0.5 * (dt / dx) * (XFluxBX(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxBX(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxBX(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxBX(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxBX(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxBX(i, j, k, intvar, Nx, Ny, Nz)); 
    }

__device__ float LaxWendroffAdvBY(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size])
                - 0.5 * (dt / dx) * (XFluxBY(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxBY(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxBY(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxBY(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxBY(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxBY(i, j, k, intvar, Nx, Ny, Nz)); 
    }

__device__ float LaxWendroffAdvBZ(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size])
                - 0.5 * (dt / dx) * (XFluxBZ(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxBZ(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxBZ(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxBZ(i, j, k, intvar, Nx, Ny, Nz)); 
    }

__device__ float LaxWendroffAdvE(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size])
                - 0.5 * (dt / dx) * (XFluxE(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxE(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxE(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxE(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxE(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxE(i, j, k, intvar, Nx, Ny, Nz)); 
    }