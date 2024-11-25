#include <stdio.h>

#include "diffusion.cuh"
#include "kernels_od.cuh"
#include "kernels_od_fluxes.cuh"
#include "kernels_od_intvar.cuh"
#include "helper_functions.cuh"

#define IDX3D(i, j, k, Nx, Ny, Nz) ((k) * (Nx * Ny) + (i) * Ny + j) // parentheses are necessary to avoid calculating `i - 1 * Ny` or `k - 1 * (Nx * Ny)`


// 80 registers per thread
__global__ void ComputeIntermediateVariables(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz, const float D,
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
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz)] = intRho(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz)
                                                            + dt * numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 0, Nx, Ny, Nz); // rho
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = intRhoVX(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz)
                                                            + dt * numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 1, Nx, Ny, Nz); // rhov_x
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz)
                                                            + dt * numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 2, Nx, Ny, Nz); // rhov_y
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz)
                                                            + dt * numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 3, Nx, Ny, Nz); // rhov_z
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = intBx(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz)
                                                            + dt * numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 4, Nx, Ny, Nz); // Bx
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = intBy(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz)
                                                            + dt * numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 5, Nx, Ny, Nz); // By
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = intBz(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz)
                                                            + dt * numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 6, Nx, Ny, Nz); // Bz
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = intE(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz)
                                                            + dt * numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 7, Nx, Ny, Nz); // e
                
                }
            }
        }


        return;
    }


// INT. VAR COMPUTATION AT THE APPROPRIATE BOUNDARY PLANES: 
// (1) k = 0
// (2) k = Nz - 1
// (3) i = 0
// (4) i = Nx - 1
// (5) j = 0
// (6) j = Ny - 1
// 56 registers per thread
__global__ void ComputeIntermediateVariablesBoundary(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz, const float D,
    const int Nx, const int Ny, const int Nz)
    {
        int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;

        int xthreads = blockDim.x * gridDim.x;
        int ythreads = blockDim.y * gridDim.y;
        int zthreads = blockDim.z * gridDim.z;
    
        int cube_size = Nx * Ny * Nz;

        // k = 0 and k = Nz - 1
        int k = 0;
        for (int i = tidx + 1; i < Nx; i += xthreads){
            for (int j = tidy + 1; j < Ny; j += ythreads){
                    // k = 0
                    k = 0;
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz)] =  fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]
                                                        - (dt / dx) * (XFluxRho(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxRho(i-1, j, k, fluidvar, Nx, Ny, Nz))
                                                        - (dt / dy) * (YFluxRho(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxRho(i, j-1, k, fluidvar, Nx, Ny, Nz))
                                                        - (dt / dz) * (ZFluxRho(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRho(i, j, Nz-2, fluidvar, Nx, Ny, Nz))
                                                        + dt * numericalDiffusionFront(i, j, fluidvar, D, dx, dy, dz, 0, Nx, Ny, Nz); // rho
                    
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] =  fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size]
                                                                    - (dt / dx) * (XFluxRhoVX(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVX(i-1, j, k, fluidvar, Nx, Ny, Nz))
                                                                    - (dt / dy) * (YFluxRhoVX(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVX(i, j-1, k, fluidvar, Nx, Ny, Nz))
                                                                    - (dt / dz) * (ZFluxRhoVX(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(i, j, Nz-2, fluidvar, Nx, Ny, Nz))
                                                                    + dt * numericalDiffusionFront(i, j, fluidvar, D, dx, dy, dz, 1, Nx, Ny, Nz);// rhov_x

                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] =  fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size]
                                                                        - (dt / dx) * (XFluxRhoVY(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVY(i-1, j, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dy) * (YFluxRhoVY(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVY(i, j-1, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dz) * (ZFluxRhoVY(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(i, j, Nz-2, fluidvar, Nx, Ny, Nz))
                                                                        + dt * numericalDiffusionFront(i, j, fluidvar, D, dx, dy, dz, 2, Nx, Ny, Nz); // rhov_y
                    
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] =  fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size]
                                                                        - (dt / dx) * (XFluxRhoVZ(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVZ(i-1, j, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dy) * (YFluxRhoVZ(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVZ(i, j-1, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dz) * (ZFluxRhoVZ(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, j, Nz-2, fluidvar, Nx, Ny, Nz))
                                                                        + dt * numericalDiffusionFront(i, j, fluidvar, D, dx, dy, dz, 3, Nx, Ny, Nz); // rhov_z
                    
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] =  fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size]
                                                                        - (dt / dx) * (XFluxBX(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxBX(i-1, j, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dy) * (YFluxBX(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxBX(i, j-1, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dz) * (ZFluxBX(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxBX(i, j, Nz-2, fluidvar, Nx, Ny, Nz))
                                                                        + dt * numericalDiffusionFront(i, j, fluidvar, D, dx, dy, dz, 4, Nx, Ny, Nz); // Bx
                    
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] =  fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size]
                                                                        - (dt / dx) * (XFluxBY(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxBY(i-1, j, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dy) * (YFluxBY(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxBY(i, j-1, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dz) * (ZFluxBY(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxBY(i, j, Nz-2, fluidvar, Nx, Ny, Nz))
                                                                        + dt * numericalDiffusionFront(i, j, fluidvar, D, dx, dy, dz, 5, Nx, Ny, Nz); // By
                    
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] =  fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size]
                                                                        - (dt / dx) * (XFluxBZ(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxBZ(i-1, j, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dy) * (YFluxBZ(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxBZ(i, j-1, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dz) * (ZFluxBZ(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxBZ(i, j, Nz-2, fluidvar, Nx, Ny, Nz))
                                                                        + dt * numericalDiffusionFront(i, j, fluidvar, D, dx, dy, dz, 6, Nx, Ny, Nz); // Bz
                    
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] =  fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size]
                                                                        - (dt / dx) * (XFluxE(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxE(i-1, j, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dy) * (YFluxE(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxE(i, j-1, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dz) * (ZFluxE(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxE(i, j, Nz-2, fluidvar, Nx, Ny, Nz))
                                                                        + dt * numericalDiffusionFront(i, j, fluidvar, D, dx, dy, dz, 7, Nx, Ny, Nz); // e
                    // k = Nz - 1
                    k = Nz - 1;
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz)] =  fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]
                                                        - (dt / dx) * (XFluxRho(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxRho(i-1, j, k, fluidvar, Nx, Ny, Nz))
                                                        - (dt / dy) * (YFluxRho(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxRho(i, j-1, k, fluidvar, Nx, Ny, Nz))
                                                        - (dt / dz) * (ZFluxRho(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRho(i, j, Nz-2, fluidvar, Nx, Ny, Nz))
                                                        + dt * numericalDiffusionBack(i, j, fluidvar, D, dx, dy, dz, 0, Nx, Ny, Nz); // rho
                    
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] =  fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size]
                                                                    - (dt / dx) * (XFluxRhoVX(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVX(i-1, j, k, fluidvar, Nx, Ny, Nz))
                                                                    - (dt / dy) * (YFluxRhoVX(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVX(i, j-1, k, fluidvar, Nx, Ny, Nz))
                                                                    - (dt / dz) * (ZFluxRhoVX(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(i, j, Nz-2, fluidvar, Nx, Ny, Nz))
                                                                    + dt * numericalDiffusionBack(i, j, fluidvar, D, dx, dy, dz, 1, Nx, Ny, Nz); // rhov_x

                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] =  fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size]
                                                                        - (dt / dx) * (XFluxRhoVY(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVY(i-1, j, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dy) * (YFluxRhoVY(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVY(i, j-1, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dz) * (ZFluxRhoVY(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(i, j, Nz-2, fluidvar, Nx, Ny, Nz))
                                                                        + dt * numericalDiffusionBack(i, j, fluidvar, D, dx, dy, dz, 2, Nx, Ny, Nz); // rhov_y
                    
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] =  fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size]
                                                                        - (dt / dx) * (XFluxRhoVZ(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVZ(i-1, j, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dy) * (YFluxRhoVZ(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVZ(i, j-1, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dz) * (ZFluxRhoVZ(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, j, Nz-2, fluidvar, Nx, Ny, Nz))
                                                                        + dt * numericalDiffusionBack(i, j, fluidvar, D, dx, dy, dz, 3, Nx, Ny, Nz); // rhov_z
                    
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] =  fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size]
                                                                        - (dt / dx) * (XFluxBX(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxBX(i-1, j, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dy) * (YFluxBX(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxBX(i, j-1, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dz) * (ZFluxBX(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxBX(i, j, Nz-2, fluidvar, Nx, Ny, Nz))
                                                                        + dt * numericalDiffusionBack(i, j, fluidvar, D, dx, dy, dz, 4, Nx, Ny, Nz); // Bx
                    
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] =  fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size]
                                                                        - (dt / dx) * (XFluxBY(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxBY(i-1, j, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dy) * (YFluxBY(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxBY(i, j-1, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dz) * (ZFluxBY(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxBY(i, j, Nz-2, fluidvar, Nx, Ny, Nz))
                                                                        + dt * numericalDiffusionBack(i, j, fluidvar, D, dx, dy, dz, 5, Nx, Ny, Nz); // By
                    
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] =  fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size]
                                                                        - (dt / dx) * (XFluxBZ(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxBZ(i-1, j, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dy) * (YFluxBZ(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxBZ(i, j-1, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dz) * (ZFluxBZ(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxBZ(i, j, Nz-2, fluidvar, Nx, Ny, Nz))
                                                                        + dt * numericalDiffusionBack(i, j, fluidvar, D, dx, dy, dz, 6, Nx, Ny, Nz); // Bz
                    
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] =  fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size]
                                                                        - (dt / dx) * (XFluxE(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxE(i-1, j, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dy) * (YFluxE(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxE(i, j-1, k, fluidvar, Nx, Ny, Nz))
                                                                        - (dt / dz) * (ZFluxE(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxE(i, j, Nz-2, fluidvar, Nx, Ny, Nz))
                                                                        + dt * numericalDiffusionBack(i, j, fluidvar, D, dx, dy, dz, 7, Nx, Ny, Nz); // e
            }
        }

        // i = 0
        int i = 0;
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){
            for (int j = tidy + 1; j < Ny; j += ythreads){
                intvar[IDX3D(i, j, k, Nx, Ny, Nz)] = intRho(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = intRhoVX(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = intBx(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = intBy(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = intBz(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = intE(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
            }
        }

        // i = Nx - 1
        i = Nx - 1;
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){
            for (int j = tidy + 1; j < Ny; j += ythreads){
                intvar[IDX3D(i, j, k, Nx, Ny, Nz)] = intRho(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = intRhoVX(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = intBx(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = intBy(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = intBz(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = intE(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
            }
        }

        // j = 0
        int j = 0;
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){
            for (int i = tidx + 1; i < Nx - 1; i += xthreads){
                intvar[IDX3D(i, j, k, Nx, Ny, Nz)] = intRho(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = intRhoVX(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = intBx(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = intBy(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = intBz(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = intE(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
            }
        }

        // j = Ny - 1
        j = Ny - 1;
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){
            for (int i = tidx + 1; i < Nx - 1; i += xthreads){
                intvar[IDX3D(i, j, k, Nx, Ny, Nz)] = intRho(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = intRhoVX(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = intBx(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = intBy(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = intBz(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = intE(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
            }
        }
        return;
    }

__device__ float intRho(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]
            - (dt / dx) * (XFluxRho(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxRho(i-1, j, k, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxRho(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxRho(i, j-1, k, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRho(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRho(i, j, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVX(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {   
        int cube_size = Nx * Ny * Nz;
        return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size]
        - (dt / dx) * (XFluxRhoVX(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVX(i-1, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dy) * (YFluxRhoVX(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVX(i, j-1, k, fluidvar, Nx, Ny, Nz))
        - (dt / dz) * (ZFluxRhoVX(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(i, j, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVY(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size]
        - (dt / dx) * (XFluxRhoVY(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVY(i-1, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dy) * (YFluxRhoVY(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVY(i, j-1, k, fluidvar, Nx, Ny, Nz))
        - (dt / dz) * (ZFluxRhoVY(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(i, j, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVZ(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size]
        - (dt / dx) * (XFluxRhoVZ(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVZ(i-1, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dy) * (YFluxRhoVZ(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVZ(i, j-1, k, fluidvar, Nx, Ny, Nz))
        - (dt / dz) * (ZFluxRhoVZ(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, j, k-1, fluidvar, Nx, Ny, Nz)); 
    }

__device__ float intBx(const int i, const int j, const int k,
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size]
        - (dt / dx) * (XFluxBX(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxBX(i-1, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dy) * (YFluxBX(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxBX(i, j-1, k, fluidvar, Nx, Ny, Nz))
        - (dt / dz) * (ZFluxBX(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxBX(i, j, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBy(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size]
        - (dt / dx) * (XFluxBY(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxBY(i-1, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dy) * (YFluxBY(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxBY(i, j-1, k, fluidvar, Nx, Ny, Nz))
        - (dt / dz) * (ZFluxBY(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxBY(i, j, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBz(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size]
        - (dt / dx) * (XFluxBZ(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxBZ(i-1, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dy) * (YFluxBZ(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxBZ(i, j-1, k, fluidvar, Nx, Ny, Nz))
        - (dt / dz) * (ZFluxBZ(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxBZ(i, j, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intE(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size]
        - (dt / dx) * (XFluxE(i, j, k, fluidvar, Nx, Ny, Nz) - XFluxE(i-1, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dy) * (YFluxE(i, j, k, fluidvar, Nx, Ny, Nz) - YFluxE(i, j-1, k, fluidvar, Nx, Ny, Nz))
        - (dt / dz) * (ZFluxE(i, j, k, fluidvar, Nx, Ny, Nz) - ZFluxE(i, j, k-1, fluidvar, Nx, Ny, Nz));
    }