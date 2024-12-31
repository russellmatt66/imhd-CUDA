#include <stdio.h>

#include "diffusion.cuh"
#include "kernels_od.cuh"
#include "kernels_od_fluxes.cuh"
#include "kernels_od_intvar.cuh"
#include "helper_functions.cuh"

/* THIS SHOULD BE DEFINED A SINGLE TIME IN A SINGLE PLACE */
#define IDX3D(i, j, k, Nx, Ny, Nz) ((k) * (Nx * Ny) + (i) * Ny + j) // parentheses are necessary to avoid calculating `i - 1 * Ny` or `k - 1 * (Nx * Ny)`

__global__ void ComputeIntVarsLocal(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz, const float D,
    const int Nx, const int Ny, const int Nz)
    {
        int tidx = threadIdx.x + blockDim.x * blockIdx.x;
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;

        int xthreads = blockDim.x * gridDim.x;
        int ythreads = blockDim.y * gridDim.y;
        int zthreads = blockDim.z * gridDim.z;

        /* 
        COMPUTE THE INTERMEDIATE VARIABLES USING LOCAL (SHARED) MEMORY
        */

        return;
    }

/* 
THESE KERNELS ALL THRASH THE CACHE 
TO ADDRESS, DO THE FOLLOWING:
(1) Megakernel w/local (shared) memory
(2) Microkernels w/local (shared) memory
- "Microkernel" = "Update a single fluid variable"
*/
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
/* 
SKIPS OVER: 
    {(0,0,k), (i,0,Nz-1), (0,j,0), (i,0,Nz-1), (Nx-1,0,k), (i,0,0)}

TO ADDRESS, DO THE FOLLOWING:
(1) Determine the influence of these points
(2) Determine what they could appropriately be set to initially
*/
/* 
This is an MVP - IT CAN BE GREATLY IMPROVED 
o Eliminate cache-thrashing 
o Compactify loops
*/
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

        // Front Face
        // k = 0
        // i \in [1,Nx-1]
        // j \in [1,Ny-1]
        for (int i = tidx + 1; i < Nx; i += xthreads){
            for (int j = tidy + 1; j < Ny; j += ythreads){
                intvar[IDX3D(i, j, 0, Nx, Ny, Nz)] = intRhoFront(i, j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
                intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + cube_size] = intRhoVXFront(i, j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
                intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYFront(i, j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
                intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZFront(i, j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
                intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 4 * cube_size] = intBXFront(i, j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
                intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 5 * cube_size] = intBYFront(i, j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
                intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 6 * cube_size] = intBZFront(i, j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
                intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 7 * cube_size] = intEFront(i, j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            }
        }

        // Back Face
        // k = Nz-1
        // i \in [1,Nx-1]
        // j \in [1,Ny-1]
        for (int i = tidx + 1; i < Nx; i += xthreads){
            for (int j = tidy + 1; j < Ny; j += ythreads){
                intvar[IDX3D(i, j, 0, Nx, Ny, Nz)] = intRho(i, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
                intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + cube_size] = intRhoVX(i, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
                intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(i, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
                intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(i, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
                intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 4 * cube_size] = intBX(i, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
                intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 5 * cube_size] = intBY(i, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
                intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 6 * cube_size] = intBZ(i, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
                intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 7 * cube_size] = intE(i, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            }
        }

        // Top Face
        // i = 0
        // j \in [1,Ny-1]
        // k \in [1,Nz-2]
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){
            for (int j = tidy + 1; j < Ny; j += ythreads){
                intvar[IDX3D(0, j, k, Nx, Ny, Nz)] = intRhoTop(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
                intvar[IDX3D(0, j, k, Nx, Ny, Nz) + cube_size] = intRhoVXTop(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
                intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYTop(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
                intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZTop(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
                intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 4 * cube_size] = intBXTop(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
                intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 5 * cube_size] = intBYTop(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
                intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 6 * cube_size] = intBZTop(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
                intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 7 * cube_size] = intETop(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
            }
        }

        // Bottom Face
        // i = Nx - 1
        // j \in [1,Ny-1]
        // k \in [1,Nz-2]
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){
            for (int j = tidy + 1; j < Ny; j += ythreads){
                intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz)] = intRho(Nx-1, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
                intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + cube_size] = intRhoVX(Nx-1, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
                intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(Nx-1, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
                intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(Nx-1, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
                intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 4 * cube_size] = intBx(Nx-1, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
                intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 5 * cube_size] = intBy(Nx-1, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
                intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 6 * cube_size] = intBz(Nx-1, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
                intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 7 * cube_size] = intE(Nx-1, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
            }
        }

        // Left Face
        // j = 0
        // i \in [1, Nx-2]
        // k \in [1, Nz-2]
        int j = 0;
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){
            for (int i = tidx + 1; i < Nx - 1; i += xthreads){
                intvar[IDX3D(i, j, k, Nx, Ny, Nz)] = intRhoLeft(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = intRhoVXLeft(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYLeft(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZLeft(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = intBXLeft(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = intBYLeft(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = intBZLeft(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
                intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = intELeft(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
            }
        }

        // Right Face
        // j = Ny - 1
        // i \in [1,Nx-2] 
        // k \in [1,Nz-2]
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){
            for (int i = tidx + 1; i < Nx - 1; i += xthreads){
                intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz)] = intRho(i, Ny-1, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
                intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + cube_size] = intRhoVX(i, Ny-1, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
                intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(i, Ny-1, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
                intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(i, Ny-1, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
                intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 4 * cube_size] = intBx(i, Ny-1, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
                intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 5 * cube_size] = intBy(i, Ny-1, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
                intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 6 * cube_size] = intBz(i, Ny-1, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
                intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 7 * cube_size] = intE(i, Ny-1, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
            }
        }

        // After the above is done, there are still SIX lines where data has not been specified
        // {(0,j,0), (0,j,Nz-1), (0,0,k), (Nx-1,0,k), (i,0,0), (i,0,Nz-1)}
        // (0, j, 0) - "FrontTop"
        for (int j = tidy + 1; j < Ny; j += ythreads){
                intvar[IDX3D(0, j, 0, Nx, Ny, Nz)] = intRhoFrontTop(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
                intvar[IDX3D(0, j, 0, Nx, Ny, Nz) + cube_size] = intRhoVXFrontTop(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
                intvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYFrontTop(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
                intvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZFrontTop(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
                intvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 4 * cube_size] = intBXFrontTop(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
                intvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 5 * cube_size] = intBYFrontTop(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
                intvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 6 * cube_size] = intBZFrontTop(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
                intvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 7 * cube_size] = intEFrontTop(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // (0, j, Nz-1) - "TopBack"
        for (int j = tidy + 1; j < Ny; j += ythreads){
                intvar[IDX3D(0, j, Nz-1, Nx, Ny, Nz)] = intRho(0, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
                intvar[IDX3D(0, j, Nz-1, Nx, Ny, Nz) + cube_size] = intRhoVX(0, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
                intvar[IDX3D(0, j, Nz-1, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(0, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
                intvar[IDX3D(0, j, Nz-1, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(0, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
                intvar[IDX3D(0, j, Nz-1, Nx, Ny, Nz) + 4 * cube_size] = intBX(0, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
                intvar[IDX3D(0, j, Nz-1, Nx, Ny, Nz) + 5 * cube_size] = intBY(0, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
                intvar[IDX3D(0, j, Nz-1, Nx, Ny, Nz) + 6 * cube_size] = intBZ(0, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
                intvar[IDX3D(0, j, Nz-1, Nx, Ny, Nz) + 7 * cube_size] = intE(0, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // (0, 0, k) - "TopLeft"
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){
            intvar[IDX3D(0, 0, k, Nx, Ny, Nz)] = intRhoTopLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(0, 0, k, Nx, Ny, Nz) + cube_size] = intRhoVXTopLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYTopLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZTopLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 4 * cube_size] = intBXTopLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 5 * cube_size] = intBYTopLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 6 * cube_size] = intBZTopLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 7 * cube_size] = intETopLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }
        
        // (Nx-1, 0, k) - "BottomLeft"
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){
            intvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz)] = intRhoBottomLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + cube_size] = intRhoVXBottomLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYBottomLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZBottomLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 4 * cube_size] = intBXBottomLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 5 * cube_size] = intBYBottomLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 6 * cube_size] = intBZBottomLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 7 * cube_size] = intEBottomLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // (i, 0, 0) - "FrontLeft"
        for (int i = tidx + 1; i < Nx; i += xthreads){
            intvar[IDX3D(i, 0, 0, Nx, Ny, Nz)] = intRhoFrontLeft(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + cube_size] = intRhoVXFrontLeft(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYFrontLeft(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZFrontLeft(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 4 * cube_size] = intBXFrontLeft(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 5 * cube_size] = intBYFrontLeft(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 6 * cube_size] = intBZFrontLeft(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 7 * cube_size] = intEFrontLeft(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // (i, 0, Nz-1) - "BackLeft"
        for (int i = tidx + 1; i < Nx; i += xthreads){
            intvar[IDX3D(i, 0, Nz-1, Nx, Ny, Nz)] = intRho(i, 0, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(i, 0, Nz-1, Nx, Ny, Nz) + cube_size] = intRhoVX(i, 0, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(i, 0, Nz-1, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(i, 0, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(i, 0, Nz-1, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(i, 0, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(i, 0, Nz-1, Nx, Ny, Nz) + 4 * cube_size] = intBX(i, 0, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(i, 0, Nz-1, Nx, Ny, Nz) + 5 * cube_size] = intBY(i, 0, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(i, 0, Nz-1, Nx, Ny, Nz) + 6 * cube_size] = intBZ(i, 0, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(i, 0, Nz-1, Nx, Ny, Nz) + 7 * cube_size] = intE(i, 0, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // Pick up straggler points: {(0, 0, 0), (), (), (), ()}
        // NOTE: THESE DON'T NEED TO BE UPDATED (?)
        // THEY AREN'T INVOLVED IN ANYTHING SALIENT (?) 
        // THEY JUST NEED TO BE SET CORRECTLY INITIALLY (?)
        return;
    }

// These kernels deal with the interior of the computational volume
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

__device__ float intBX(const int i, const int j, const int k,
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

__device__ float intBY(const int i, const int j, const int k, 
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

__device__ float intBZ(const int i, const int j, const int k, 
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

// Kernels that deal with the interior of the k = 0 plane: (i, j, 0)
__device__ float intRhoFront(const int i, const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz)]
            - (dt / dx) * (XFluxRho(i, j, 0, fluidvar, Nx, Ny, Nz) - XFluxRho(i-1, j, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxRho(i, j, 0, fluidvar, Nx, Ny, Nz) - YFluxRho(i, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRho(i, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxRho(i, j, Nz-2, fluidvar, Nx, Ny, Nz));        
    }

__device__ float intRhoVXFront(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVX(i, j, 0, fluidvar, Nx, Ny, Nz) - XFluxRhoVX(i-1, j, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxRhoVX(i, j, 0, fluidvar, Nx, Ny, Nz) - YFluxRhoVX(i, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRhoVX(i, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(i, j, Nz-2, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVYFront(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVY(i, j, 0, fluidvar, Nx, Ny, Nz) - XFluxRhoVY(i-1, j, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxRhoVY(i, j, 0, fluidvar, Nx, Ny, Nz) - YFluxRhoVY(i, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRhoVY(i, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(i, j, Nz-2, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVZFront(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 3 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVZ(i, j, 0, fluidvar, Nx, Ny, Nz) - XFluxRhoVZ(i-1, j, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxRhoVZ(i, j, 0, fluidvar, Nx, Ny, Nz) - YFluxRhoVZ(i, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRhoVZ(i, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, j, Nz-2, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBXFront(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBX(i, j, 0, fluidvar, Nx, Ny, Nz) - XFluxBX(i-1, j, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxBX(i, j, 0, fluidvar, Nx, Ny, Nz) - YFluxBX(i, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxBX(i, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxBX(i, j, Nz-2, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBYFront(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBY(i, j, 0, fluidvar, Nx, Ny, Nz) - XFluxBY(i-1, j, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxBY(i, j, 0, fluidvar, Nx, Ny, Nz) - YFluxBY(i, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxBY(i, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxBY(i, j, Nz-2, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBZFront(const int i, const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBZ(i, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBZ(i, j, 0, fluidvar, Nx, Ny, Nz) - YFluxBZ(i, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxBZ(i, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxBZ(i, j, Nz-2, fluidvar, Nx, Ny, Nz));  
    }

__device__ float intEFront(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxE(i, j, 0, fluidvar, Nx, Ny, Nz) - XFluxE(i-1, j, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxE(i, j, 0, fluidvar, Nx, Ny, Nz) - YFluxE(i, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxE(i, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxE(i, j, Nz-2, fluidvar, Nx, Ny, Nz));
    }

// Kernels that deal with the (0, j, 0) line
__device__ float intRhoFrontTop(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz)]
            - (dt / dx) * (XFluxRho(0, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRho(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxRho(0, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRho(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxRho(0, j, Nz-2, fluidvar, Nx, Ny, Nz));        
    }

__device__ float intRhoVXFrontTop(const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVX(0, j, 0, fluidvar, Nx, Ny, Nz)) // i = -1 flux is zero => rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVX(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxRhoVX(0, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRhoVX(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(0, j, Nx-2, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVYFrontTop(const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVY(0, j, 0, fluidvar, Nx, Ny, Nz)) // i = -1 flux is zero => rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVY(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxRhoVY(0, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRhoVY(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(0, j, Nz-2, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVZFrontTop(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 3 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVZ(0, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVZ(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxRhoVZ(0, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRhoVZ(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(0, j, Nz-2, fluidvar, Nx, Ny, Nz));  
    }

__device__ float intBXFrontTop(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBX(0, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBX(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxBX(0, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxBX(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxBX(0, j, Nz-2, fluidvar, Nx, Ny, Nz));  
    }

__device__ float intBYFrontTop(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBY(0, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBY(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxBY(0, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxBY(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxBY(0, j, Nz-2, fluidvar, Nx, Ny, Nz));  
    }

__device__ float intBZFrontTop(const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBZ(0, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBZ(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxBZ(0, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxBZ(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxBZ(0, j, Nz-2, fluidvar, Nx, Ny, Nz));  
    }

__device__ float intEFrontTop(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxE(0, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxE(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxE(0, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxE(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxE(0, j, Nz-2, fluidvar, Nx, Ny, Nz));  
    }

// Kernels that deal with Front Left Face
// i \in [1, Nx-1]
// j = 0
// k = 0
__device__ float intRhoFrontLeft(const int i, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz)]
            - (dt / dx) * (XFluxRho(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxRho(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxRho(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRho(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall       
    }

__device__ float intRhoVXFrontLeft(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVX(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxRhoVX(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxRhoVX(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVX(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
    }

__device__ float intRhoVYFrontLeft(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVY(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxRhoVY(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxRhoVY(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVY(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
    }

__device__ float intRhoVZFrontLeft(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 3 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVZ(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxRhoVZ(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxRhoVZ(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVZ(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
    }

__device__ float intBXFrontLeft(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBX(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxBX(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxBX(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBX(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
    }

__device__ float intBYFrontLeft(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBY(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxBY(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxBY(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBY(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
    }

__device__ float intBZFrontLeft(const int i, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBZ(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxBZ(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxBZ(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBZ(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall  
    }

__device__ float intEFrontLeft(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxE(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxE(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxE(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxE(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
    }

// Kernels that deal with the Left Face 
// j = 0 
// i \in [1, Nx-1]
// k \in [1, Nz-2]
__device__ float intRhoLeft(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz)]
            - (dt / dx) * (XFluxRho(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRho(i-1, 0, k, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxRho(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRho(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRho(i, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVXLeft(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVX(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVX(i-1, 0, k, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxRhoVX(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVX(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(i, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVYLeft(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVY(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVY(i-1, 0, k, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxRhoVY(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVY(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(i, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVZLeft(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVZ(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVZ(i-1, 0, k, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxRhoVZ(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVZ(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBXLeft(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBX(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxBX(i-1, 0, k, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxBX(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBX(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBX(i, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBYLeft(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBY(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxBY(i-1, 0, k, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxBY(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBY(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBY(i, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBZLeft(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBZ(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxBZ(i-1, 0, k, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxBZ(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBZ(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBZ(i, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intELeft(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxE(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxE(i-1, 0, k, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxE(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxE(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxE(i, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

// Kernels that deal with Top Face
// i = 0
// j \in [1,Ny-1]
// k \in [1,Nz-2] - leave the front / back faces alone
__device__ float intRhoTop(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz)]
            - (dt / dx) * (XFluxRho(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRho(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxRho(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxRho(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRho(0, j, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVXTop(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVX(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVX(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVX(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxRhoVX(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(0, j, k-1, fluidvar, Nx, Ny, Nz));
    }
    
__device__ float intRhoVYTop(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVY(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVY(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVY(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxRhoVY(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(0, j, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVZTop(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 3 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVZ(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVZ(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVZ(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxRhoVZ(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(0, j, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBXTop(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBX(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBX(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxBX(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxBX(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxBX(0, j, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBYTop(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBY(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBY(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxBY(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxBY(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxBY(0, j, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBZTop(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBZ(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBZ(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxBZ(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxBZ(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxBZ(0, j, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intETop(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxE(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxE(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxE(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxE(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxE(0, j, k-1, fluidvar, Nx, Ny, Nz));
    }

// Kernels that deal with the Top Left Line
// i = 0
// j = 0
// k \in [1, Nz-1]
__device__ float intRhoTopLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz)]
            - (dt / dx) * (XFluxRho(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRho(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRho(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRho(0, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVXTopLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVX(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVX(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVX(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(0, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVYTopLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVY(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVY(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVY(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(0, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVZTopLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVZ(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVZ(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVZ(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(0, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBXTopLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBX(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBX(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBX(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBX(0, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBYTopLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBY(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBY(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBY(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBY(0, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBZTopLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBZ(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBZ(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBZ(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBZ(0, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intETopLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxE(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxE(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxE(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxE(0, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

// Kernels that deal with the Bottom Left Face
// i = Nx-1
// j = 0
// k \in [1, Nz-2]
__device__ float intRhoBottomLeft(const int k, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz)]
            - (dt / dx) * (XFluxRho(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRho(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxRho(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRho(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRho(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));        
    }

__device__ float intRhoVXBottomLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVX(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVX(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxRhoVX(Nx-1, 0, k, fluidvar, Nx, Ny, Nz))  // No flux on opposite site of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVX(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVYBottomLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVY(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVY(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxRhoVY(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVY(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVZBottomLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 3 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVZ(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVZ(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxRhoVZ(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVZ(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBXBottomLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBX(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxBX(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxBX(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBX(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBX(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBYBottomLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBY(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxBY(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxBY(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBY(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBY(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBZBottomLeft(const int k, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBZ(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxBZ(Nx-2, 0, k, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxBZ(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBZ(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBZ(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));  
    }

__device__ float intEBottomLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxE(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxE(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxE(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxE(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxE(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));
    }
/* 
THERE WILL BE OTHERS 
*/