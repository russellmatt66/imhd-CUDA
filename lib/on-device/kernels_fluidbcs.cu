// What: This is where the kernels for the specification of the fluid boundary conditions go
// Why: Separation of Concerns

#include <stdio.h>

#include "kernels_fluidbcs.cuh"
#include "kernels_od.cuh"
#include "kernels_od_fluxes.cuh"
#include "diffusion.cuh"
#include "helper_functions.cuh"

/* THIS SHOULD BE DEFINED A SINGLE TIME IN A SINGLE PLACE */
#define IDX3D(i, j, k, Nx, Ny, Nz) ((k) * (Nx) * (Ny) + (i) * (Ny) + j) // parentheses are necessary to avoid calculating `i - 1 * Ny` or `k - 1 * (Nx * Ny)`

/* 
REGISTER PRESSURES: (registers per thread)
BoundaryConditions=74
BoundaryConditionsNoDiff=40
rigidConductingWallBCsLeftRight=30
rigidConductingWallBCsTopBottom=30
PBCs=28
*/

/* 
NOTE:
This is a  really ugly MVP
It's still here because it works and serves as a reference
REALLY INEFFICIENT
DON'T USE
*/ 
// 74 registers per thread
__global__ void BoundaryConditions(float* fluidvar, const float* intvar, 
    const float D, const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
    int i = threadIdx.x + blockDim.x * blockIdx.x; 
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;

    int xthreads = blockDim.x * gridDim.x;
    int ythreads = blockDim.y * gridDim.y;
    int zthreads = blockDim.z * gridDim.z;

    int cube_size = Nx * Ny * Nz;

    // if (tidx > Nx || tidy > Ny){
    //     printf("Beginning Hello from (%d, %d, %d))\n", tidx, tidy, tidz);
    // }

    // k = 0
    k = 0;
    if (i > 0 && i < Nx - 1 && j > 0 && j < Ny - 1){
        // k = 0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] + intvar[IDX3D(i, j, k, Nx, Ny, Nz)])
                                                    - 0.5 * (dt / dx) * (XFluxRho(i, j, k, intvar, Nx, Ny, Nz) - XFluxRho(i-1, j, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dy) * (YFluxRho(i, j, k, intvar, Nx, Ny, Nz) - YFluxRho(i, j-1, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dz) * (ZFluxRho(i, j, k, intvar, Nx, Ny, Nz) - ZFluxRho(i, j, Nz-2, intvar, Nx, Ny, Nz))
                                                    + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 0, Nx, Ny, Nz);
        // printf("Made it past rho update - Front\n");

        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size])
                                                    - 0.5 * (dt / dx) * (XFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVX(i-1, j, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dy) * (YFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz) - YFluxRhoVX(i, j-1, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dz) * (ZFluxRhoVX(i, j, k , intvar, Nx, Ny, Nz) - ZFluxRhoVX(i, j, Nz-2, intvar, Nx, Ny, Nz))
                                                    + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 1, Nx, Ny, Nz);
        // printf("Made it past rhovx update - Front\n");
        

        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size])
                                                    - 0.5 * (dt / dx) * (XFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVY(i-1, j, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dy) * (YFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz) - YFluxRhoVY(i, j-1, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dz) * (ZFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz) - ZFluxRhoVY(i, j, Nz-2, intvar, Nx, Ny, Nz))
                                                    + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 2, Nx, Ny, Nz);
        // printf("Made it past rhovy update - Front\n");


        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size])
                                                    - 0.5 * (dt / dx) * (XFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVZ(i-1, j, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dy) * (YFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz) - YFluxRhoVZ(i, j-1, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dz) * (ZFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, j, Nz-2, intvar, Nx, Ny, Nz))
                                                    + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 3, Nx, Ny, Nz);
        // printf("Made it past rhovz update - Front\n");


        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size])
                                                    - 0.5 * (dt / dx) * (XFluxBX(i, j, k, intvar, Nx, Ny, Nz) - XFluxBX(i-1, j, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dy) * (YFluxBX(i, j, k, intvar, Nx, Ny, Nz) - YFluxBX(i, j-1, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dz) * (ZFluxBX(i, j, k, intvar, Nx, Ny, Nz) - ZFluxBX(i, j, Nz-2, intvar, Nx, Ny, Nz))
                                                    + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 4, Nx, Ny, Nz);
        // printf("Made it past Bx update - Front\n");


        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size])
                                                    - 0.5 * (dt / dx) * (XFluxBY(i, j, k, intvar, Nx, Ny, Nz) - XFluxBY(i-1, j, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dy) * (YFluxBY(i, j, k, intvar, Nx, Ny, Nz) - YFluxBY(i, j-1, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dz) * (ZFluxBY(i, j, k, intvar, Nx, Ny, Nz) - ZFluxBY(i, j, Nz-2, intvar, Nx, Ny, Nz))
                                                    + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 5, Nx, Ny, Nz);
        // printf("Made it past By update - Front\n");


        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size])
                                                    - 0.5 * (dt / dx) * (XFluxBZ(i, j, k, intvar, Nx, Ny, Nz) - XFluxBZ(i-1, j, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dy) * (YFluxBZ(i, j, k, intvar, Nx, Ny, Nz) - YFluxBZ(i, j-1, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dz) * (ZFluxBZ(i, j, k, intvar, Nx, Ny, Nz) - ZFluxBZ(i, j, Nz-2, intvar, Nx, Ny, Nz))
                                                    + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 6, Nx, Ny, Nz);
        // printf("Made it past Bz update - Front\n");


        // printf("Trying to update e: (%d, %d, %d)\n", i, j, k);
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size])
                                                    - 0.5 * (dt / dx) * (XFluxE(i, j, k, intvar, Nx, Ny, Nz) - XFluxE(i-1, j, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dy) * (YFluxE(i, j, k, intvar, Nx, Ny, Nz) - YFluxE(i, j-1, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dz) * (ZFluxE(i, j, k, intvar, Nx, Ny, Nz) - ZFluxE(i, j, Nz-2, intvar, Nx, Ny, Nz))
                                                    + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 7, Nx, Ny, Nz);
        // printf("Made it past e update - Front\n");
    }
    // printf("Made it past Front update\n");

    /* DONT NEED THIS - PBCs*/
    // printf("Trying to update Back\n");
    // k = Nz - 1;
    // if (i > 0 && i < Nx - 1 && j > 0 && j < Ny - 1){
    //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvRho(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz);
    //                                                 + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 0, Nx, Ny, Nz);
    //     // printf("Made it past rho update - Back\n");

    //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = LaxWendroffAdvRhoVX(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz)
    //                                                 + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 1, Nx, Ny, Nz);
    //     // printf("Made it past rhovx update - Back\n");
        
    //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = LaxWendroffAdvRhoVY(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz)
    //                                                 + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 2, Nx, Ny, Nz);
    //     // printf("Made it past rhovy update - Back\n");

    //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = LaxWendroffAdvRhoVZ(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz)
    //                                                 + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 3, Nx, Ny, Nz);
    //     // printf("Made it past rhovz update - Back\n");

    //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = LaxWendroffAdvBX(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz)
    //                                                 + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 4, Nx, Ny, Nz);
    //     // printf("Made it past Bx update - Back\n");

    //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = LaxWendroffAdvBY(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz)
    //                                                 + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 5, Nx, Ny, Nz);
    //     // printf("Made it past By update - Back\n");

    //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = LaxWendroffAdvBZ(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz)
    //                                                 + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 6, Nx, Ny, Nz);
    //     // printf("Made it past Bz update - Back\n");

    //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = LaxWendroffAdvE(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz)
    //                                                 + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 7, Nx, Ny, Nz);
    //     // printf("Made it past e update - Back\n");
    // }
    // printf("Made it past update - Back\n");

    /* 
    B.Cs on Top (II) 
    (i = 0, j, k) 
    and
    B.Cs on Bottom (IV)
    (i = Nx-1, j, k) 
    */
    i = 0; 
    if (k < Nz && j < Ny){
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0; /* Magic vacuum number */
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
    }
    // printf("Made it past update - Top\n");


    i = Nx - 1;
    if (k < Nz && j < Ny){
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0; /* Magic vacuum number */
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
    }
    // printf("Made it past update - Bottom\n");

    /* 
    B.Cs on LEFT (V)
    (i, j = 0, k) 
    and
    B.Cs on RIGHT (III)
    (i, j = N-1, k) 
    */
    j = 0;
    if (k < Nz && i > 0 && i < Nx - 1){
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0; // Magic wall number
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
    }      
    // printf("Made it past update - Left\n");

    j = Ny - 1;
    if (k < Nz && i > 0 && i < Nx - 1){
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0; // Magic wall number 
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
    }     
    // printf("Made it past update - Right\n");

    __syncthreads(); // PBCs require this

    // PBCs
    if (i < Nx && j < Ny){ // Do not ignore the edges b/c intermediate variables got calculated there
        for (int ivf = 0; ivf < 8; ivf++){ // PBCs - they are the SAME point 
            fluidvar[IDX3D(i, j, Nz - 1, Nx, Ny, Nz) + ivf * cube_size] = fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz) + ivf * cube_size];
        }
    }


    return;
}

// 40 registers per thread
__global__ void BoundaryConditionsNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
    int i = threadIdx.x + blockDim.x * blockIdx.x; 
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;

    int xthreads = blockDim.x * gridDim.x;
    int ythreads = blockDim.y * gridDim.y;
    int zthreads = blockDim.z * gridDim.z;

    int cube_size = Nx * Ny * Nz;

    // if (tidx > Nx || tidy > Ny){
    //     printf("Beginning Hello from (%d, %d, %d))\n", tidx, tidy, tidz);
    // }

    // k = 0
    k = 0;
    if (i > 0 && i < Nx - 1 && j > 0 && j < Ny - 1){
        // k = 0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] + intvar[IDX3D(i, j, k, Nx, Ny, Nz)])
                                                    - 0.5 * (dt / dx) * (XFluxRho(i, j, k, intvar, Nx, Ny, Nz) - XFluxRho(i-1, j, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dy) * (YFluxRho(i, j, k, intvar, Nx, Ny, Nz) - YFluxRho(i, j-1, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dz) * (ZFluxRho(i, j, k, intvar, Nx, Ny, Nz) - ZFluxRho(i, j, Nz-2, intvar, Nx, Ny, Nz));
        // printf("Made it past rho update - Front\n");

        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size])
                                                    - 0.5 * (dt / dx) * (XFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVX(i-1, j, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dy) * (YFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz) - YFluxRhoVX(i, j-1, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dz) * (ZFluxRhoVX(i, j, k , intvar, Nx, Ny, Nz) - ZFluxRhoVX(i, j, Nz-2, intvar, Nx, Ny, Nz));
        // printf("Made it past rhovx update - Front\n");
        

        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size])
                                                    - 0.5 * (dt / dx) * (XFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVY(i-1, j, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dy) * (YFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz) - YFluxRhoVY(i, j-1, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dz) * (ZFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz) - ZFluxRhoVY(i, j, Nz-2, intvar, Nx, Ny, Nz));
        // printf("Made it past rhovy update - Front\n");


        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size])
                                                    - 0.5 * (dt / dx) * (XFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVZ(i-1, j, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dy) * (YFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz) - YFluxRhoVZ(i, j-1, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dz) * (ZFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, j, Nz-2, intvar, Nx, Ny, Nz));
        // printf("Made it past rhovz update - Front\n");


        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size])
                                                    - 0.5 * (dt / dx) * (XFluxBX(i, j, k, intvar, Nx, Ny, Nz) - XFluxBX(i-1, j, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dy) * (YFluxBX(i, j, k, intvar, Nx, Ny, Nz) - YFluxBX(i, j-1, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dz) * (ZFluxBX(i, j, k, intvar, Nx, Ny, Nz) - ZFluxBX(i, j, Nz-2, intvar, Nx, Ny, Nz));
        // printf("Made it past Bx update - Front\n");


        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size])
                                                    - 0.5 * (dt / dx) * (XFluxBY(i, j, k, intvar, Nx, Ny, Nz) - XFluxBY(i-1, j, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dy) * (YFluxBY(i, j, k, intvar, Nx, Ny, Nz) - YFluxBY(i, j-1, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dz) * (ZFluxBY(i, j, k, intvar, Nx, Ny, Nz) - ZFluxBY(i, j, Nz-2, intvar, Nx, Ny, Nz));
        // printf("Made it past By update - Front\n");


        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size])
                                                    - 0.5 * (dt / dx) * (XFluxBZ(i, j, k, intvar, Nx, Ny, Nz) - XFluxBZ(i-1, j, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dy) * (YFluxBZ(i, j, k, intvar, Nx, Ny, Nz) - YFluxBZ(i, j-1, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dz) * (ZFluxBZ(i, j, k, intvar, Nx, Ny, Nz) - ZFluxBZ(i, j, Nz-2, intvar, Nx, Ny, Nz));
        // printf("Made it past Bz update - Front\n");


        // printf("Trying to update e: (%d, %d, %d)\n", i, j, k);
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size])
                                                    - 0.5 * (dt / dx) * (XFluxE(i, j, k, intvar, Nx, Ny, Nz) - XFluxE(i-1, j, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dy) * (YFluxE(i, j, k, intvar, Nx, Ny, Nz) - YFluxE(i, j-1, k, intvar, Nx, Ny, Nz))
                                                    - 0.5 * (dt / dz) * (ZFluxE(i, j, k, intvar, Nx, Ny, Nz) - ZFluxE(i, j, Nz-2, intvar, Nx, Ny, Nz));
        // printf("Made it past e update - Front\n");
    }
    // printf("Made it past Front update\n");

    /* DONT NEED THIS - PBCs*/
    // printf("Trying to update Back\n");
    // k = Nz - 1;
    // if (i > 0 && i < Nx - 1 && j > 0 && j < Ny - 1){
    //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvRho(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz);
    //                                                 + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 0, Nx, Ny, Nz);
    //     // printf("Made it past rho update - Back\n");

    //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = LaxWendroffAdvRhoVX(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz)
    //                                                 + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 1, Nx, Ny, Nz);
    //     // printf("Made it past rhovx update - Back\n");
        
    //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = LaxWendroffAdvRhoVY(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz)
    //                                                 + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 2, Nx, Ny, Nz);
    //     // printf("Made it past rhovy update - Back\n");

    //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = LaxWendroffAdvRhoVZ(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz)
    //                                                 + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 3, Nx, Ny, Nz);
    //     // printf("Made it past rhovz update - Back\n");

    //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = LaxWendroffAdvBX(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz)
    //                                                 + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 4, Nx, Ny, Nz);
    //     // printf("Made it past Bx update - Back\n");

    //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = LaxWendroffAdvBY(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz)
    //                                                 + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 5, Nx, Ny, Nz);
    //     // printf("Made it past By update - Back\n");

    //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = LaxWendroffAdvBZ(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz)
    //                                                 + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 6, Nx, Ny, Nz);
    //     // printf("Made it past Bz update - Back\n");

    //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = LaxWendroffAdvE(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz)
    //                                                 + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 7, Nx, Ny, Nz);
    //     // printf("Made it past e update - Back\n");
    // }
    // printf("Made it past update - Back\n");

    /* 
    B.Cs on Top (II) 
    (i = 0, j, k) 
    and
    B.Cs on Bottom (IV)
    (i = Nx-1, j, k) 
    */
    i = 0; 
    if (k < Nz && j < Ny){
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0; /* Magic vacuum number */
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
    }
    // printf("Made it past update - Top\n");


    i = Nx - 1;
    if (k < Nz && j < Ny){
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0; /* Magic vacuum number */
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
    }
    // printf("Made it past update - Bottom\n");

    /* 
    B.Cs on LEFT (V)
    (i, j = 0, k) 
    and
    B.Cs on RIGHT (III)
    (i, j = N-1, k) 
    */
    j = 0;
    if (k < Nz && i > 0 && i < Nx - 1){
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0; // Magic wall number
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
    }      
    // printf("Made it past update - Left\n");

    j = Ny - 1;
    if (k < Nz && i > 0 && i < Nx - 1){
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0; // Magic wall number 
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
    }     
    // printf("Made it past update - Right\n");

    __syncthreads(); // PBCs require this

    // PBCs
    if (i < Nx && j < Ny){ // Do not ignore the edges b/c intermediate variables got calculated there
        for (int ivf = 0; ivf < 8; ivf++){ // PBCs - they are the SAME point 
            fluidvar[IDX3D(i, j, Nz - 1, Nx, Ny, Nz) + ivf * cube_size] = fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz) + ivf * cube_size];
        }
    }


    return;
}

// 30 registers per thread 
__global__ void rigidConductingWallBCsLeftRight(float* fluidvar, const int Nx, const int Ny, const int Nz){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int k = threadIdx.z + blockDim.z * blockIdx.z;

    int cube_size = Nx * Ny * Nz;

    if (i < Nx && k > 0 && k < Nz-1){
        // Left
        fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz)] = 1.0; // Magic wall number for density
        fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
        fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
        fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
        fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, 0, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
        
        // Right
        fluidvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz)] = 1.0; // Magic wall number for density
        fluidvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
        fluidvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
        fluidvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
        fluidvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, Ny-1, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
    } 
    return;
}

// 30 registers per thread
__global__ void rigidConductingWallBCsTopBottom(float* fluidvar, const int Nx, const int Ny, const int Nz){
    int j = threadIdx.x + blockDim.x * blockIdx.x;
    int k = threadIdx.z + blockDim.z * blockIdx.z;

    int cube_size = Nx * Ny * Nz;

    if (j > 0 && j < Ny && k > 0 && k < Nz-1){
        // Top
        fluidvar[IDX3D(0, j, k, Nx, Ny, Nz)] = 1.0; // Magic wall number for density
        fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
        fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
        fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
        fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(0, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
        
        // Bottom
        fluidvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz)] = 1.0; // Magic wall number for density
        fluidvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
        fluidvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
        fluidvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
        fluidvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(Nx-1, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
    } 
    return;
}

// 28 registers per thread
__global__ void PBCs(float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    int cube_size = Nx * Ny * Nz;

    if (i < Nx && j < Ny){
        for (int ivf = 0; ivf < 8; ivf++){
            fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz) + ivf * cube_size] = fluidvar[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + ivf * cube_size];
        }
    }
    return;
}