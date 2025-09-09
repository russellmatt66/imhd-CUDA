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
CUDA (__global__) kernels

REGISTER PRESSURES: (registers per thread)
PBCsInX=?
PBCsInY=?
PBCsInZ=28
rigidConductingWallBCsLeftRight=30
rigidConductingWallBCsTopBottom=30
*/

// Periodic Boundary Conditions (PBCs)
/* REG / THREAD? */
__global__ void PBCsInX(float* fluidvars, const int Nx, const int Ny, const int Nz)
{
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;

    int cube_size = Nx * Ny * Nz;

    if (j < Ny && k < Nz){
        for (int ivf = 0; ivf < 8; ivf++){
            fluidvars[IDX3D(0, j, k, Nx, Ny, Nz) + ivf * cube_size] = fluidvars[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + ivf * cube_size];
        }
    }

    return;
}

/* REG / THREAD? */
__global__ void PBCsInY(float* fluidvars, const int Nx, const int Ny, const int Nz)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int k = threadIdx.z + blockDim.z * blockIdx.z;

    int cube_size = Nx * Ny * Nz;

    if (i < Nx && k < Nz){
        for (int ivf = 0; ivf < 8; ivf++){
            fluidvars[IDX3D(i, 0, k, Nx, Ny, Nz) + ivf * cube_size] = fluidvars[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + ivf * cube_size];
        }
    }

    return;
}

// 28 registers per thread
__global__ void PBCsInZ(float* fluidvars, const int Nx, const int Ny, const int Nz)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    int cube_size = Nx * Ny * Nz;

    if (i < Nx && j < Ny){
        for (int ivf = 0; ivf < 8; ivf++){
            fluidvars[IDX3D(i, j, 0, Nx, Ny, Nz) + ivf * cube_size] = fluidvars[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + ivf * cube_size];
        }
    }
    return;
}

// Perfectly-Conducting, Rigid Wall (PCRW) kernels
// Granular, single-face, kernels
// Left = (i, 0, k)
__global__ void PCRWFluidBCsLeft(float* fluidvars, const int Nx, const int Ny, const int Nz)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int k = threadIdx.z + blockDim.z * blockIdx.z;

    int cube_size = Nx * Ny * Nz;

    if (i < Nx && k < Nz){
        fluidvars[IDX3D(i, 0, k, Nx, Ny, Nz)] = 1.0; // Magic wall number for density
        fluidvars[IDX3D(i, 0, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
        fluidvars[IDX3D(i, 0, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvars[IDX3D(i, 0, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvars[IDX3D(i, 0, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
        fluidvars[IDX3D(i, 0, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
        fluidvars[IDX3D(i, 0, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvars[IDX3D(i, 0, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, 0, k, fluidvars, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
    }
    return;
}

// Right = (i, Ny-1, k)
__global__ void PCRWFluidBCsRight(float* fluidvars, const int Nx, const int Ny, const int Nz)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int k = threadIdx.z + blockDim.z * blockIdx.z;

    int cube_size = Nx * Ny * Nz;

    if (i < Nx && k < Nz){
        fluidvars[IDX3D(i, Ny-1, k, Nx, Ny, Nz)] = 1.0; // Magic wall number for density
        fluidvars[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
        fluidvars[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvars[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvars[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
        fluidvars[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
        fluidvars[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvars[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, Ny-1, k, fluidvars, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
    }
    return;
}

// Top = (0, j, k)
__global__ void PCRWFluidBCsTop(float* fluidvars, const int Nx, const int Ny, const int Nz)
{
    int j = threadIdx.x + blockDim.x * blockIdx.x;
    int k = threadIdx.z + blockDim.z * blockIdx.z;

    int cube_size = Nx * Ny * Nz;

    if (j < Ny && k < Nz){ 
        fluidvars[IDX3D(0, j, k, Nx, Ny, Nz)] = 1.0; // Magic wall number for density
        fluidvars[IDX3D(0, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
        fluidvars[IDX3D(0, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvars[IDX3D(0, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvars[IDX3D(0, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
        fluidvars[IDX3D(0, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
        fluidvars[IDX3D(0, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvars[IDX3D(0, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(0, j, k, fluidvars, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
    } 
    return;
}

// Bottom = (Nx-1, j, k)
__global__ void PCRWFluidBCsBottom(float* fluidvars, const int Nx, const int Ny, const int Nz)
{
    int j = threadIdx.x + blockDim.x * blockIdx.x;
    int k = threadIdx.z + blockDim.z * blockIdx.z;

    int cube_size = Nx * Ny * Nz;

    if (j < Ny && k < Nz){ 
        fluidvars[IDX3D(Nx-1, j, k, Nx, Ny, Nz)] = 1.0; // Magic wall number for density
        fluidvars[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
        fluidvars[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvars[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvars[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
        fluidvars[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
        fluidvars[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvars[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(Nx-1, j, k, fluidvars, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
    } 
    return;
}

// Front = (i, j, 0)
__global__ void PCRWFluidBCsFront(float* fluidvars, const int Nx, const int Ny, const int Nz)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    int cube_size = Nx * Ny * Nz;

    if (i < Nx && j < Ny){
        fluidvars[IDX3D(i, j, 0, Nx, Ny, Nz)] = 1.0;
        fluidvars[IDX3D(i, j, 0, Nx, Ny, Nz) + cube_size] = 0.0;
        fluidvars[IDX3D(i, j, 0, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvars[IDX3D(i, j, 0, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvars[IDX3D(i, j, 0, Nx, Ny, Nz) + 4 * cube_size] = 0.0;
        fluidvars[IDX3D(i, j, 0, Nx, Ny, Nz) + 5 * cube_size] = 0.0;
        fluidvars[IDX3D(i, j, 0, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvars[IDX3D(i, j, 0, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, 0, fluidvars, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
    }

    return;
}

// Back = (i, j, Nz-1)
__global__ void PCRWFluidBCsBack(float* fluidvars, const int Nx, const int Ny, const int Nz)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    int cube_size = Nx * Ny * Nz;

    if (i < Nx && j < Ny){
        fluidvars[IDX3D(i, j, Nz-1, Nx, Ny, Nz)] = 1.0;
        fluidvars[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + cube_size] = 0.0;
        fluidvars[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvars[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvars[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 4 * cube_size] = 0.0;
        fluidvars[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 5 * cube_size] = 0.0;
        fluidvars[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvars[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, Nz-1, fluidvars, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
    }

    return;
}


// xz-planes (j = 0, j = Ny-1)
// 30 registers per thread 
__global__ void rigidConductingWallBCsLeftRight(float* fluidvars, const int Nx, const int Ny, const int Nz){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int k = threadIdx.z + blockDim.z * blockIdx.z;

    int cube_size = Nx * Ny * Nz;

    // if (i < Nx && k > 0 && k < Nz-1){ // This was the previous line - change is because this kernel should do a complete sweep over the sides
    if (i < Nx && k < Nz){
        // Left
        fluidvars[IDX3D(i, 0, k, Nx, Ny, Nz)] = 1.0; // Magic wall number for density
        fluidvars[IDX3D(i, 0, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
        fluidvars[IDX3D(i, 0, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvars[IDX3D(i, 0, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvars[IDX3D(i, 0, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
        fluidvars[IDX3D(i, 0, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
        fluidvars[IDX3D(i, 0, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvars[IDX3D(i, 0, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, 0, k, fluidvars, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
        
        // Right
        fluidvars[IDX3D(i, Ny-1, k, Nx, Ny, Nz)] = 1.0; // Magic wall number for density
        fluidvars[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
        fluidvars[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvars[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvars[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
        fluidvars[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
        fluidvars[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvars[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, Ny-1, k, fluidvars, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
    } 
    return;
}

// yz-planes (i = 0, Nx-1) 
// 30 registers per thread
__global__ void rigidConductingWallBCsTopBottom(float* fluidvars, const int Nx, const int Ny, const int Nz){

    int j = threadIdx.x + blockDim.x * blockIdx.x;
    int k = threadIdx.z + blockDim.z * blockIdx.z;

    int cube_size = Nx * Ny * Nz;

    // if (j < Ny && k > 0 && k < Nz-1){ // This was the previous line - change is because this kernel should do a complete sweep over the sides
    if (j < Ny && k < Nz){ 
        // Top
        fluidvars[IDX3D(0, j, k, Nx, Ny, Nz)] = 1.0; // Magic wall number for density
        fluidvars[IDX3D(0, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
        fluidvars[IDX3D(0, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvars[IDX3D(0, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvars[IDX3D(0, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
        fluidvars[IDX3D(0, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
        fluidvars[IDX3D(0, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvars[IDX3D(0, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(0, j, k, fluidvars, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
        
        // Bottom
        fluidvars[IDX3D(Nx-1, j, k, Nx, Ny, Nz)] = 1.0; // Magic wall number for density
        fluidvars[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
        fluidvars[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvars[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvars[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
        fluidvars[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
        fluidvars[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvars[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(Nx-1, j, k, fluidvars, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
    } 
    return;
}

// xy-planes (k = 0, Nz-1)
/* REG / THREAD? */
__global__ void rigidConductingWallBCsFrontBack(float* fluidvars, const int Nx, const int Ny, const int Nz){

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    int cube_size = Nx * Ny * Nz;

    if (i < Nx && j < Ny){
        // Front, k = 0
        fluidvars[IDX3D(i, j, 0, Nx, Ny, Nz)] = 1.0;
        fluidvars[IDX3D(i, j, 0, Nx, Ny, Nz) + cube_size] = 0.0;
        fluidvars[IDX3D(i, j, 0, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvars[IDX3D(i, j, 0, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvars[IDX3D(i, j, 0, Nx, Ny, Nz) + 4 * cube_size] = 0.0;
        fluidvars[IDX3D(i, j, 0, Nx, Ny, Nz) + 5 * cube_size] = 0.0;
        fluidvars[IDX3D(i, j, 0, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvars[IDX3D(i, j, 0, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, 0, fluidvars, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
        
        // Back, k = Nz-1
        fluidvars[IDX3D(i, j, Nz-1, Nx, Ny, Nz)] = 1.0;
        fluidvars[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + cube_size] = 0.0;
        fluidvars[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        fluidvars[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        fluidvars[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 4 * cube_size] = 0.0;
        fluidvars[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 5 * cube_size] = 0.0;
        fluidvars[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        fluidvars[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, Nz-1, fluidvars, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
    }

    return;
}

/*
// Non-blocking launchers of the CUDA kernels
*/
// Whole point is to collect information into `BoundaryConfig` data type for modularity
void LaunchFluidBCsPCRWXYPBCZ(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg){
   rigidConductingWallBCsLeftRight<<<bcfg.egd_leftright, bcfg.tbd_leftright>>>(fluidvars, Nx, Ny, Nz);
   rigidConductingWallBCsTopBottom<<<bcfg.egd_topbottom, bcfg.tbd_topbottom>>>(fluidvars, Nx, Ny, Nz);
   PBCsInZ<<<bcfg.egd_frontback, bcfg.tbd_frontback>>>(fluidvars, Nx, Ny, Nz);
   return;
}

/* TOO GRANULAR - delete when safe */
// void LaunchFluidBCsPBCX(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg){
//     PBCsInX<<<bcfg.egd_topbottom, bcfg.tbd_topbottom>>>(fluidvars, Nx, Ny, Nz);
//     return;
// }

// void LaunchFluidBCsPBCY(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg){
//     PBCsInY<<<bcfg.egd_leftright, bcfg.tbd_leftright>>>(fluidvars, Nx, Ny, Nz);
//     return;
// }

// void LaunchFluidBCsPBCZ(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg){
//     PBCsInZ<<<bcfg.egd_frontback, bcfg.tbd_frontback>>>(fluidvars, Nx, Ny, Nz);
//     return;
// }

// // Granular, single-face kernels that specify a PCRW boundary condition  
// void LaunchFluidBCsPCRWTop(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg){
//     PCRWFluidBCsTop<<<bcfg.egd_topbottom, bcfg.tbd_topbottom>>>(fluidvars, Nx, Ny, Nz);
//     return;
// }

// void LaunchFluidBCsPCRWBottom(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg){
//     PCRWFluidBCsBottom<<<bcfg.egd_topbottom, bcfg.tbd_topbottom>>>(fluidvars, Nx, Ny, Nz);
//     return;
// }

// void LaunchFluidBCsPCRWLeft(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg){
//     PCRWFluidBCsLeft<<<bcfg.egd_leftright, bcfg.tbd_leftright>>>(fluidvars, Nx, Ny, Nz);
//     return;
// }

// void LaunchFluidBCsPCRWRight(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg){
//     PCRWFluidBCsRight<<<bcfg.egd_leftright, bcfg.tbd_leftright>>>(fluidvars, Nx, Ny, Nz);
//     return;
// }

// void LaunchFluidBCsPCRWFront(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg){
//     PCRWFluidBCsFront<<<bcfg.egd_frontback, bcfg.tbd_frontback>>>(fluidvars, Nx, Ny, Nz);
//     return;
// }

// void LaunchFluidBCsPCRWBack(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg){
//     PCRWFluidBCsBack<<<bcfg.egd_frontback, bcfg.tbd_frontback>>>(fluidvars, Nx, Ny, Nz);
//     return;
// }

// // Legacy kernels which launch more than one PCRW kernel
// // Non-blocking. See inside for reason.
// void LaunchFluidBCsPCRWXY(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg){
//     /*
//     The overlap in the computational sub-domains of these kernels should not be a serious cause of concern for a data race
//     Any permutation of the order in which the threads could write to the locations will result in the same output 
//     because the same data is being written regardless. 
//     */
//     rigidConductingWallBCsLeftRight<<<bcfg.egd_leftright, bcfg.tbd_leftright>>>(fluidvars, Nx, Ny, Nz);
//     rigidConductingWallBCsTopBottom<<<bcfg.egd_topbottom, bcfg.tbd_topbottom>>>(fluidvars, Nx, Ny, Nz);
//     return;
// }

// void LaunchFluidBCsPCRWXZ(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg){
//     /*
//     See `void LaunchFluidBCsPCRWXY(...)` for explanation of why this is non-blocking 
//     */
//     rigidConductingWallBCsTopBottom<<<bcfg.egd_topbottom, bcfg.tbd_topbottom>>>(fluidvars, Nx, Ny, Nz);
//     rigidConductingWallBCsFrontBack<<<bcfg.egd_frontback, bcfg.tbd_frontback>>>(fluidvars, Nx, Ny, Nz);
//     return;
// }

// void LaunchFluidBCsPCRWYZ(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg){
//     /*
//     See previous for explanation of why this is non-blocking
//     */
//     rigidConductingWallBCsLeftRight<<<bcfg.egd_leftright, bcfg.tbd_leftright>>>(fluidvars, Nx, Ny, Nz);
//     rigidConductingWallBCsFrontBack<<<bcfg.egd_frontback, bcfg.tbd_frontback>>>(fluidvars, Nx, Ny, Nz);
//     return;
// }