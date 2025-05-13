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
Global kernels

REGISTER PRESSURES: (registers per thread)
rigidConductingWallBCsLeftRight=30
rigidConductingWallBCsTopBottom=30
PBCs=28
*/
// Rigid, perfectly-conducting wall boundary conditions
// xz-planes (j = 0, j = Ny-1)
// 30 registers per thread 
__global__ void rigidConductingWallBCsLeftRight(float* fluidvars, const int Nx, const int Ny, const int Nz){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int k = threadIdx.z + blockDim.z * blockIdx.z;

    int cube_size = Nx * Ny * Nz;

    if (i < Nx && k > 0 && k < Nz-1){
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

    if (j > 0 && j < Ny && k > 0 && k < Nz-1){
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

// Periodic Boundary Conditions (PBCs)
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

// Non-blocking launchers of the above global kernels
// Whole point is to collect information into `BoundaryConfig` data type for modularity
void LaunchFluidBCsPBCZ(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg){
    PBCsInZ<<<bcfg.egd_frontback, bcfg.tbd_frontback>>>(fluidvars, Nx, Ny, Nz);
    return;
}

void LaunchFluidBCsPCRWXY(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg){
    rigidConductingWallBCsLeftRight<<<bcfg.egd_leftright, bcfg.tbd_leftright>>>(fluidvars, Nx, Ny, Nz);
    rigidConductingWallBCsTopBottom<<<bcfg.egd_topbottom, bcfg.tbd_topbottom>>>(fluidvars, Nx, Ny, Nz);
    return;
}