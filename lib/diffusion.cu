#include <stdio.h>
#include "diffusion.cuh"

#define IDX3D(i, j, k, Nx, Ny, Nz) ((k) * (Nx * Ny) + (i) * Ny + j) // parentheses are necessary to avoid calculating `i - 1 * Ny` or `k - 1 * (Nx * Ny)`

__device__ float numericalDiffusion(const int i, const int j, const int k, const float* fluid_var, 
    const float D, const float dx, const float dy, const float dz,
    const int ivf, const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        float num_diff = 0.0;

        // printf("(i, j, k) = (%d, %d, %d), Problem index is %d, cube size is %d, ivf is %d, computed linear index is %d, computed linear index should be %d, literal computed linear index is %d\n"
        //     , i, j, k, IDX3D(i, j, k-1, Nx, Ny, Nz) + ivf * cube_size, cube_size, ivf, IDX3D(i, j, k-1, Nx, Ny, Nz), (k - 1) * (Nx * Ny) + i * Ny + j, k - 1 * (Nx * Ny) + i * Ny + j);

        num_diff = D * (
            (1.0 / pow(dx, 2)) 
            * (
                fluid_var[IDX3D(i+1, j, k, Nx, Ny, Nz) + ivf * cube_size] 
                - 2.0*fluid_var[IDX3D(i, j, k, Nx, Ny, Nz) + ivf * cube_size] 
                + fluid_var[IDX3D(i-1, j, k, Nx, Ny, Nz) + ivf * cube_size]
                )
            + (1.0 / pow(dy, 2)) 
            * (
                fluid_var[IDX3D(i, j+1, k, Nx, Ny, Nz) + ivf * cube_size] 
                - 2.0*fluid_var[IDX3D(i, j, k, Nx, Ny, Nz) + ivf * cube_size] 
                + fluid_var[IDX3D(i, j-1, k, Nx, Ny, Nz) + ivf * cube_size]
                )
            + (1.0 / pow(dz, 2)) 
            * (
                fluid_var[IDX3D(i, j, k+1, Nx, Ny, Nz) + ivf * cube_size] 
                - 2.0*fluid_var[IDX3D(i, j, k, Nx, Ny, Nz) + ivf * cube_size] 
                + fluid_var[IDX3D(i, j, k-1, Nx, Ny, Nz) + ivf * cube_size]
                )
            );
        return num_diff;
    }

// Implements numerical diffusion on the front plane of the simulation grid (k = 0)
// Periodic boundary conditions are the reason
// 2nd-order central difference
// 19 AO, 9 MR
__device__ float numericalDiffusionFront(const int i, const int j, const float* fluid_var, 
    const float D, const float dx, const float dy, const float dz, 
    const int ivf, const int Nx, const int Ny, const int Nz)
    {   
        int cube_size = Nx * Ny * Nz;
        float num_diff = 0.0;
        num_diff = D * (
            (1.0 / pow(dx, 2)) 
                * (
                fluid_var[IDX3D(i+1, j, 0, Nx, Ny, Nz) + ivf * cube_size] 
                - 2.0*fluid_var[IDX3D(i, j, 0, Nx, Ny, Nz) + ivf * cube_size] 
                + fluid_var[IDX3D(i-1, j, 0, Nx, Ny, Nz) + ivf * cube_size]
                )
            + (1.0 / pow(dy, 2)) 
                * (
                fluid_var[IDX3D(i, j+1, 0, Nx, Ny, Nz) + ivf * cube_size] 
                - 2.0*fluid_var[IDX3D(i, j, 0, Nx, Ny, Nz) + ivf * cube_size] 
                + fluid_var[IDX3D(i, j-1, 0, Nx, Ny, Nz) + ivf * cube_size]
                )
            + (1.0 / pow(dz, 2)) 
                * (
                fluid_var[IDX3D(i, j, 1, Nx, Ny, Nz)  + ivf * cube_size] 
                - 2.0*fluid_var[IDX3D(i, j, 0, Nx, Ny, Nz) + ivf * cube_size] 
                + fluid_var[IDX3D(i, j, Nz - 2, Nx, Ny, Nz) + ivf * cube_size]
                )
            );
        return num_diff;
    }

// Implements numerical diffusion on the back plane of the simulation grid (k = Nz - 1)
// Periodic boundary conditions are the reason
// 2nd-order central difference
// 19 AO, 9 MR
__device__ float numericalDiffusionBack(const int i, const int j, const float* fluid_var, 
    const float D, const float dx, const float dy, const float dz, 
    const int ivf, const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        float num_diff = 0.0;
        num_diff = D * (
            (1.0 / pow(dx, 2)) 
            * (
                fluid_var[IDX3D(i+1, j, Nz - 1, Nx, Ny, Nz) + ivf * cube_size] 
                - 2.0*fluid_var[IDX3D(i, j, Nz - 1, Nx, Ny, Nz) + ivf * cube_size] 
                + fluid_var[IDX3D(i-1, j, Nz - 1, Nx, Ny, Nz) + ivf * cube_size]
                )
            + (1.0 / pow(dy, 2)) 
            * (
                fluid_var[IDX3D(i, j+1, Nz - 1, Nx, Ny, Nz) + ivf * cube_size] 
                - 2.0*fluid_var[IDX3D(i, j, Nz - 1, Nx, Ny, Nz) + ivf * cube_size] 
                + fluid_var[IDX3D(i, j-1, Nz - 1, Nx, Ny, Nz) + ivf * cube_size]
                )
            + (1.0 / pow(dz, 2)) 
            * (
                fluid_var[IDX3D(i, j, 1, Nx, Ny, Nz) + ivf * cube_size] 
                - 2.0*fluid_var[IDX3D(i, j, Nz - 1, Nx, Ny, Nz) + ivf * cube_size] 
                + fluid_var[IDX3D(i, j, Nz - 2, Nx, Ny, Nz) + ivf * cube_size])
            );
        return num_diff;
    }