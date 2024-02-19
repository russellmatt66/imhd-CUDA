#include "grid.cuh"

#include <stdio.h>

// Uniform cartesian grid
/* Should this be a 1D execution configuration? */
__global__ void InitializeGrid(Grid3D* grid, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max, 
    const int Nx, const int Ny, const int Nz)
    {
        int tidx = threadIdx.x + blockDim.x * blockIdx.x;
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;
        int xthreads = gridDim.x * blockDim.x;
        int ythreads = gridDim.y * blockDim.y;
        int zthreads = gridDim.z * blockDim.z;
        /* TODO: implement uniform cartesian grid */
        float dx = (x_max - x_min) / (Nx - 1);
        float dy = (y_max - y_min) / (Ny - 1);
        float dz = (z_max - z_min) / (Nz - 1);

        /* Uniform cartesian grid */
        for (int i = tidx; i < Nx; i += xthreads){
            grid->x_[i] = x_min + i * dx;
        } 
        
        for (int j = tidy; j < Ny; j += ythreads){
            grid->y_[j] = y_min + j * dy;
        }
        
        for (int k = tidz; k < Nz; k += zthreads){
            grid->z_[k] = z_min + k * dz;
        }
        return;
} 

__global__ void PrintGrid(Grid3D* grid, const int Nx, const int Ny, const int Nz){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    int tidz = threadIdx.z + blockDim.z * blockIdx.z;
    int xthreads = gridDim.x * blockDim.x;
    int ythreads = gridDim.y * blockDim.y;
    int zthreads = gridDim.z * blockDim.z;

    printf("Inside PrintGrid()");
    for (int i = tidx; i < Nx; i += xthreads){
        for (int j = tidy; j < Ny; j += ythreads){
            for (int k = tidz; k < Nz; k += zthreads){
                printf("The point at (i, j, k) = (%d, %d, %d) is (%f, %f, %f)\n", i, j, k, grid->x_[i], grid->y_[j], grid->z_[k]);
            }
        }
    }
    return;
}