#include "utils.cuh"
#include "helper_functions.cuh"
#include "diffusion.cuh"
#include "kernels_od.cuh"
#include "kernels_od_fluxes.cuh"

#include <stdio.h>

#define IDX3D(i, j, k, Nx, Ny, Nz) (k) * (Nx * Ny) + (i) * (Ny) + j 

// Print values of intermediate variables
__global__ void PrintIntvar(const float* intvar, const float* fluidvar, const size_t Nx, const size_t Ny, const size_t Nz){
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
                float Bsq = B_sq(i, j, k, fluidvar, Nx, Ny, Nz);
                float ke = KE(i, j, k, fluidvar, Nx, Ny, Nz);
                float pressure = p(i, j, k, fluidvar, Bsq, ke, Nx, Ny, Nz);
                printf("Printing intvar values. "
                    "For (%d, %d, %d), the value of pressure is: %5.4f, Bsq: %5.4f, ke: %5.4f, " 
                    "int. rho: %5.4f, int. rhovx: %5.4f, int. rhovy: %5.4f, int. rhovz: %5.4f, " 
                    "int. Bx: %5.4f, int. By: %5.4f, int. Bz: %5.4f, int. energy: %5.4f, "
                    "gamma: %f\n", 
                    i, j, k, pressure, Bsq, ke, 
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz)], intvar[IDX3D(i, j, k, Nx, Ny, Nz) +  cube_size], 
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size], intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size],  
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size], intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size],  
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size], intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size], 
                    gamma);
            }
        }
    }
    return;
}

// Print values of fluid variables
__global__ void PrintFluidvar(const float* fluidvar, const size_t Nx, const size_t Ny, const size_t Nz){
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
                float Bsq = B_sq(i, j, k, fluidvar, Nx, Ny, Nz);
                float ke = KE(i, j, k, fluidvar, Nx, Ny, Nz);
                float pressure = p(i, j, k, fluidvar, Bsq, ke, Nx, Ny, Nz);
                printf("Printing fluidvar values. " 
                    "For (%d, %d, %d), the value of pressure is: %5.4f, Bsq: %5.4f, ke: %5.4f, " 
                    "rho: %5.4f, rhovx: %5.4f, rhovy: %5.4f, rhovz: %5.4f, " 
                    "Bx: %5.4f, By: %5.4f, Bz: %5.4f, energy: %5.4f, "
                    "gamma: %f\n", 
                    i, j, k, pressure, Bsq, ke, 
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)], fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) +  cube_size], 
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size], fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size],  
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size], fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size],  
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size], fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size], 
                    gamma);
            }
        }
    }   
    return;
}