#ifndef INIT_OD_DECL
#define INIT_OD_DECL

#include "initialize_od.cuh"
#include "grid.cuh"

// row-major, column-minor
#define IDX3D(i, j, k, Nx, Ny, Nz) (k * (Nx * Ny) + i * Nx + j)

__global__ void InitialConditions(float* rho, float* rhov_x, float* rhov_y, float* rhov_z, float* B_x, float* B_y, float* B_z, float* e,
    const int Nx, const int Ny, const int Nz, const int ICs_flag)
    {
        int tidx = threadIdx.x + blockDim.x * blockIdx.x;
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;
        int xthreads = gridDim.x * blockDim.x;
        int ythreads = gridDim.y * blockDim.y;
        int zthreads = gridDim.z * blockDim.z;

        if (ICs_flag == 1){ // screw-pinch
            /* TODO: implement screw-pinch ICs */
            for (int i = tidx; i < Nx; i += xthreads){
                for (int j = tidy; j < Ny; j += ythreads){
                    for (int k = tidz; k < Nz; k += zthreads){
                        /* Screw-pinch ICs */
                    }
                }
            }
        }
        return;
    }
#endif