#ifndef INIT_OD_DECL
#define INIT_OD_DECL

#include "initialize_od.cuh"

// row-major, column-minor
#define IDX3D(i, j, k, Nx, Ny, Nz) (k * (Nx * Ny) + i * Nx + j)
  
// Linear arrays are the best kind of array to use on a GPU
__global__ void InitialConditions(float* rho, float* rhov_x, float* rhov_y, float* rhov_z, float* B_x, float* B_y, float* B_z, float* e,
    const float* grid_x, const float* grid_y, const float* grid_z,
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
            for (int k = tidz; k < Nz; k += zthreads){ // THIS LOOP ORDER IMPLEMENTS CONTIGUOUS MEMORY ACCESSES
                for (int i = tidx; i < Nx; i += xthreads){
                    for (int j = tidy; j < Ny; j += ythreads){
                        /* Screw-pinch ICs */
                    }
                }
            }
        }
        return;
    }
#endif