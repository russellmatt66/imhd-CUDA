#ifndef INIT_OD_DECL
#define INIT_OD_DECL

#include "initialize_od.cuh"

// row-major, column-minor
#define IDX3D(i, j, k, Nx, Ny, Nz) (k * (Nx * Ny) + i * Nx + j)

// Linear arrays are the best kind of array to use on a GPU
__global__ void InitialConditions(float* rho, float* rhov_x, float* rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e,
    const float J0, const float* grid_x, const float* grid_y, const float* grid_z,
    const int Nx, const int Ny, const int Nz)
    {
        int tidx = threadIdx.x + blockDim.x * blockIdx.x;
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;
        int xthreads = gridDim.x * blockDim.x;
        int ythreads = gridDim.y * blockDim.y;
        int zthreads = gridDim.z * blockDim.z;

        float r = 0.0;
        float r_pinch = 0.25 * sqrtf(pow(grid_x[Nx-1],2) + pow(grid_y[Ny-1],2)); // r_pinch = 0.25 * r_max 
        /* TODO: implement screw-pinch ICs */ 
        for (int k = tidz; k < Nz; k += zthreads){ // THIS LOOP ORDER IMPLEMENTS CONTIGUOUS MEMORY ACCESSES
            for (int i = tidx; i < Nx; i += xthreads){
                for (int j = tidy; j < Ny; j += ythreads){
                    r = sqrtf(pow(grid_x[i], 2) + pow(grid_y[j], 2) + pow(grid_z[k], 2));
                    /* Screw-pinch ICs */
                    rho[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.0;
                    rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.0;
                    rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.0;
                    rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.0;
                    Bx[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.0;
                    By[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.0;
                    Bz[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.0;
                    e[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.0;
                    if (r < r_pinch){
                        rho[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0;
                        rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.0;
                        rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.0;
                        rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)] = J0 / (q_e / m) * (1.0 - pow(r, 2) / pow(r_pinch, 2));
                        Bx[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.0;
                        By[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.0;
                        Bz[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0;
                        e[IDX3D(i, j, k, Nx, Ny, Nz)] = (1.0 / (gamma - 1.0)) 
                                                        + pow(rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)], 2) / (2.0 * rho[IDX3D(i, j, k, Nx, Ny, Nz)])
                                                        + pow(Bx[IDX3D(i, j, k, Nx, Ny, Nz)], 2) + pow(By[IDX3D(i, j, k, Nx, Ny, Nz)], 2) 
                                                        + pow(Bz[IDX3D(i, j, k, Nx, Ny, Nz)], 2);
                    }

                }
            }
        }
        return;
    }
#endif