#include <stdio.h>

#include "initialize_od.cuh"

#define IDX3D(i, j, k, Nx, Ny, Nz) ((k) * (Nx * Ny) + (i) * Ny + j) // parentheses are necessary to avoid calculating `i - 1 * Ny` or `k - 1 * (Nx * Ny)`


// 16 registers / thread
__global__ void InitializeGrid(const float x_min, const float x_max, const float y_min, const float y_max, const float z_min, const float z_max, 
    const float dx, const float dy, const float dz, float* grid_x, float* grid_y, float* grid_z, 
    const int Nx, const int Ny, const int Nz)
    {
        int tidx = threadIdx.x + blockDim.x * blockIdx.x;
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;

        if (tidx < Nx && tidy < Ny && tidz < Nz){
            grid_z[tidz] = z_min + tidz * dz;
            grid_x[tidx] = x_min + tidx * dx;
            grid_y[tidy] = y_min + tidy * dy;
        }

        return;
    }

// 56 registers / thread
__global__ void InitialConditions(float* fluidvar, const float J0, const float* grid_x, const float* grid_y, const float* grid_z, 
    const int Nx, const int Ny, const int Nz)
    {
        int tidx = threadIdx.x + blockDim.x * blockIdx.x;
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;

        int xthreads = blockDim.x * gridDim.x;
        int ythreads = blockDim.y * gridDim.y;
        int zthreads = blockDim.z * gridDim.z;

        float r = 0.0;
        float r_pinch = 0.25 * sqrtf(pow(grid_x[Nx-1],2) + pow(grid_y[Ny-1],2)); // r_pinch = 0.25 * r_max 

        int cube_size = Nx * Ny * Nz;

        float p = 0.0;

        float C1 = 0.0; // Constants from equilibrium solution
        float C2 = 0.0;
        float C3 = 0.0;

        float Jr = 0.0; // Makes conversion from cylindrical coordinates more readable 
        float Jphi = 0.0;

        float Br = 0.0;
        float Bphi = 0.0;
        float B0 = 1.0;

        float x = 0.0; // More readable
        float y = 0.0;
        for (int k = tidz; k < Nz; k += zthreads){ // THIS LOOP ORDER IMPLEMENTS CONTIGUOUS MEMORY ACCESSES
            for (int i = tidx; i < Nx; i += xthreads){
                for (int j = tidy; j < Ny; j += ythreads){
                    x = grid_x[i];
                    y = grid_y[j];
                    r = sqrtf(pow(x, 2) + pow(y, 2));

                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = 0.0;
                    
                    if (r < r_pinch){
                        Br = C2 / r;
                        Bphi = 0.5 * J0 * r * (1.0 - 0.5 * pow(r, 2) / pow(r_pinch, 2) + 2.0 / pow(r_pinch, 2) * C1); 

                        p = C3 
                            + (0.5 + C1 * pow(r_pinch, -2)) * pow(r,2) 
                            - (3.0 / (8.0 * pow(r_pinch, 2)) + C1 / (2.0 * pow(r_pinch, 4))) * pow(r, 4) 
                            + (1.0 / (24.0 * pow(r_pinch, 4))) * pow(r,6);
                        
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0;
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = Jr * x - Jphi * y / r;
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = Jr * y + Jphi * x / r;
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = J0 * (1 - pow(r, 2) / pow(r_pinch, 2));
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = Br * x - Bphi * y / r;
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = Br * y + Bphi * x / r;
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = B0;
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = (p / (gamma - 1.0)) 
                                                                                + (pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size], 2) 
                                                                                    + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size], 2)
                                                                                    + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size], 2)
                                                                                    ) / (2.0 * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)])
                                                                                + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size], 2) 
                                                                                + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size], 2) 
                                                                                + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size], 2);                
                        // printf("For (%d, %d, %d) the value of the mhd energy is %5.4f, density: %5.4f, rhovz: %5.4f, Bx: %5.4f, By: %5.4f, Bz: %5.4f, gammacoeff: %f\n", 
                        //     i, j, k, 
                        //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size], fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size], 
                        //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size], fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size],
                        //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size], fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size],
                        //     (1.0 / (gamma - 1.0)));
                    }
                }
            }
        }
        return;
    }

// 15 registers / thread
__global__ void InitializeIntAndSwap(float* fluidvar_np1, float* intvar, const int Nx, const int Ny, const int Nz)
{
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
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.0;
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0;
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0;
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0;
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = 0.0; 

                    fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.0;
                    fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0;
                    fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
                    fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
                    fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0;
                    fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0;
                    fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
                    fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = 0.0; 
            }
        }
    }
    return;
}