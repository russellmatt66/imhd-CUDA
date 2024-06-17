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

        for (int k = tidz; k < Nz; k += zthreads){ // THIS LOOP ORDER IMPLEMENTS CONTIGUOUS MEMORY ACCESSES
            for (int i = tidx; i < Nx; i += xthreads){
                for (int j = tidy; j < Ny; j += ythreads){
                    
                    r = sqrtf(pow(grid_x[i], 2) + pow(grid_y[j], 2));
                    
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.0001;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = 0.0;        
                    
                    if (r < r_pinch){
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0;
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0;
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = J0 * (1.0 - pow(r, 2) / pow(r_pinch, 2));
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0;
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0;
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 1.0;
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = (1.0 / (gamma - 1.0)) 
                                                                                + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size], 2) / (2.0 * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size])
                                                                                + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size], 2) + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size], 2) 
                                                                                + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size], 2);  
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