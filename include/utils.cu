#include "utils.cuh"

#include <stdio.h>

/* 
USE DEBUGGER 
While it's not known to me EXACTLY why device code like this will sometimes not print the specified information, my understanding is that there are times when
the buffer to stdout gets flushed, and due to the asynchronous nature of execution on a GPU, when the time finally comes to transmit the buffer to stdout, it 
has been previously flushed, and the specified data does not get sent.
*/
__global__ void PrintGrid(const float* x_grid, const float* y_grid, const float* z_grid, const size_t Nx, const size_t Ny, const size_t Nz){
    size_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t tidy = threadIdx.y + blockDim.y * blockIdx.y;
    size_t tidz = threadIdx.z + blockDim.z * blockIdx.z;

    size_t xthreads = blockDim.x * gridDim.x;
    size_t ythreads = blockDim.y * gridDim.y;
    size_t zthreads = blockDim.z * gridDim.z;

    for (size_t k = tidz; k < Nz; k += zthreads){
        for (size_t i = tidx; i < Nx; i += xthreads){
            for (size_t j = tidy; j < Ny; j += ythreads){
                printf("(x%d, y%d, z%d) = (%f, %f, %f)\n", i, j, k, x_grid[i], y_grid[j], z_grid[k]);
            }
        }
    }
    return;
}