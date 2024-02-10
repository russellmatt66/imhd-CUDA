#include <iostream>
#include <cuda_runtime.h>
#include "library/kernels.cuh"

int main(int argc, char* argv[]){
    unsigned long num_blocks_x = atoll(argv[1]);
    unsigned long num_blocks_y = atoll(argv[2]);
    unsigned long num_blocks_z = atoll(argv[3]);
    unsigned long num_threads_per_block_x = atoll(argv[4]);
    unsigned long num_threads_per_block_y = atoll(argv[5]);
    unsigned long num_threads_per_block_z = atoll(argv[6]);

    unsigned long N = num_blocks_x * num_blocks_y * num_blocks_z * num_threads_per_block_x * num_threads_per_block_y * num_threads_per_block_z;

    // Need to write a wrapper for the below
    dim3 block_dims(num_threads_per_block_x, num_threads_per_block_y, num_threads_per_block_z);
    dim3 grid_dims(num_blocks_x, num_blocks_y, num_blocks_z);

    HelloLauncher<<<grid_dims, block_dims>>>(N);

    // Check for CUDA errors
    cudaDeviceSynchronize();
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaError) << std::endl;
        return 1;
    }
    return 0;
}