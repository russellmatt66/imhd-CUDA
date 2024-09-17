// Attempting to call CUDA code from inside a C file
#include <stdio.h>
#include <cuda_runtime.h>
#include "library/kernels.cuh"

/* Write wrapper around CUDA kernel */
int main(int argc, char* argv[]){
    // int num_blocks_x = atoi(argv[1]);
    // int num_blocks_y = atoi(argv[2]);
    // int num_blocks_z = atoi(argv[3]);
    // int num_threads_per_block_x = atoi(argv[4]);
    // int num_threads_per_block_y = atoi(argv[5]);
    // int num_threads_per_block_z = atoi(argv[6]);

    // int N = num_blocks_x * num_blocks_y * num_blocks_z * num_threads_per_block_x * num_threads_per_block_y * num_threads_per_block_z;
    unsigned long num_blocks_x = atoll(argv[1]);
    unsigned long num_blocks_y = atoll(argv[2]);
    unsigned long num_blocks_z = atoll(argv[3]);
    unsigned long num_threads_per_block_x = atoll(argv[4]);
    unsigned long num_threads_per_block_y = atoll(argv[5]);
    unsigned long num_threads_per_block_z = atoll(argv[6]);

    unsigned long N = num_blocks_x * num_blocks_y * num_blocks_z * num_threads_per_block_x * num_threads_per_block_y * num_threads_per_block_z;

    /* TODO - Need to write a wrapper for the below */ 
    // dim3 block_dims(num_threads_per_block_x, num_threads_per_block_y, num_threads_per_block_z);
    // dim3 grid_dims(num_blocks_x, num_blocks_y, num_blocks_z);

    // HelloLauncher<<<grid_dims, block_dims>>>(N);

    // // Check for CUDA errors
    // cudaDeviceSynchronize();
    // cudaError_t cudaError = cudaGetLastError();
    // if (cudaError != cudaSuccess) {
    //     std::cerr << "CUDA error: " << cudaGetErrorString(cudaError) << std::endl;
    //     return 1;
    // }
    return 0;
}