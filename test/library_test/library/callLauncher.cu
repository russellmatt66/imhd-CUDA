#include "kernels.cuh"
#include "helloLauncher.cu"
#include <iostream>

int helloWrapper(long* exec_config, const long N){
    long num_threads_per_block_x = exec_config[0];
    long num_threads_per_block_y = exec_config[1];
    long num_threads_per_block_z = exec_config[2];
    long num_blocks_x = exec_config[3];
    long num_blocks_y = exec_config[4];
    long num_blocks_z = exec_config[5];

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