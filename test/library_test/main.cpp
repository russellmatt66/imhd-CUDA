// Attempting to call CUDA code from inside a C++ file
#include <iostream>
#include <cuda_runtime.h>
#include "library/kernels.cuh"
#include "library/callLauncher.cu"

// extern "C" int helloWrapper(const long* exec_config, const long N);

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

    long* exec_config;
    exec_config = (long*)malloc(6*sizeof(unsigned long));

    int hello_status = 0;

    // Calls HelloLauncher 
    hello_status = helloWrapper(exec_config, N);

    free(exec_config);
    return 0;
}