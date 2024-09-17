#ifndef HELLO_LAUNCHER_CU
#define HELLO_LAUNCHER_CU

#include <iostream>
#include "kernels.cuh"
#include <device_launch_parameters.h>

__global__ void HelloLauncher(const unsigned long N){ // N = total number of threads
    // unsigned long tidx = threadIdx.x + blockDim.x * blockIdx.x;
    // unsigned long tidy = threadIdx.y + blockDim.y * blockIdx.y;
    // unsigned long tidz = threadIdx.z + blockDim.z * blockIdx.z;
    // unsigned long xthreads = blockDim.x * gridDim.x;
    // unsigned long ythreads = blockDim.y * gridDim.y;
    // unsigned long zthreads = blockDim.z * gridDim.z;

    // I derived all these myself
    unsigned long num_threads_per_block = blockDim.x * blockDim.y * blockDim.z;

    unsigned long tnum_b = threadIdx.x 
        + threadIdx.y * blockDim.x 
        + threadIdx.z * blockDim.x * blockDim.y;

    unsigned long bnum = blockIdx.x 
        + blockIdx.y * gridDim.x
        + blockIdx.z * gridDim.x * gridDim.y; 

    unsigned long tnum = tnum_b
        + blockIdx.x * num_threads_per_block
        + blockIdx.y * num_threads_per_block * gridDim.x
        + blockIdx.z * num_threads_per_block * gridDim.x * gridDim.y; 

    printf("Hello world! From thread %lu, in block %lu, where my number is %lu\n", tnum, bnum, tnum_b);
    return;
}
#endif