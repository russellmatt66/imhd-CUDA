#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <iostream>
// WIP
__global__ void HelloLauncher(int* int_ptr, const int N){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    int tidz = threadIdx.z + blockDim.z * blockIdx.z;
    int xthreads = blockDim.x * gridDim.x;
    int ythreads = blockDim.y * gridDim.y;
    int zthreads = blockDim.z * gridDim.z;

    int num_threads_per_block = blockDim.x * blockDim.y * blockDim.z;

    int tnum_b = threadIdx.x 
        + threadIdx.y * blockDim.x 
        + threadIdx.z * blockDim.x * blockDim.y;

    int bnum = blockIdx.x 
        + blockIdx.y * gridDim.x
        + blockIdx.z * gridDim.x * gridDim.y; 

    int tnum = tnum_b
        + blockIdx.x * num_threads_per_block
        + blockIdx.y * num_threads_per_block * gridDim.x
        + blockIdx.z * num_threads_per_block * gridDim.x * gridDim.y; 

    for (int i = tidx; i < N; i += xthreads){
        for (int j = tidy; j < N; j += ythreads){
            for (int k = tidz; k < N; k += zthreads){
                std::cout << "Hello world! From thread " << tnum << std::endl;
                std::cout << "I am in block " << bnum << std::endl;
                std::cout << "My number here is " << tnum_b << std::endl; 
            }
        }
    }
}
#endif