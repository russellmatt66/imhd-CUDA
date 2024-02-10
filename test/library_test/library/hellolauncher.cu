#include <iostream>
#include "kernels.cuh"

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
    // for (unsigned long i = tidx; i < N; i += xthreads){
    //     for (unsigned long j = tidy; j < N; j += ythreads){
    //         for (unsigned long k = tidz; k < N; k += zthreads){
    //             // std::cout << "Hello world! From thread " << tnum << std::endl;
    //             // std::cout << "I am in block " << bnum << std::endl;
    //             // std::cout << "My number here is " << tnum_b << std::endl; 
    //             printf("Hello world! From thread %lu, in block %lu, where my number is %lu\n", tnum, bnum, tnum_b);
    //         }
    //     }
    // }
    return;
}

// One danger is too many threads will lead to overflow 
// __global__ void HelloLauncher(const int N){ // N = total number of threads
//     int tidx = threadIdx.x + blockDim.x * blockIdx.x;
//     int tidy = threadIdx.y + blockDim.y * blockIdx.y;
//     int tidz = threadIdx.z + blockDim.z * blockIdx.z;
//     int xthreads = blockDim.x * gridDim.x;
//     int ythreads = blockDim.y * gridDim.y;
//     int zthreads = blockDim.z * gridDim.z;

//     int num_threads_per_block = blockDim.x * blockDim.y * blockDim.z;

//     int tnum_b = threadIdx.x 
//         + threadIdx.y * blockDim.x 
//         + threadIdx.z * blockDim.x * blockDim.y;

//     int bnum = blockIdx.x 
//         + blockIdx.y * gridDim.x
//         + blockIdx.z * gridDim.x * gridDim.y; 

//     int tnum = tnum_b
//         + blockIdx.x * num_threads_per_block
//         + blockIdx.y * num_threads_per_block * gridDim.x
//         + blockIdx.z * num_threads_per_block * gridDim.x * gridDim.y; 

//     for (int i = tidx; i < N; i += xthreads){
//         for (int j = tidy; j < N; j += ythreads){
//             for (int k = tidz; k < N; k += zthreads){
//                 std::cout << "Hello world! From thread " << tnum << std::endl;
//                 std::cout << "I am in block " << bnum << std::endl;
//                 std::cout << "My number here is " << tnum_b << std::endl; 
//             }
//         }
//     }
// }