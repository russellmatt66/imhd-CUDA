/* 
WIP 
Test the Initial Conditions, and look at them
*/
#include <stdio.h>
#include <cuda_runtime.h>
#include <cufile.h>

#include "../include/initialize_od.cuh"


int main(int argc, char* argv[]){
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed, error: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    return 0;
}