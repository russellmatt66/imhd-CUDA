/* 
Figure out how to write using GDS

Use this as a reference for how to write a function that writes data out 
*/
#include <stdio.h>
#include <fcntl.h>
#include <cuda_runtime_api.h>
#include <cufile.h>
#include <iostream>
#include <string>

#include "cufile_sample_utils.h"

__global__ void InitializeBuffer(float* buffer, const size_t size){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int xthreads = blockDim.x * gridDim.x;

    for (size_t i = tidx; i < size; i += xthreads){
        buffer[i] = 1.0;
        // printf("Value is: %f\n", buffer[i]);
    }

    return;
}

int main(int argc, char* argv[]){
    // Boilerplate in case of multiple GPUs
    int fd = -1;
    ssize_t ret = -1;

    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed, error: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    const char* filename = "ics.dat";
    CUfileError_t status;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;

    // std::cout << "Opening cuFile driver" << std::endl;
    status = cuFileDriverOpen();
    // std::cout << "cuFile driver opened" << std::endl;
    if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "CUDA Driver open error: " << cuFileGetErrorString(status) << std::endl;
        return -11;
    }

    const size_t bufferSize = sizeof(float) * 1024 * 1024; 
    const size_t N = 1024 * 1024;
    float* gpuBuffer;
    cudaMalloc(&gpuBuffer, bufferSize);

    fd = open(filename, O_CREAT | O_WRONLY, 0664);
	memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    
    status = cuFileBufRegister(gpuBuffer, bufferSize, 0);
    if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "cuFile Buffer registration error: " << cuFileGetErrorString(status) << std::endl;
        return -11;
    }

	status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "cuFile Handle registration error: " << cuFileGetErrorString(status) << std::endl;
        return -11;
    }

    dim3 grid_dimensions(60,1,1);
    dim3 block_dimensions(1024,1,1);
    std::cout << "Initializing buffer" << std::endl;
    InitializeBuffer<<<grid_dimensions, block_dimensions>>>(gpuBuffer, N);
    cudaDeviceSynchronize();
    std::cout << "Buffer initialized" << std::endl;

    ret = cuFileWrite(cf_handle, gpuBuffer, bufferSize, 0, 0);
    if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "cuFile File write error: " << cuFileGetErrorString(status) << std::endl;
        return -11;
    }

    cudaFree(gpuBuffer);
    cuFileBufDeregister(gpuBuffer);
    cuFileHandleDeregister(cf_handle);

    status = cuFileDriverClose();
    if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "CUDA Driver close error: " << cuFileGetErrorString(status) << std::endl;
        return -11;
    }
    return ret;
}