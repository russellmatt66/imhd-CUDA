/* 
Figure out how to write using GDS
(1) There's a segfault in here
(2) It would be useful to error check the cuFile API calls
*/
#include <stdio.h>
#include <fcntl.h>
#include <cuda_runtime_api.h>
#include <cufile.h>
#include <iostream>
#include <string>

__global__ void InitializeBuffer(float* buffer, const size_t size){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int xthreads = blockDim.x * gridDim.x;

    for (size_t i = tidx; i < size; i += xthreads){
        buffer[i] = 0.0;
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
    // if (status.err != CU_FILE_SUCCESS) {
    //     fprintf(stderr, "cuFile error: %s\n", status);
    //     return -11;
    // }

    const size_t bufferSize = 1024 * 1024; 
    float* gpuBuffer;
    cudaMalloc(&gpuBuffer, sizeof(float)*bufferSize);

    dim3 grid_dimensions(60,1,1);
    dim3 block_dimensions(1024,1,1);

    fd = open(filename, O_CREAT | O_RDWR | O_DIRECT, 0664);
    cf_descr.handle.fd = fd;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileBufRegister(gpuBuffer, bufferSize, 0);
	status = cuFileHandleRegister(&cf_handle, &cf_descr);
    // if (status.err != CU_FILE_SUCCESS) {
    //     fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
    //     return -11;
    // }

    std::cout << "Initializing buffer" << std::endl;
    InitializeBuffer<<<grid_dimensions, block_dimensions>>>(gpuBuffer, bufferSize);
    cudaDeviceSynchronize();
    std::cout << "Buffer initialized" << std::endl;

    // if (status.err != CU_FILE_SUCCESS) {
    //     fprintf(stderr, "cuFile error: %s\n", status);
    //     return -11;
    // }
    ret = cuFileWrite(cf_handle, gpuBuffer, bufferSize, 0, 0);

    // if (status.err != CU_FILE_SUCCESS) {
    //     fprintf(stderr, "cuFile error: %s\n", status);
    //     return -11;
    // }

    cudaFree(gpuBuffer);
    status = cuFileBufDeregister(gpuBuffer);
    cuFileHandleDeregister(cf_handle);
    status = cuFileDriverClose();
    return ret;
}