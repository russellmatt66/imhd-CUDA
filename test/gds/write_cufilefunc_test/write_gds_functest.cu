/* 
WIP 
Test the Initial Conditions, and look at them
*/
#include <stdio.h>
#include <fcntl.h>
#include <cuda_runtime_api.h>
#include <cufile.h>

#include "cufile_sample_utils.h"

void writeDataGDS(const char* filename, const float* data, const int size);
__global__ void Kernel1(float* buffer, const size_t size);
__global__ void Kernel2(float* buffer, const size_t size);
__global__ void Kernel3(float* buffer, const size_t size);

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* ADD TIMING TO THIS */
int main(int argc, char* argv[]){
    const size_t N = 8192 * 8192; // >50,000,000 points of data which is what sim will create
    const size_t bufferSize = sizeof(float) * N;

    float *d_buf1, *d_buf2, *d_buf3;

    checkCuda(cudaMalloc(&d_buf1, bufferSize));
    checkCuda(cudaMalloc(&d_buf2, bufferSize));
    checkCuda(cudaMalloc(&d_buf3, bufferSize));
    
    dim3 grid_dimensions(60,1,1);
    dim3 block_dimensions(1024,1,1);

    Kernel1<<<grid_dimensions, block_dimensions>>>(d_buf1, N);
    Kernel2<<<grid_dimensions, block_dimensions>>>(d_buf2, N);
    Kernel3<<<grid_dimensions, block_dimensions>>>(d_buf3, N);
    checkCuda(cudaDeviceSynchronize());

    writeDataGDS("buf1.dat", d_buf1, N);
    writeDataGDS("buf2.dat", d_buf2, N);
    writeDataGDS("buf3.dat", d_buf3, N);

    cudaFree(d_buf1);
    cudaFree(d_buf2);
    cudaFree(d_buf3);
    return 0;
}

__global__ void Kernel1(float* buffer, const size_t size){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int xthreads = blockDim.x * gridDim.x;

    for (size_t i = tidx; i < size; i += xthreads){
        buffer[i] = 1.0;
        // printf("Value is: %f\n", buffer[i]);
    }

    return;
}

__global__ void Kernel2(float* buffer, const size_t size){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int xthreads = blockDim.x * gridDim.x;

    for (size_t i = tidx; i < size; i += xthreads){
        buffer[i] = 2.0;
        // printf("Value is: %f\n", buffer[i]);
    }

    return;
}

__global__ void Kernel3(float* buffer, const size_t size){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int xthreads = blockDim.x * gridDim.x;

    for (size_t i = tidx; i < size; i += xthreads){
        buffer[i] = 3.0;
        // printf("Value is: %f\n", buffer[i]);
    }

    return;
}
void writeDataGDS(const char* filename, const float* data, const int size){
    int fd = -1;
    ssize_t ret = -1;

    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed, error: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    CUfileError_t status;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;

    status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS){
        std::cerr << "cuFileDriverOpen failed, error: " << cuFileGetErrorString(status) << std::endl;
        return;
    }

    const int bufferSize = sizeof(float) * size;

    fd = open(filename, O_CREAT | O_WRONLY, 0644);
	memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = open(filename, O_WRONLY);
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    status = cuFileBufRegister(data, bufferSize, 0);
    if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "cuFile Buffer registration error: " << cuFileGetErrorString(status) << std::endl;
        return;
    }

    status = cuFileHandleRegister(&cf_handle, &cf_descr);
        if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "cuFile Handle registration error: " << cuFileGetErrorString(status) << std::endl;
        return;
    }

    ret = cuFileWrite(cf_handle, data, bufferSize, 0, 0);
    if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "cuFile File write error: " << cuFileGetErrorString(status) << std::endl;
        return;
    }

    cuFileBufDeregister(data);
    cuFileHandleDeregister(cf_handle);

    status = cuFileDriverClose();
    if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "CUDA Driver close error: " << cuFileGetErrorString(status) << std::endl;
        return;
    }
    return;
}