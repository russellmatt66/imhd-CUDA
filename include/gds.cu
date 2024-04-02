#include <stdio.h>
#include <fcntl.h>
#include <cuda_runtime_api.h>
#include <cufile.h>

#include "gds.cuh"
#include "cufile_sample_utils.h"

#define IDX3D(i, j, k, Nx, Ny, Nz) (k * (Nx * Ny) + i * Ny + j) // row-major, column-minor order

// data looks like d000 d010 d020 ... d0,Ny-1,0 d100 d110 ... dNx-1,Ny-1,0 d001 d011 ... dNx-1,Ny-1,Nz-1
// row-major, column-minor order
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
    std::cout << "Fluid data successfully written out" << std::endl;
    return;
}

// write x0 x1 ... xN-1 y0 y1 ... yN-1 z0 z1 ... zN-1
void writeGridBasisGDS(const char* filename, const float* x_grid, const float* y_grid, const float* z_grid, const int Nx, const int Ny, const int Nz){
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

    fd = open(filename, O_CREAT | O_WRONLY, 0644);
	memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = open(filename, O_WRONLY);
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    status = cuFileHandleRegister(&cf_handle, &cf_descr);
        if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "cuFile Handle registration error: " << cuFileGetErrorString(status) << std::endl;
        return;
    }

    int bufferSize_X = sizeof(float) * Nx; 
    int bufferSize_Y = sizeof(float) * Ny;
    int bufferSize_Z = sizeof(float) * Nz;

    status = cuFileBufRegister(x_grid, bufferSize_X, 0);
    if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "cuFile Buffer registration error: " << cuFileGetErrorString(status) << std::endl;
        return;
    }

    ret = cuFileWrite(cf_handle, x_grid, bufferSize_X, 0, 0);
    if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "cuFile File write error: " << cuFileGetErrorString(status) << std::endl;
        return;
    }
    
    status = cuFileBufRegister(y_grid, bufferSize_Y, 0);
    if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "cuFile Buffer registration error: " << cuFileGetErrorString(status) << std::endl;
        return;
    }

    ret = cuFileWrite(cf_handle, y_grid, bufferSize_Y, 0, 0);
    if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "cuFile File write error: " << cuFileGetErrorString(status) << std::endl;
        return;
    }


    status = cuFileBufRegister(z_grid, bufferSize_Z, 0);
    if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "cuFile Buffer registration error: " << cuFileGetErrorString(status) << std::endl;
        return;
    }

    ret = cuFileWrite(cf_handle, z_grid, bufferSize_Z, 0, 0);
    if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "cuFile File write error: " << cuFileGetErrorString(status) << std::endl;
        return;
    }

    cuFileBufDeregister(x_grid);
    cuFileBufDeregister(y_grid);
    cuFileBufDeregister(z_grid);
    cuFileHandleDeregister(cf_handle);

    status = cuFileDriverClose();
    if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "CUDA Driver close error: " << cuFileGetErrorString(status) << std::endl;
        return;
    }
    return;
}

// Write x0 y0 z0 x0 y1 z0 x0 y2 z0 ... x1 y0 z0 x1 y1 z0 ... xN-1 yN-1 z0 x0 y0 z1 x0 y1 z1 ... xN-1 yN-1 zN-1
void writeGridGDS(const char* filename, const float* grid_data, const int Nx, const int Ny, const int Nz){
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

    fd = open(filename, O_CREAT | O_WRONLY, 0644);
	memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = open(filename, O_WRONLY);
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    status = cuFileHandleRegister(&cf_handle, &cf_descr);
        if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "cuFile Handle registration error: " << cuFileGetErrorString(status) << std::endl;
        return;
    }

    // Create data buffer
    const int bufferSize = sizeof(float) * Nx * Ny * Nz * 3;
    // float* data;
    // data = (float*)malloc(bufferSize);
    // cudaMalloc(&data, bufferSize);

    status = cuFileBufRegister(grid_data, bufferSize, 0);
    if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "cuFile Buffer registration error: " << cuFileGetErrorString(status) << std::endl;
        return;
    }

    ret = cuFileWrite(cf_handle, grid_data, bufferSize, 0, 0);
    if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "cuFile File write error: " << cuFileGetErrorString(status) << std::endl;
        return;
    }

    cuFileBufDeregister(grid_data);
    cuFileHandleDeregister(cf_handle);

    status = cuFileDriverClose();
    if (status.err != CU_FILE_SUCCESS) {
        // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
        std::cerr << "CUDA Driver close error: " << cuFileGetErrorString(status) << std::endl;
        return;
    }
    std::cout << "Grid data successfully written out" << std::endl;
    return;
}

// buffer looks like: x0 y0 z0 x0 y1 z0 ... x0 yN-1 z0 x1 y0 z0 x1 y1 z0 ... xN-1 yN-1 z0 x0 y0 z1 x0 y1 x1 ... xN-1 yN-1 zN-1 
// row-major, column-minor order
__global__ void WriteGridBuffer(float* buffer, const float* x_grid, const float* y_grid, const float* z_grid, const int Nx, const int Ny, const int Nz){
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    int tidz = threadIdx.z + blockIdx.z * blockDim.z;

    int xthreads = gridDim.x * blockDim.x;
    int ythreads = gridDim.y * blockDim.y;
    int zthreads = gridDim.z * blockDim.z;

    // buffer looks like: x0 y0 z0 x0 y1 z0 ... x0 yN-1 z0 x1 y0 z0 x1 y1 z0 ... xN-1 yN-1 z0 x0 y0 z1 x0 y1 x1 ... xN-1 yN-1 zN-1 
    for (int k = tidz; k < Nz; k += zthreads){
        for (int i = tidx; i < Nx; i += xthreads){
            for (int j = tidy; j < Ny; j += ythreads){
                buffer[IDX3D(i, j, k, Nx, Ny, Nz)] = x_grid[i];
                buffer[IDX3D(i, j, k, Nx, Ny, Nz) + 1] = y_grid[j];
                buffer[IDX3D(i, j, k, Nx, Ny, Nz) + 2] = z_grid[k]; 
            }
        }
    }
    
    return;
}


