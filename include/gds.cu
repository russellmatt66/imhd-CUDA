#include <stdio.h>
#include <fcntl.h>
#include <cuda_runtime_api.h>
#include <cufile.h>

#include "gds.cuh"
#include "cufile_sample_utils.h"

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

void writeGridGDS(const char* filename, const float* x_grid, const float* y_grid, const float* z_grid, const int Nx, const int Ny, const int Nz){
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

    int bufferSize_X = sizeof(float) * Nx; 
    int bufferSize_Y = sizeof(float) * Ny;
    int bufferSize_Z = sizeof(float) * Nz;

    // const int bufferSize = sizeof(float) * 3;
    // float* data;
    // data = (float*)malloc(bufferSize);
    // cudaMalloc(&data, bufferSize);

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

    // for (int k = 0; k < Nz; k++){
    //     for (int i = 0; i < Nx; i++){
    //         for (int j = 0; j < Ny; j++){
    //             data[0] = x_grid[i];
    //             data[1] = y_grid[j];
    //             data[2] = z_grid[k];
    //             ret = cuFileWrite(cf_handle, data, bufferSize, 0, 0);
    //             if (status.err != CU_FILE_SUCCESS) {
    //                 // fprintf(stderr, "cuFile error: %s\n", cuFileGetErrorString(status));
    //                 std::cerr << "cuFile File write error: " << cuFileGetErrorString(status) << std::endl;
    //                 return;
    //             }
    //         }
    //     }
    // }

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