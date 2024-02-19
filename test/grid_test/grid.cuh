#ifndef GRID_CUH
#define GRID_CUH

#include <stdio.h>

#define IDX3D(i, j, k, Nx, Ny, Nz) (k * (Nx * Ny) + i * Nx + j)

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

typedef struct Grid3D{
    float *x_, *y_, *z_;
    int Nx_, Ny_, Nz_;

    Grid3D(int Nx, int Ny, int Nz) : Nx_(Nx), Ny_(Ny), Nz_(Nz) {
        checkCuda(cudaMalloc(&x_, Nx_ * sizeof(float)));
        checkCuda(cudaMalloc(&y_, Ny_ * sizeof(float)));
        checkCuda(cudaMalloc(&z_, Nz_ * sizeof(float)));
    }
};

void freeGrid3D(Grid3D* grid){
    cudaFree(grid->x_);
    cudaFree(grid->y_);
    cudaFree(grid->z_);
}

__global__ void InitializeGrid(Grid3D* grid, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max, 
    const int Nx, const int Ny, const int Nz);

__global__ void PrintGrid(Grid3D* grid, const int Nx, const int Ny, const int Nz);
#endif