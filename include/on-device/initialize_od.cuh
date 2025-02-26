#ifndef INIT_OD_CUH
#define INIT_OD_CUH

// physical constants
#define gamma (5.0 / 3.0)

#include "utils.cuh" // Contains `InitConfig` class

// #include <string>
// #include <map>
// #include <functional>
// #include <stdexcept>

__global__ void InitializeGrid(const float x_min, const float x_max, const float y_min, const float y_max, const float z_min, const float z_max, 
    const float dx, const float dy, const float dz, float* grid_x, float* grid_y, float* grid_z, 
    const int Nx, const int Ny, const int Nz);

__global__ void InitializeX(float* grid_x, const float x_min, const float dx, const int Nx);
__global__ void InitializeY(float* grid_y, const float y_min, const float dy, const int Ny);
__global__ void InitializeZ(float* grid_z, const float z_min, const float dz, const int Nz);

/* I don't need these */
__global__ void ZeroVars(float* vars, const int Nx, const int Ny, const int Nz);
__global__ void ZeroVarsStride(float* vars, const int Nx, const int Ny, const int Nz);

// Data structure for holding the initialization parameters
struct InitConfig {
    dim3 gridDim;
    dim3 blockDim;
    float J0; // amplitude of current
    float r_max_coeff; // r_pinch = r_max * r_max_coeff
    float *x_grid, *y_grid, *z_grid; // pointers to grid data 
    int Nx, Ny, Nz; // dimensions of grid
 };

__global__ void ScrewPinch(float* fluidvar, 
    const float J0, const float r_max_coeff, 
    const float* grid_x, const float* grid_y, const float* grid_z,
    const int Nx, const int Ny, const int Nz);

// void LaunchScrewPinch(float *fluidvar, const InitConfig& cfg, const dim3 gridDim, const dim3 blockDim); // Wrapper for launching ScrewPinch kernel
void LaunchScrewPinch(float *fluidvar, const InitConfig& cfg); // Wrapper for launching ScrewPinch kernel

__global__ void ScrewPinchStride(float* fluidvar, const float J0, const float* grid_x, const float* grid_y, const float* grid_z, 
    const int Nx, const int Ny, const int Nz);

// void LaunchScrewPinchStride(float *fluidvar, const InitConfig& cfg, const dim3 gridDim, const dim3 blockDim); // Wrapper for launching ScrewPinchStride kernel
void LaunchScrewPinchStride(float *fluidvar, const InitConfig& cfg); // Wrapper for launching ScrewPinchStride kernel

__global__ void ZPinch(float* fluidvar, const float Btheta_a, const float* grid_x, const float* grid_y, const float* grid_z,
    const int Nx, const int Ny, const int Nz); 

#endif