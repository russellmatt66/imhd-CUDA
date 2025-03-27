#ifndef FLUIDBCS_CUH
#define FLUIDBCS_CUH

struct BoundaryConfig {

};

__global__ void BoundaryConditions(float* fluidvar, const float* intvar, 
    const float D, const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void BoundaryConditionsNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void rigidConductingWallBCsLeftRight(float* fluidvar, const int Nx, const int Ny, const int Nz);

__global__ void rigidConductingWallBCsTopBottom(float* fluidvar, const int Nx, const int Ny, const int Nz);

__global__ void PBCs(float* fluidvar, const int Nx, const int Ny, const int Nz);

#endif