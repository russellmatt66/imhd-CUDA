#ifndef FLUIDBCS_CUH
#define FLUIDBCS_CUH

struct BoundaryConfig { // variables need to launch BC kernels   
    dim3 egd_frontback, tbd_frontback; /* What geometric plane does this correspond to? */
    float dt, dx, dy, dz;
    int Nx, Ny, Nz;
};

__global__ void rigidConductingWallBCsLeftRight(float* fluidvar, const int Nx, const int Ny, const int Nz);

__global__ void rigidConductingWallBCsTopBottom(float* fluidvar, const int Nx, const int Ny, const int Nz);

__global__ void PBCsInZ(float* fluidvar, const int Nx, const int Ny, const int Nz);

void LaunchFluidBCsPBCZ(float* fluidvars, const int Nx, const int Ny, const int Nz, BoundaryConfig& bcfg); 

__global__ void BoundaryConditions(float* fluidvar, const float* intvar, 
    const float D, const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void BoundaryConditionsNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

#endif