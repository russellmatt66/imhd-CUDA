#ifndef FLUIDBCS_CUH
#define FLUIDBCS_CUH

struct BoundaryConfig { // variables needed to launch BC kernels   
    dim3 egd_frontback, egd_leftright, egd_topbottom;
    dim3 tbd_frontback, tbd_leftright, tbd_topbottom; 
    float dt, dx, dy, dz;
    int Nx, Ny, Nz;
};

// GPU Kernels
__global__ void rigidConductingWallBCsLeftRight(float* fluidvar, const int Nx, const int Ny, const int Nz);
__global__ void rigidConductingWallBCsTopBottom(float* fluidvar, const int Nx, const int Ny, const int Nz);
__global__ void rigidConductingWallBCsFrontBack(float* fluidvar, const int Nx, const int Ny, const int Nz);

__global__ void PBCsInX(float* fluidvars, const int Nx, const int Ny, const int Nz);
__global__ void PBCsInY(float* fluidvars, const int Nx, const int Ny, const int Nz);
__global__ void PBCsInZ(float* fluidvars, const int Nx, const int Ny, const int Nz);

// Non-blocking launchers for the device kernels
void LaunchFluidBCsPBCX(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg); 
void LaunchFluidBCsPBCY(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg); 
void LaunchFluidBCsPBCZ(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg); 

/* 
This needs more granularity w.r.t the face you specify
*/
void LaunchFluidBCsPCRWXY(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg);
void LaunchFluidBCsPCRWXZ(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg);
void LaunchFluidBCsPCRWYZ(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg);

#endif