#ifndef FLUIDBCS_CUH
#define FLUIDBCS_CUH

/*
TODO:
- Check alignment
*/
struct BoundaryConfig { // variables needed to launch BC kernels   
    dim3 egd_frontback, egd_leftright, egd_topbottom;
    dim3 tbd_frontback, tbd_leftright, tbd_topbottom; 
    float dt, dx, dy, dz;
    int Nx, Ny, Nz;
};

// Non-blocking launchers for the device kernels
// Periodic Boundary Conditions (PBCs)
void LaunchFluidBCsPBCX(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg); 
void LaunchFluidBCsPBCY(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg); 
void LaunchFluidBCsPBCZ(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg); 

// Perfectly-Conducting, Rigid, Wall (PCRW) kernels
// Launchers for more granular PCRW kernels
// Top = (0, j, k), Bottom = (Nx-1, j, k)
// Left = (i, 0, k), Right = (i, Ny-1, k)
// Front = (i, j, 0), Back = (i, j, Nz-1)
void LaunchFluidBCsPCRWTop(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg);
void LaunchFluidBCsPCRWBottom(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg);
void LaunchFluidBCsPCRWLeft(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg);
void LaunchFluidBCsPCRWRight(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg);
void LaunchFluidBCsPCRWFront(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg);
void LaunchFluidBCsPCRWBack(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg);

// Legacy PCRW kernel launchers 
// XY = "TopBottom" + "LeftRight"
// XZ = "TopBottom" + "FrontBack"
// YZ = "LeftRight" + "FrontBack"
void LaunchFluidBCsPCRWXY(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg);
void LaunchFluidBCsPCRWXZ(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg);
void LaunchFluidBCsPCRWYZ(float* fluidvars, const int Nx, const int Ny, const int Nz, const BoundaryConfig& bcfg);

// CUDA (__global__) Kernels
// Periodic Boundary COnditions (PBCs)
__global__ void PBCsInX(float* fluidvars, const int Nx, const int Ny, const int Nz);
__global__ void PBCsInY(float* fluidvars, const int Nx, const int Ny, const int Nz);
__global__ void PBCsInZ(float* fluidvars, const int Nx, const int Ny, const int Nz);

// Granular PCRW kernels
__global__ void PCRWFluidBCsLeft(float* fluidvars, const int Nx, const int Ny, const int Nz);
__global__ void PCRWFluidBCsRight(float* fluidvars, const int Nx, const int Ny, const int Nz);
__global__ void PCRWFluidBCsTop(float* fluidvars, const int Nx, const int Ny, const int Nz);
__global__ void PCRWFluidBCsBottom(float* fluidvars, const int Nx, const int Ny, const int Nz);
__global__ void PCRWFluidBCsFront(float* fluidvars, const int Nx, const int Ny, const int Nz);
__global__ void PCRWFluidBCsBack(float* fluidvars, const int Nx, const int Ny, const int Nz);

// Legacy PCRW kernels - lack granularity (therefore flexibility), but still in here as PoC
__global__ void rigidConductingWallBCsLeftRight(float* fluidvars, const int Nx, const int Ny, const int Nz);
__global__ void rigidConductingWallBCsTopBottom(float* fluidvars, const int Nx, const int Ny, const int Nz);
__global__ void rigidConductingWallBCsFrontBack(float* fluidvars, const int Nx, const int Ny, const int Nz);

#endif