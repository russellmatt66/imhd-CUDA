#ifndef INTVAR_BCS_CUH
#define INTVAR_BCS_CUH

// Megakernels that suffer from thread divergence
__global__ void ComputeIntermediateVariablesBoundaryNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void ComputeIntermediateVariablesBoundary(const float* fluidvar, float* intvar,
    const float D, const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

// Microkernels to eliminate the problem of thread divergence in the megakernels that make them a serious bottleneck
__global__ void QintBdryFrontNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void QintBdryPBCs(const float* fluidvar, float* intvar,
    const int Nx, const int Ny, const int Nz);

// Can deal with Left and Right Faces at the same time 
__global__ void QintBdryLeftRightNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz); 

// Can deal with Top and Bottom Faces at the same time
__global__ void QintBdryTopBottomNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz); 

__global__ void QintBdryFrontRightNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__global__ void QintBdryFrontBottomNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz); 

__global__ void QintBdryBottomRightNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz); 

// Device kernels that deal with the Right Face
// j = Ny-1
// i \in [0, Nx-1]
// k \in [1, Nz-2]
__device__ float intRhoRight(const int i, const int k, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVXRight(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVYRight(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVZRight(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBXRight(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBYRight(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBZRight(const int i, const int k, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intERight(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

// Device kernels that deal with the Bottom Face
// i = Nx-1
// j \in [1,Ny-1]
// k \in [1,Nz-2] - leave the front / back faces alone
__device__ float intRhoBottom(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVXBottom(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);
    
__device__ float intRhoVYBottom(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVZBottom(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBXBottom(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBYBottom(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBZBottom(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intEBottom(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

/* 
I THINK THE FOLLOWING IS UNNECESSARY
COMPUTING ANYTHING TO DO WITH THE Back Face IS POINTLESS 
*/
// Kernels that deal with the Back Face 
// k = Nz-1
// i \in [1, Nx-2]
// j \in [1, Nz-2]
__device__ float intRhoBack(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVXBack(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVYBack(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVZBack(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBXBack(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBYBack(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBZBack(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intEBack(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

// Device kernels that deal with the Front Right line
// k = 0
// i \in [0, Nx-2]
// j = Ny-1 
__device__ float intRhoFrontRight(const int i, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);
    
__device__ float intRhoVXFrontRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVYFrontRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVZFrontRight(const int i, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBXFrontRight(const int i, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBYFrontRight(const int i, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBZFrontRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intEFrontRight(const int i, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

// Device kernels that deal with the Front Bottom Line
// i = Nx-1
// j \in [0, Ny-2]
// k = 0
__device__ float intRhoFrontBottom(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVXFrontBottom(const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVYFrontBottom(const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVZFrontBottom(const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBXFrontBottom(const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBYFrontBottom(const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBZFrontBottom(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);
__device__ float intEFrontBottom(const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

// Device kernels that deal with the Bottom Right Line
// i = Nx-1
// j = Ny-1
// k \in [0, Nz-2]
__device__ float intRhoBottomRight(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVXBottomRight(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVYBottomRight(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVZBottomRight(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBXBottomRight(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBYBottomRight(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBZBottomRight(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intEBottomRight(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

/* 
I THINK THE FOLLOWING IS UNNECESSARY
COMPUTING ANYTHING TO DO WITH THE Back Face IS POINTLESS 
*/
// Device kernels that deal with the Back Bottom Line
// i = Nx-1
// j \in [0, Ny-2]
// k = Nz-1
__device__ float intRhoBackBottom(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVXBackBottom(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVYBackBottom(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);
__device__ float intRhoVZBackBottom(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBXBackBottom(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBYBackBottom(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);
__device__ float intBZBackBottom(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intEBackBottom(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

/* 
I THINK THE FOLLOWING IS UNNECESSARY
COMPUTING ANYTHING TO DO WITH THE Back Face IS POINTLESS 
*/
// Device kernels that deal with the Back Right line
// i \in [0, Nx-2]
// j = Ny-1
// k = Nz-1 
__device__ float intRhoBackRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVXBackRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVYBackRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intRhoVZBackRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBXBackRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBYBackRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intBZBackRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);

__device__ float intEBackRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz);
// I think that's everything! 

#endif