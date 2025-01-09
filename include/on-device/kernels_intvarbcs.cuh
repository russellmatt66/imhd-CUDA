#ifndef INTVAR_BCS_CUH
#define INTVAR_BCS_CUH

__global__ void ComputeIntermediateVariablesBoundary(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz, const float D,
    const int Nx, const int Ny, const int Nz);

/* REST OF DEVICE KERNELS DECLARATIONS */

#endif