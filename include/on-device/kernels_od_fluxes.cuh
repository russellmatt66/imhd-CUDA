#ifndef KERNELS_OD_FLUXES_CUH
#define KERNELS_OD_FLUXES_CUH

// Rho
__device__ float XFluxRho(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);
__device__ float YFluxRho(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);
__device__ float ZFluxRho(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);

// RhoVX
__device__ float XFluxRhoVX(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);
__device__ float YFluxRhoVX(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);
__device__ float ZFluxRhoVX(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);

// RhoVY
__device__ float XFluxRhoVY(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);
__device__ float YFluxRhoVY(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);
__device__ float ZFluxRhoVY(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);

// RhoVZ
__device__ float XFluxRhoVZ(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);
__device__ float YFluxRhoVZ(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);
__device__ float ZFluxRhoVZ(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);

// Bx
__device__ float XFluxBX(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);
__device__ float YFluxBX(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);
__device__ float ZFluxBX(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);

// By
__device__ float XFluxBY(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);
__device__ float YFluxBY(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);
__device__ float ZFluxBY(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);

// Bz
__device__ float XFluxBZ(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);
__device__ float YFluxBZ(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);
__device__ float ZFluxBZ(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);

// e
__device__ float XFluxE(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);
__device__ float YFluxE(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);
__device__ float ZFluxE(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);

#endif