#ifndef KERNELS_OD_FLUXES_CUH
#define KERNELS_OD_FLUXES_CUH

// These do not thrash the cache, and are more readable
__device__ float XFluxRho(const float rhovx);
__device__ float XFluxRhoVx(const float rho, const float rhovx, const float Bx, const float p, const float Bsq);
__device__ float XFluxRhoVy(const float rho, const float rhovx, const float rhovy, const float Bx, const float By);
__device__ float XFluxRhoVz(const float rho, const float rhovx, const float rhovz, const float Bx, const float Bz);
__device__ float XFluxBx();
__device__ float XFluxBy(const float rho, const float rhovx, const float rhovy, const float Bx, const float By);
__device__ float XFluxBz(const float rho, const float rhovx, const float rhovz, const float Bx, const float Bz);
__device__ float XFluxE(const float rho, const float rhovx, const float Bx, const float e, const float p, const float Bsq, const float Bdotu);

// Y-Fluxes
__device__ float YFluxRho(const float rhovy);
__device__ float YFluxRhoVx(const float rho, const float rhovx, const float rhovy, const float Bx, const float By);
__device__ float YFluxRhoVy(const float rho, const float rhovy, const float By, const float p, const float Bsq);
__device__ float YFluxRhoVz(const float rho, const float rhovy, const float rhovz, const float By, const float Bz);
__device__ float YFluxBx(const float rho, const float rhovx, const float rhovy, const float Bx, const float By);
__device__ float YFluxBy();
__device__ float YFluxBz(const float rho, const float rhovy, const float rhovz, const float By, const float Bz);
__device__ float YFluxE(const float rho, const float rhovy, const float By, const float e, const float p, const float Bsq, const float Bdotu);

// Z-Fluxes
__device__ float ZFluxRho(const float rhovz);
__device__ float ZFluxRhoVx(const float rho, const float rhovx, const float rhovz, const float Bx, const float Bz);
__device__ float ZFluxRhoVy(const float rho, const float rhovy, const float rhovz, const float By, const float Bz);
__device__ float ZFluxRhoVz(const float rho, const float rhovz, const float Bz, const float p, const float Bsq);
__device__ float ZFluxBx(const float rho, const float rhovx, const float rhovz, const float Bx, const float Bz);
__device__ float ZFluxBy(const float rho, const float rhovy, const float rhovz, const float By, const float Bz);
__device__ float ZFluxBz();
__device__ float ZFluxE(const float rho, const float rhovz, const float Bz, const float e, const float p, const float Bsq, const float Bdotu);

// These thrash the cache
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