#ifndef HELPER_CUH
#define HELPER_CUH

__device__ float B_sq_local(const float Bx, const float By, const float Bz);
__device__ float p_local(const float e, const float Bsq, const float KE);
__device__ float KE_local(const float rho, const float rhovx, const float rhovy, const float rhovz);
__device__ float B_dot_u_local(const float rho, const float rhovx, const float rhovz, const float Bx, const float By, const float Bz);

// These thrash the cache
__device__ float B_sq(int i, int j, int k, const float* fluidvar, const int Nx, const int Ny, const int Nz); // B / \sqrt{\mu_{0}} -> B 
__device__ float p(int i, int j, int k, const float* fluidvar, const float B_sq, const float KE, const int Nx, const int Ny, const int Nz);
__device__ float KE(int i, int j, int k, const float* fluidvar, const int Nx, const int Ny, const int Nz); // \rho * \vec{u}\cdot\vec{u} * 0.5
__device__ float B_dot_u(int i, int j, int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);

#endif