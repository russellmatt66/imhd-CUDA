#include "helper_functions.cuh"
#include "kernels_od.cuh"

/* THIS NEEDS TO BE DEFINED IN ONE PLACE */
#define IDX3D(i, j, k, Nx, Ny, Nz) ((k) * (Nx) * (Ny) + (i) * (Ny) + j) // parentheses are necessary to avoid calculating `i - 1 * Ny` or `k - 1 * (Nx * Ny)`

__device__ float B_sq_local(const float Bx, const float By, const float Bz)
{
    return pow(Bx, 2) + pow(By, 2) + pow(Bz, 2);
}

__device__ float p_local(const float e, const float Bsq, const float KE)
{
    return (gamma - 1.0) * (e - KE - Bsq / 2.0);
}

__device__ float KE_local(const float rho, const float rhovx, const float rhovy, const float rhovz)
{
    return (1.0 / rho) * (pow(rhovx, 2) + pow(rhovy, 2) + pow(rhovz, 2));
}

__device__ float B_dot_u_local(const float rho, const float rhovx, const float rhovy, const float rhovz, 
    const float Bx, const float By, const float Bz)
{
    return (1.0 / rho) * (rhovx * Bx + rhovy * By + rhovz * Bz);
}

// These thrash the cache
__device__ float B_sq(int i, int j, int k, const float* fluidvar, const int Nx, const int Ny, const int Nz) // B / \sqrt{\mu_{0}} -> B 
{
    int cube_size = Nx * Ny * Nz;
    return pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size], 2) 
        + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size], 2)
        + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size], 2);
}

__device__ float p(int i, int j, int k, const float* fluidvar, const float B_sq, const float KE, const int Nx, const int Ny, const int Nz)
{
    int cube_size = Nx * Ny * Nz;
    return (gamma - 1.0) * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] - KE - B_sq / 2.0);
}

__device__ float KE(int i, int j, int k, const float* fluidvar, const int Nx, const int Ny, const int Nz) // \rho * \vec{u}\cdot\vec{u} * 0.5
{
    int cube_size = Nx * Ny * Nz;
    float KE = (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * (
        pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size], 2) 
        + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size], 2) 
        + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size], 2));
    return KE;
}

__device__ float B_dot_u(int i, int j, int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    int cube_size = Nx * Ny * Nz;
    float B_dot_u = 0.0;
    B_dot_u = (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * (
        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size]
        + fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] 
        + fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size]);
    return B_dot_u;
}