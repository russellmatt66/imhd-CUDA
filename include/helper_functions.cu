#include "helper_functions.cuh"

// Helper Functions
__device__ float B_sq(int i, int j, int k, const float* fluidvar, const int Nx, const int Ny, const int Nz) // B / \sqrt{\mu_{0}} -> B 
{
    /* IMPLEMENT B^2 */
}

__device__ float p(int i, int j, int k, const float* fluidvar, const float B_sq, const float KE, const int Nx, const int Ny, const int Nz)
{
    /* IMPLEMENT PRESSURE */
}

__device__ float KE(int i, int j, int k, const float* fluidvar, const int Nx, const int Ny, const int Nz) // \rho * \vec{u}\cdot\vec{u} * 0.5
{
    /* IMPLEMENT KE */
}

__device__ float B_dot_u(int i, int j, int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    /* IMPLEMENT \vec{B}\cdot\vec{u}*/
}