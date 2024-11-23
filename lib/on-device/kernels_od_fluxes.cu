#include "helper_functions.cuh"

// row-major, column-minor order
#define IDX3D(i, j, k, Nx, Ny, Nz) ((k) * (Nx * Ny) + (i) * Ny + j) // parentheses are necessary to avoid calculating `i - 1 * Ny` or `k - 1 * (Nx * Ny)`


// Rho
__device__ float XFluxRho(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    int cube_size = Nx * Ny * Nz;
    return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size]; // x-momentum density
}
__device__ float YFluxRho(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    int cube_size = Nx * Ny * Nz;
    return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size]; // y-momentum density
}
__device__ float ZFluxRho(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    int cube_size = Nx * Ny * Nz;
    return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size]; // z-momentum density
}

// RhoVX
__device__ float XFluxRhoVX(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    float Bsq = B_sq(i, j, k, fluidvar, Nx, Ny, Nz);
    float ke = KE(i, j, k, fluidvar, Nx, Ny, Nz);
    int cube_size = Nx * Ny * Nz;
    return (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size], 2)
        - pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size], 2) + p(i, j, k, fluidvar, Bsq, ke, Nx, Ny, Nz)
        + Bsq / 2.0;
}
__device__ float YFluxRhoVX(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    int cube_size = Nx * Ny * Nz;
    return (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size]
        - fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size];
}
__device__ float ZFluxRhoVX(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    int cube_size = Nx * Ny * Nz;
    return (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size]
        - fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size];
}

// RhoVY
__device__ float XFluxRhoVY(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    // int cube_size = Nx * Ny * Nz;
    // return (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size]
    //     - fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size];
    return YFluxRhoVX(i, j, k, fluidvar, Nx, Ny, Nz);
}

__device__ float YFluxRhoVY(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    float Bsq = B_sq(i, j, k, fluidvar, Nx, Ny, Nz);
    float ke = KE(i, j, k, fluidvar, Nx, Ny, Nz);
    int cube_size = Nx * Ny * Nz;
    return (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size], 2)
        - pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size], 2) + p(i, j, k, fluidvar, Bsq, ke, Nx, Ny, Nz)
        + Bsq / 2.0;
}
__device__ float ZFluxRhoVY(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    int cube_size = Nx * Ny * Nz;
    return (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size]
        - fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size];
}

// RhoVZ
__device__ float XFluxRhoVZ(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    return ZFluxRhoVX(i, j, k, fluidvar, Nx, Ny, Nz);
}

__device__ float YFluxRhoVZ(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    return ZFluxRhoVY(i, j, k, fluidvar, Nx, Ny, Nz);
}

__device__ float ZFluxRhoVZ(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    float Bsq = B_sq(i, j, k, fluidvar, Nx, Ny, Nz);
    float ke = KE(i, j, k, fluidvar, Nx, Ny, Nz);
    int cube_size = Nx * Ny * Nz;
    return (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size], 2)
        - pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size], 2) + p(i, j, k, fluidvar, Bsq, ke, Nx, Ny, Nz)
        + Bsq / 2.0;
}

// Bx
__device__ float XFluxBX(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    return 0.0;
}

__device__ float YFluxBX(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    int cube_size = Nx * Ny * Nz;
    return (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size]
        - fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size];
}

__device__ float ZFluxBX(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    int cube_size = Nx * Ny * Nz;
    return (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size]
        - fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size];
}

// By
__device__ float XFluxBY(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    return -1.0 * YFluxBX(i, j, k, fluidvar, Nx, Ny, Nz);
}

__device__ float YFluxBY(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    return 0.0;
}

__device__ float ZFluxBY(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    int cube_size = Nx * Ny * Nz;
    return (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size]
        - fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size];
}

// Bz
__device__ float XFluxBZ(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    return -1.0 * ZFluxBX(i, j, k, fluidvar, Nx, Ny, Nz);
}

__device__ float YFluxBZ(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    return -1.0 * ZFluxBY(i, j, k, fluidvar, Nx, Ny, Nz);
}

__device__ float ZFluxBZ(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    return 0.0;
}

// e
__device__ float XFluxE(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    float Bsq = B_sq(i, j, k, fluidvar, Nx, Ny, Nz);
    float ke = KE(i, j, k, fluidvar, Nx, Ny, Nz);
    float pressure = p(i, j, k, fluidvar, Bsq, ke, Nx, Ny, Nz);
    float Bdotu = B_dot_u(i, j, k, fluidvar, Nx, Ny, Nz);
    int cube_size = Nx * Ny * Nz;
    return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] + pressure + Bsq 
        * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)])
        - Bdotu * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size];
}

__device__ float YFluxE(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    float Bsq = B_sq(i, j, k, fluidvar, Nx, Ny, Nz);
    float ke = KE(i, j, k, fluidvar, Nx, Ny, Nz);
    float pressure = p(i, j, k, fluidvar, Bsq, ke, Nx, Ny, Nz);
    float Bdotu = B_dot_u(i, j, k, fluidvar, Nx, Ny, Nz);
    int cube_size = Nx * Ny * Nz;
    return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] + pressure + Bsq 
        * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)])
        - Bdotu * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size];
}

__device__ float ZFluxE(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    float Bsq = B_sq(i, j, k, fluidvar, Nx, Ny, Nz);
    float ke = KE(i, j, k, fluidvar, Nx, Ny, Nz);
    float pressure = p(i, j, k, fluidvar, Bsq, ke, Nx, Ny, Nz);
    float Bdotu = B_dot_u(i, j, k, fluidvar, Nx, Ny, Nz);
    int cube_size = Nx * Ny * Nz;
    return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] + pressure + Bsq 
        * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)])
        - Bdotu * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size];
}