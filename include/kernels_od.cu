#include <math.h>
#include "kernels_od.cuh"
#include "kernels_od_fluxes.cuh"
#include "helper_functions.cuh"

// row-major, column-minor order
#define IDX3D(i, j, k, Nx, Ny, Nz) (k * (Nx * Ny) + i * Ny + j)

/* 
Needed to be refactored to compactify all data into three arrays:
(1) float *fluidvar
(2) float *intvar
(3) float *fluidvar_np1

Furthermore, there was a race condition amongst the intermediate variable calculation. 

To solve this, the intermediate variable's computation is moved outside of the fluid advance loop, and into its own kernel. 
They are precomputed after the initial conditions, and then compute in the loop while writing data out in order to add minimal synchronization barriers.   

Here is what the storage pattern looks like:
fluidvar -> [rho_{000}, rho_{010}, rho_{020}, ..., rho_{0,Ny-1,0}, rho_{100}, ..., rho_{Nx-1,Ny-1,Nz-1}, rhov_x_{000}, rhov_x_{010}, ... , e_{Nx-1,Ny-1,Nz-1}]
*/

// Global kernels
__global__ void SwapSimData(float* fluidvar, const float* fluidvar_np1, const int Nx, const int Ny, const int Nz)
    {
    int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    int tidz = threadIdx.z + blockDim.z * blockIdx.z;

    int xthreads = blockDim.x * gridDim.x;
    int ythreads = blockDim.y * gridDim.y;
    int zthreads = blockDim.z * gridDim.z;

    int cube_size = Nx * Ny * Nz;
    for (int k = tidz; k < Nz; k += zthreads){
        for (int i = tidx; i < Nx; i += xthreads){
            for (int j = tidy; j < Ny; j += ythreads){
                fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz)]; // rho
                fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size]; // rhov_x
                fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size]; // rhov_y
                fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size]; // rhov_z
                fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size]; // Bx
                fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size]; // By 
                fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size]; // Bz
                fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size]; // e
            }
        }
    }
    return;
    }

__global__ void FluidAdvance(float* fluidvar_np1, const float* fluidvar, const float* intvar, 
    const float D, const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
    {
    int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    int tidz = threadIdx.z + blockDim.z * blockIdx.z;

    int xthreads = blockDim.x * gridDim.x;
    int ythreads = blockDim.y * gridDim.y;
    int zthreads = blockDim.z * gridDim.z;

    int cube_size = Nx * Ny * Nz;
    for (int k = tidz + 1; k < Nz - 1; k += zthreads){
        for (int i = tidx + 1; i < Nx - 1; k += xthreads){
            for (int j = tidy + 1; j < Ny - 1; j += ythreads){
                fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvRho(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                            + numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, Nx, Ny, Nz); // rho
                fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] =  LaxWendroffAdvRhoVX(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                            + numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, Nx, Ny, Nz); // rhov_x
                fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] =  LaxWendroffAdvRhoVY(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                            + numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, Nx, Ny, Nz); // rhov_y
                fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] =  LaxWendroffAdvRhoVZ(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                            + numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, Nx, Ny, Nz); // rhov_z
                fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] =  LaxWendroffAdvBX(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                            + numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, Nx, Ny, Nz); // Bx
                fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] =  LaxWendroffAdvBY(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                            + numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, Nx, Ny, Nz); // By 
                fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] =  LaxWendroffAdvBZ(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                            + numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, Nx, Ny, Nz); // Bz
                fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] =  LaxWendroffAdvE(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                            + numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, Nx, Ny, Nz); // e
            }
        }
    }
    return;
    }

__global__ void BoundaryConditions(float* fluidvar_np1, const float* fluidvar, const float* intvar, 
    const float D, const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
    int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    int tidz = threadIdx.z + blockDim.z * blockIdx.z;

    /* IMPLEMENT PBCs */
    
    return;
    }

// Device kernels
__device__ float LaxWendroffAdvRho(const int i, const int j, const int k, 
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return 0.5 * (rho[IDX3D(i, j, k, Nx, Ny, Nz)] + rho_int[IDX3D(i, j, k, Nx, Ny, Nz)])
        - 0.5 * (dt / dx) * (XFluxRho(i+1, j, k, rhovx_int, Nx, Ny, Nz) - XFluxRho(i, j, k, rhovx_int, Nx, Ny, Nz))
        - 0.5 * (dt / dy) * (YFluxRho(i, j+1, k, rhovy_int, Nx, Ny, Nz) - YFluxRho(i, j, k, rhovy_int, Nx, Ny, Nz))
        - 0.5 * (dt / dz) * (ZFluxRho(i, j, k+1, rhovz_int, Nx, Ny, Nz) - ZFluxRho(i, j, k, rhovz_int, Nx, Ny, Nz));
    }

__device__ float LaxWendroffAdvRhoVX(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {

    }

__device__ float LaxWendroffAdvRhoVY(const int i, const int j, const int k, 
    const float* fluidvar, const float* intvar,  
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
    
    }

__device__ float LaxWendroffAdvRhoVZ(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
    
    }

__device__ float LaxWendroffAdvBX(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {

    }

__device__ float LaxWendroffAdvBY(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
    
    }

__device__ float LaxWendroffAdvBZ(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
    
    }

__device__ float LaxWendroffAdvE(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
    
    }

// Helper Functions
__device__ float B_sq(int i, int j, int k, const float* fluidvar, const int Nx, const int Ny, const int Nz); // B / \sqrt{\mu_{0}} -> B 
__device__ float p(int i, int j, int k, const float* fluidvar, const float B_sq, const float KE, const int Nx, const int Ny, const int Nz);
__device__ float KE(int i, int j, int k, const float* fluidvar, const int Nx, const int Ny, const int Nz); // \rho * \vec{u}\cdot\vec{u} * 0.5
__device__ float B_dot_u(int i, int j, int k, const float* fluidvar, const int Nx, const int Ny, const int Nz);

__device__ float numericalDiffusion(const int i, const int j, const int k, const float* fluid_var, 
    const float D, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

__device__ float numericalDiffusionFront(const int i, const int j, const float* fluid_var, 
    const float D, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);

__device__ float numericalDiffusionBack(const int i, const int j, const float* fluid_var, 
    const float D, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz);