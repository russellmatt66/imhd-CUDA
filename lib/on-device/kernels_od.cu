#include <math.h>
#include <stdio.h>

#include "kernels_od.cuh"
#include "kernels_od_fluxes.cuh"
#include "helper_functions.cuh"
#include "diffusion.cuh"

// row-major, column-minor order
#define IDX3D(i, j, k, Nx, Ny, Nz) ((k) * (Nx * Ny) + (i) * Ny + j) // parentheses are necessary to avoid calculating `i - 1 * Ny` or `k - 1 * (Nx * Ny)`

/* 
Here is what the storage pattern looks like:
fluidvar -> [rho_{000}, rho_{010}, rho_{020}, ..., rho_{0,Ny-1,0}, rho_{100}, ..., rho_{Nx-1,Ny-1,Nz-1}, rhov_x_{000}, rhov_x_{010}, ... , e_{Nx-1,Ny-1,Nz-1}]
*/

// Global kernels

// FluidAdvance uses kernels that thrash the cache, very badly
// FluidAdvance2 uses kernels that thrash the cache, less badly
// FluidAdvanceLocal uses kernels that do not thrash the cache, but it puts a lot of pressure on the registers
__global__ void FluidAdvanceLocal(float* fluidvar, const float* intvar, 
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

        float rho = 0.0, rhovx = 0.0, rhovy = 0.0, rhovz = 0.0, Bx = 0.0, By = 0.0, Bz = 0.0, e = 0.0;

        float rho_int = 0.0, rhovx_int = 0.0, rhovy_int = 0.0, rhovz_int = 0.0, Bx_int = 0.0, By_int = 0.0, Bz_int = 0.0, e_int = 0.0;

        float rho_int_im1 = 0.0, rhovx_int_im1 = 0.0, rhovy_int_im1 = 0.0, rhovz_int_im1 = 0.0, Bx_int_im1 = 0.0, By_int_im1 = 0.0, Bz_int_im1 = 0.0, e_int_im1 = 0.0;
        float rho_int_jm1 = 0.0, rhovx_int_jm1 = 0.0, rhovy_int_jm1 = 0.0, rhovz_int_jm1 = 0.0, Bx_int_jm1 = 0.0, By_int_jm1 = 0.0, Bz_int_jm1 = 0.0, e_int_jm1 = 0.0;
        float rho_int_km1 = 0.0, rhovx_int_km1 = 0.0, rhovy_int_km1 = 0.0, rhovz_int_km1 = 0.0, Bx_int_km1 = 0.0, By_int_km1 = 0.0, Bz_int_km1 = 0.0, e_int_km1 = 0.0;


        // The analogue to the `2` __device__ kernels here were the parts that were thrashing the cache in `FluidAdvance`
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){
            for (int i = tidx + 1; i < Nx - 1; i += xthreads){
                for (int j = tidy + 1; j < Ny - 1; j += ythreads){
                    /* 
                    There's a tradeoff here between register pressure, and thrashing the cache: 
                    Either the register pressure is kept low, and, instead of using local memory, global/shared memory is passed to the `__device__` kernels 
                    WHICH MEANS that global/shared memory is then thrashed because it is repeatedly re-accessed by the `__device__` kernels
                    OR
                    Register pressure is made high by storing all the necessary data locally, but global/shared memory is not thrashed because the access is not repeated 
                    */
                    rho = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)];
                    rhovx = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size];
                    rhovy = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size];
                    rhovz = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size];
                    Bx = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size];
                    By = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size];
                    Bz = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size];
                    e = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size];

                    rho_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz)];
                    rhovx_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size];
                    rhovy_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size];
                    rhovz_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size];
                    Bx_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size];
                    By_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size];
                    Bz_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size];
                    e_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size];
                    
                    rho_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz)];
                    rhovx_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + cube_size];
                    rhovy_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + 2 * cube_size];
                    rhovz_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + 3 * cube_size];
                    Bx_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + 4 * cube_size];
                    By_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + 5 * cube_size];
                    Bz_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + 6 * cube_size];
                    e_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + 7 * cube_size];
                    
                    rho_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz)];
                    rhovx_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + cube_size];
                    rhovy_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 2 * cube_size];
                    rhovz_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 3 * cube_size];
                    Bx_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 4 * cube_size];
                    By_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 5 * cube_size];
                    Bz_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 6 * cube_size];
                    e_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 7 * cube_size];
                    
                    rho_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz)];
                    rhovx_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + cube_size];
                    rhovy_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 2 * cube_size];
                    rhovz_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 3 * cube_size];
                    Bx_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 4 * cube_size];
                    By_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 5 * cube_size];
                    Bz_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 6 * cube_size];
                    e_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 7 * cube_size];

                    /* 
                    Write the `*Local` kernels so that they don't read data, at any point, and just use the already-read data 
                    */
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvRhoLocal(rho, rho_int, rhovx_int, rho_int_im1, rhovy_int, rhovy_int_jm1, rhovz_int, rhovz_int_km1, dt, dx, dy, dz) 
                                                                + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 0, Nx, Ny, Nz); // rho

                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] =  LaxWendroffAdvRhoVXLocal(dt, dx, dy, dz) 
                                                                + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 1, Nx, Ny, Nz); // rhov_x

                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] =  LaxWendroffAdvRhoVYLocal(dt, dx, dy, dz) 
                                                                + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 2, Nx, Ny, Nz); // rhov_y

                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] =  LaxWendroffAdvRhoVZLocal(dt, dx, dy, dz) 
                                                                + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 3, Nx, Ny, Nz); // rhov_z

                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] =  LaxWendroffAdvBXLocal(dt, dx, dy, dz) 
                                                                + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 4, Nx, Ny, Nz); // Bx

                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] =  LaxWendroffAdvBYLocal(dt, dx, dy, dz) 
                                                                + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 5, Nx, Ny, Nz); // By 

                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] =  LaxWendroffAdvBZLocal(dt, dx, dy, dz) 
                                                                + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 6, Nx, Ny, Nz); // Bz

                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] =  LaxWendroffAdvELocal(dt, dx, dy, dz) 
                                                                + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 7, Nx, Ny, Nz); // e
                }
            }
        }
        return;
    }

// Use non-cache thrashing, and readable, over-loaded flux functions
// Accept all local data, read nothing in from memory
__device__ float LaxWendroffAdvRhoLocal(const float rho, const float rho_int, 
    const float rhovx_int, const float rhovx_int_im1,
    const float rhovy_int, const float rhovy_int_jm1,
    const float rhovz_int, const float rhovz_int_km1,
    const float dt, const float dx, const float dy, const float dz)
    {
        /* Can go a step deeper, and just do away with *FluxRho - they're just the associated momentum density */
        return 0.5 * (rho + rho_int)
        - 0.5 * (dt / dx) * (XFluxRho(rhovx_int) - XFluxRho(rhovx_int_im1))
        - 0.5 * (dt / dy) * (YFluxRho(rhovy_int) - YFluxRho(rhovy_int_jm1))
        - 0.5 * (dt / dz) * (ZFluxRho(rhovz_int) - ZFluxRho(rhovz_int_km1));
    }

__device__ float LaxWendroffAdvRhoVXLocal(
    const float dt, const float dx, const float dy, const float dz)
    {
        /* Put code that advances state of variable here */ 
    }

__device__ float LaxWendroffAdvRhoVYLocal(
    const float dt, const float dx, const float dy, const float dz)
    {
        /* Put code that advances state of variable here */ 
    }

__device__ float LaxWendroffAdvRhoVZLocal(
    const float dt, const float dx, const float dy, const float dz)
    {
        /* Put code that advances state of variable here */ 
    }

__device__ float LaxWendroffAdvBXLocal(
    const float dt, const float dx, const float dy, const float dz)
    {
        /* Put code that advances state of variable here */ 
    }

__device__ float LaxWendroffAdvBYLocal(
    const float dt, const float dx, const float dy, const float dz)
    {
        /* Put code that advances state of variable here */ 
    }

__device__ float LaxWendroffAdvBZLocal(
    const float dt, const float dx, const float dy, const float dz)
    {
        /* Put code that advances state of variable here */ 
    }

__device__ float LaxWendroffAdvELocal(
    const float dt, const float dx, const float dy, const float dz)
    {
        /* Put code that advances state of variable here */ 
    }

/* 
LaxWendroffAdv{}2 functions use flux functions that do not access memory, but they themselves DO
Therefore, it is likely that an implementation which leverages them will thrash the cache
*/
__device__ float LaxWendroffAdvRho2(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    { 
        /* Put the code for advancing the fluid state here */
    }
__device__ float LaxWendroffAdvRhoVX2(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size])
                - 0.5 * (dt / dx) * (XFluxRhoVX(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxRhoVX(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxRhoVX(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz));  
    }

__device__ float LaxWendroffAdvRhoVY2(const int i, const int j, const int k, 
    const float* fluidvar, const float* intvar,  
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size])
                - 0.5 * (dt / dx) * (XFluxRhoVY(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxRhoVY(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxRhoVY(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz)); 
    }

__device__ float LaxWendroffAdvRhoVZ2(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size])
                - 0.5 * (dt / dx) * (XFluxRhoVZ(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxRhoVZ(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxRhoVZ(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz)); 
    }

__device__ float LaxWendroffAdvBX2(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size])
                - 0.5 * (dt / dx) * (XFluxBX(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxBX(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxBX(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxBX(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxBX(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxBX(i, j, k, intvar, Nx, Ny, Nz)); 
    }

__device__ float LaxWendroffAdvBY2(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size])
                - 0.5 * (dt / dx) * (XFluxBY(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxBY(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxBY(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxBY(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxBY(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxBY(i, j, k, intvar, Nx, Ny, Nz)); 
    }

__device__ float LaxWendroffAdvBZ2(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size])
                - 0.5 * (dt / dx) * (XFluxBZ(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxBZ(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxBZ(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxBZ(i, j, k, intvar, Nx, Ny, Nz)); 
    }

__device__ float LaxWendroffAdvE2(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size])
                - 0.5 * (dt / dx) * (XFluxE(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxE(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxE(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxE(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxE(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxE(i, j, k, intvar, Nx, Ny, Nz)); 
    }

// THESE KERNELS THRASH THE CACHE
// 66 registers per thread
__global__ void FluidAdvance(float* fluidvar, const float* intvar, 
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
            for (int i = tidx + 1; i < Nx - 1; i += xthreads){
                for (int j = tidy + 1; j < Ny - 1; j += ythreads){
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvRho(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                                + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 0, Nx, Ny, Nz); // rho
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] =  LaxWendroffAdvRhoVX(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                                + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 1, Nx, Ny, Nz); // rhov_x
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] =  LaxWendroffAdvRhoVY(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                                + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 2, Nx, Ny, Nz); // rhov_y
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] =  LaxWendroffAdvRhoVZ(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                                + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 3, Nx, Ny, Nz); // rhov_z
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] =  LaxWendroffAdvBX(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                                + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 4, Nx, Ny, Nz); // Bx
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] =  LaxWendroffAdvBY(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                                + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 5, Nx, Ny, Nz); // By 
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] =  LaxWendroffAdvBZ(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                                + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 6, Nx, Ny, Nz); // Bz
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] =  LaxWendroffAdvE(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
                                                                + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 7, Nx, Ny, Nz); // e
                }
            }
        }
        return;
    }

// 120 registers per thread
__global__ void BoundaryConditions(float* fluidvar, const float* intvar, 
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

    // k = 0 and k = Nz - 1
    int k = 0;
    for (int i = tidx + 1; i < Nx - 1; i += xthreads){
        for (int j = tidy + 1; j < Ny - 1; j += ythreads){
            k = 0;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] + intvar[IDX3D(i, j, k, Nx, Ny, Nz)])
                                                        - 0.5 * (dt / dx) * (XFluxRho(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRho(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxRho(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRho(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxRho(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxRho(i, j, k, intvar, Nx, Ny, Nz));
                                                        + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 0, Nx, Ny, Nz);

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxRhoVX(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxRhoVX(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxRhoVX(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz));
                                                        + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 1, Nx, Ny, Nz);
            
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxRhoVY(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxRhoVY(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxRhoVY(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz));
                                                        + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 2, Nx, Ny, Nz);

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxRhoVZ(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxRhoVZ(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxRhoVZ(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz));
                                                        + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 3, Nx, Ny, Nz);

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxBX(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBX(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxBX(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBX(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxBX(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxBX(i, j, k, intvar, Nx, Ny, Nz));
                                                        + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 4, Nx, Ny, Nz);

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxBY(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBY(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxBY(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBY(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxBY(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxBY(i, j, k, intvar, Nx, Ny, Nz));
                                                        + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 5, Nx, Ny, Nz);

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxBZ(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxBZ(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxBZ(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxBZ(i, j, k, intvar, Nx, Ny, Nz));
                                                        + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 6, Nx, Ny, Nz);

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxE(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxE(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxE(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxBZ(i, j, k, intvar, Nx, Ny, Nz));
                                                        + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 7, Nx, Ny, Nz);

            k = Nz - 1;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] + intvar[IDX3D(i, j, k, Nx, Ny, Nz)])
                                                        - 0.5 * (dt / dx) * (XFluxRho(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRho(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxRho(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRho(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxRho(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxRho(i, j, k, intvar, Nx, Ny, Nz));
                                                        + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 0, Nx, Ny, Nz);

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxRhoVX(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxRhoVX(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxRhoVX(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz));
                                                        + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 1, Nx, Ny, Nz);
            
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxRhoVY(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxRhoVY(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxRhoVY(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz));
                                                        + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 2, Nx, Ny, Nz);

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxRhoVZ(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxRhoVZ(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxRhoVZ(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz));
                                                        + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 3, Nx, Ny, Nz);

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxBX(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBX(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxBX(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBX(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxBX(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxBX(i, j, k, intvar, Nx, Ny, Nz));
                                                        + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 4, Nx, Ny, Nz);

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxBY(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBY(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxBY(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBY(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxBY(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxBY(i, j, k, intvar, Nx, Ny, Nz));
                                                        + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 5, Nx, Ny, Nz);

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxBZ(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxBZ(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxBZ(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxBZ(i, j, k, intvar, Nx, Ny, Nz));
                                                        + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 6, Nx, Ny, Nz);

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size])
                                                        - 0.5 * (dt / dx) * (XFluxE(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dy) * (YFluxE(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                                                        - 0.5 * (dt / dz) * (ZFluxE(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxBZ(i, j, k, intvar, Nx, Ny, Nz));
                                                        + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 7, Nx, Ny, Nz);
            
            // THEN, ACCUMULATE THE RESULTS ONTO ONE FACE, MAP AROUND TO THE OTHER, AND CONTINUE
            for (int ivf = 0; ivf < 8; ivf++){
                fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz) + ivf * cube_size] += fluidvar[IDX3D(i, j, Nz - 1, Nx, Ny, Nz) + ivf * cube_size];
                fluidvar[IDX3D(i, j, Nz - 1, Nx, Ny, Nz) + ivf * cube_size] = fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz) + ivf * cube_size];
            }
        }
    }

    /* 
    B.Cs on BOTTOM (II) 
    (i = 0, j, k) 
    and
    B.Cs on TOP (IV)
    (i = Nx-1, j, k) 
    */
    int i = 0; 
    for (int k = tidz; k < Nz; k += zthreads){ 
        for (int j = tidy; j < Ny; j += ythreads){
            i = 0;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 10.0; /* Magic vacuum number */
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);

            i = Nx - 1;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 10.0; /* Magic vacuum number */
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
        }
    }

    /* 
    B.Cs on LEFT (V)
    (i, j = 0, k) 
    and
    B.Cs on RIGHT (III)
    (i, j = N-1, k) 
    */
    int j = 0;
    for (int k = tidz; k < Nz; k += zthreads){
        for (int i = tidx + 1; i < Nx - 1; i += xthreads){
            j = 0;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 10.0; /* Magic vacuum number */
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
            
            j = Ny - 1;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 10.0; /* Magic vacuum number */
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
        }
    }

    return;
    }

// Device kernels
__device__ float LaxWendroffAdvRho(const int i, const int j, const int k, 
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] + intvar[IDX3D(i, j, k, Nx, Ny, Nz)])
        - 0.5 * (dt / dx) * (XFluxRho(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxRho(i, j, k, intvar, Nx, Ny, Nz))
        - 0.5 * (dt / dy) * (YFluxRho(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxRho(i, j, k, intvar, Nx, Ny, Nz))
        - 0.5 * (dt / dz) * (ZFluxRho(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxRho(i, j, k, intvar, Nx, Ny, Nz));
    }

__device__ float LaxWendroffAdvRhoVX(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size])
                - 0.5 * (dt / dx) * (XFluxRhoVX(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxRhoVX(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxRhoVX(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz));  
    }

__device__ float LaxWendroffAdvRhoVY(const int i, const int j, const int k, 
    const float* fluidvar, const float* intvar,  
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size])
                - 0.5 * (dt / dx) * (XFluxRhoVY(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxRhoVY(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxRhoVY(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz)); 
    }

__device__ float LaxWendroffAdvRhoVZ(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size])
                - 0.5 * (dt / dx) * (XFluxRhoVZ(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxRhoVZ(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxRhoVZ(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz)); 
    }

__device__ float LaxWendroffAdvBX(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size])
                - 0.5 * (dt / dx) * (XFluxBX(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxBX(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxBX(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxBX(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxBX(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxBX(i, j, k, intvar, Nx, Ny, Nz)); 
    }

__device__ float LaxWendroffAdvBY(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size])
                - 0.5 * (dt / dx) * (XFluxBY(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxBY(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxBY(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxBY(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxBY(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxBY(i, j, k, intvar, Nx, Ny, Nz)); 
    }

__device__ float LaxWendroffAdvBZ(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size])
                - 0.5 * (dt / dx) * (XFluxBZ(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxBZ(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxBZ(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxBZ(i, j, k, intvar, Nx, Ny, Nz)); 
    }

__device__ float LaxWendroffAdvE(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size])
                - 0.5 * (dt / dx) * (XFluxE(i+1, j, k, intvar, Nx, Ny, Nz) - XFluxE(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxE(i, j+1, k, intvar, Nx, Ny, Nz) - YFluxE(i, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxE(i, j, k+1, intvar, Nx, Ny, Nz) - ZFluxE(i, j, k, intvar, Nx, Ny, Nz)); 
    }

/* 
DELETE THESE WHEN IT'S SAFE 
WHAT ARE THEY? 
- Old kernels for doing the boundary conditions and solving the advection-diffusion step

WHY ARE THEY HERE? 
- They are from a period in development where the numerical diffusion was being implemented in an invalid way
- This invalid way would introduce a race condition if Q^{n} was updated in-place to Q^{n+1}
- Therefore, `fluidvar_np1` stored Q^{n+1}, which was written to, and `fluidvar` stored Q^{n}, which was read from
- `__global__ void SwapSimData(...)` transferred `fluidvar_np1` to `fluidvar` after a timestep was over
*/

// __global__ void FluidAdvance(float* fluidvar_np1, const float* fluidvar, const float* intvar, 
// const float D, const float dt, const float dx, const float dy, const float dz, 
// const int Nx, const int Ny, const int Nz)
// {
// int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
// int tidy = threadIdx.y + blockDim.y * blockIdx.y;
// int tidz = threadIdx.z + blockDim.z * blockIdx.z;

// int xthreads = blockDim.x * gridDim.x;
// int ythreads = blockDim.y * gridDim.y;
// int zthreads = blockDim.z * gridDim.z;

// int cube_size = Nx * Ny * Nz;

// for (int k = tidz + 1; k < Nz - 1; k += zthreads){
//     for (int i = tidx + 1; i < Nx - 1; i += xthreads){
//         for (int j = tidy + 1; j < Ny - 1; j += ythreads){
//             fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvRho(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
//                                                         + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 0, Nx, Ny, Nz); // rho
//             fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] =  LaxWendroffAdvRhoVX(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
//                                                         + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 1, Nx, Ny, Nz); // rhov_x
//             fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] =  LaxWendroffAdvRhoVY(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
//                                                         + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 2, Nx, Ny, Nz); // rhov_y
//             fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] =  LaxWendroffAdvRhoVZ(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
//                                                         + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 3, Nx, Ny, Nz); // rhov_z
//             fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] =  LaxWendroffAdvBX(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
//                                                         + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 4, Nx, Ny, Nz); // Bx
//             fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] =  LaxWendroffAdvBY(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
//                                                         + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 5, Nx, Ny, Nz); // By 
//             fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] =  LaxWendroffAdvBZ(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
//                                                         + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 6, Nx, Ny, Nz); // Bz
//             fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] =  LaxWendroffAdvE(i, j, k, fluidvar, intvar, dt, dx, dy, dz, Nx, Ny, Nz) 
//                                                         + dt * numericalDiffusion(i, j, k, intvar, D, dx, dy, dz, 7, Nx, Ny, Nz); // e
//         }
//     }
// }
// return;
// }

// // 120 registers per thread
// __global__ void BoundaryConditions(volatile float* fluidvar_np1, const float* fluidvar, const float* intvar, 
// const float D, const float dt, const float dx, const float dy, const float dz,
// const int Nx, const int Ny, const int Nz)
// {
// int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
// int tidy = threadIdx.y + blockDim.y * blockIdx.y;
// int tidz = threadIdx.z + blockDim.z * blockIdx.z;

// int xthreads = blockDim.x * gridDim.x;
// int ythreads = blockDim.y * gridDim.y;
// int zthreads = blockDim.z * gridDim.z;

// int cube_size = Nx * Ny * Nz;

// // k = 0 and k = Nz - 1
// int k = 0;
// for (int i = tidx + 1; i < Nx - 1; i += xthreads){
//     for (int j = tidy + 1; j < Ny - 1; j += ythreads){
//         k = 0;
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] + intvar[IDX3D(i, j, k, Nx, Ny, Nz)])
//                                                     - 0.5 * (dt / dx) * (XFluxRho(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRho(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dy) * (YFluxRho(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRho(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dz) * (ZFluxRho(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxRho(i, j, k, intvar, Nx, Ny, Nz));
//                                                     + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 0, Nx, Ny, Nz);

//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size])
//                                                     - 0.5 * (dt / dx) * (XFluxRhoVX(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dy) * (YFluxRhoVX(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dz) * (ZFluxRhoVX(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz));
//                                                     + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 1, Nx, Ny, Nz);
        
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size])
//                                                     - 0.5 * (dt / dx) * (XFluxRhoVY(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dy) * (YFluxRhoVY(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dz) * (ZFluxRhoVY(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz));
//                                                     + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 2, Nx, Ny, Nz);

//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size])
//                                                     - 0.5 * (dt / dx) * (XFluxRhoVZ(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dy) * (YFluxRhoVZ(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dz) * (ZFluxRhoVZ(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz));
//                                                     + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 3, Nx, Ny, Nz);

//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size])
//                                                     - 0.5 * (dt / dx) * (XFluxBX(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBX(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dy) * (YFluxBX(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBX(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dz) * (ZFluxBX(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxBX(i, j, k, intvar, Nx, Ny, Nz));
//                                                     + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 4, Nx, Ny, Nz);

//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size])
//                                                     - 0.5 * (dt / dx) * (XFluxBY(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBY(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dy) * (YFluxBY(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBY(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dz) * (ZFluxBY(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxBY(i, j, k, intvar, Nx, Ny, Nz));
//                                                     + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 5, Nx, Ny, Nz);

//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size])
//                                                     - 0.5 * (dt / dx) * (XFluxBZ(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dy) * (YFluxBZ(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dz) * (ZFluxBZ(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxBZ(i, j, k, intvar, Nx, Ny, Nz));
//                                                     + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 6, Nx, Ny, Nz);

//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size])
//                                                     - 0.5 * (dt / dx) * (XFluxE(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dy) * (YFluxE(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dz) * (ZFluxE(i, j, k + 1, intvar, Nx, Ny, Nz) - ZFluxBZ(i, j, k, intvar, Nx, Ny, Nz));
//                                                     + dt * numericalDiffusionFront(i, j, intvar, D, dx, dy, dz, 7, Nx, Ny, Nz);

//         k = Nz - 1;
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] + intvar[IDX3D(i, j, k, Nx, Ny, Nz)])
//                                                     - 0.5 * (dt / dx) * (XFluxRho(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRho(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dy) * (YFluxRho(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRho(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dz) * (ZFluxRho(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxRho(i, j, k, intvar, Nx, Ny, Nz));
//                                                     + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 0, Nx, Ny, Nz);

//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size])
//                                                     - 0.5 * (dt / dx) * (XFluxRhoVX(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dy) * (YFluxRhoVX(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dz) * (ZFluxRhoVX(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz));
//                                                     + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 1, Nx, Ny, Nz);
        
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size])
//                                                     - 0.5 * (dt / dx) * (XFluxRhoVY(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dy) * (YFluxRhoVY(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dz) * (ZFluxRhoVY(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz));
//                                                     + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 2, Nx, Ny, Nz);

//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size])
//                                                     - 0.5 * (dt / dx) * (XFluxRhoVZ(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dy) * (YFluxRhoVZ(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dz) * (ZFluxRhoVZ(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz));
//                                                     + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 3, Nx, Ny, Nz);

//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size])
//                                                     - 0.5 * (dt / dx) * (XFluxBX(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBX(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dy) * (YFluxBX(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBX(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dz) * (ZFluxBX(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxBX(i, j, k, intvar, Nx, Ny, Nz));
//                                                     + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 4, Nx, Ny, Nz);

//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size])
//                                                     - 0.5 * (dt / dx) * (XFluxBY(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBY(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dy) * (YFluxBY(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBY(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dz) * (ZFluxBY(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxBY(i, j, k, intvar, Nx, Ny, Nz));
//                                                     + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 5, Nx, Ny, Nz);

//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size])
//                                                     - 0.5 * (dt / dx) * (XFluxBZ(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dy) * (YFluxBZ(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dz) * (ZFluxBZ(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxBZ(i, j, k, intvar, Nx, Ny, Nz));
//                                                     + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 6, Nx, Ny, Nz);

//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size])
//                                                     - 0.5 * (dt / dx) * (XFluxE(i + 1, j, k, intvar, Nx, Ny, Nz) - XFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dy) * (YFluxE(i, j + 1, k, intvar, Nx, Ny, Nz) - YFluxBZ(i, j, k, intvar, Nx, Ny, Nz))
//                                                     - 0.5 * (dt / dz) * (ZFluxE(i, j, 1, intvar, Nx, Ny, Nz) - ZFluxBZ(i, j, k, intvar, Nx, Ny, Nz));
//                                                     + dt * numericalDiffusionBack(i, j, intvar, D, dx, dy, dz, 7, Nx, Ny, Nz);
        
//         // THEN, ACCUMULATE THE RESULTS ONTO ONE FACE, MAP AROUND TO THE OTHER, AND CONTINUE
//         for (int ivf = 0; ivf < 8; ivf++){
//             fluidvar_np1[IDX3D(i, j, 0, Nx, Ny, Nz) + ivf * cube_size] += fluidvar_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz) + ivf * cube_size];
//             fluidvar_np1[IDX3D(i, j, Nz - 1, Nx, Ny, Nz) + ivf * cube_size] = fluidvar_np1[IDX3D(i, j, 0, Nx, Ny, Nz) + ivf * cube_size];
//         }
//     }
// }

// /* 
// B.Cs on BOTTOM (II) 
// (i = 0, j, k) 
// and
// B.Cs on TOP (IV)
// (i = Nx-1, j, k) 
// */
// int i = 0; 
// for (int k = tidz; k < Nz; k += zthreads){ 
//     for (int j = tidy; j < Ny; j += ythreads){
//         i = 0;
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0; /* Magic vacuum number */
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);

//         i = Nx - 1;
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0; /* Magic vacuum number */
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
//     }
// }

// /* 
// B.Cs on LEFT (V)
// (i, j = 0, k) 
// and
// B.Cs on RIGHT (III)
// (i, j = N-1, k) 
// */
// int j = 0;
// for (int k = tidz; k < Nz; k += zthreads){
//     for (int i = tidx + 1; i < Nx - 1; i += xthreads){
//         j = 0;
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0; /* Magic vacuum number */
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
        
//         j = Ny - 1;
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0; /* Magic vacuum number */
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0; // Rigid wall
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0; // Perfectly-conducting wall
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0; 
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
//         fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = p(i, j, k, fluidvar, 0.0, 0.0, Nx, Ny, Nz) / (gamma - 1.0);
//     }
// }

// return;
// }

// 40 registers per thread
// __global__ void SwapSimData(float* fluidvar, const float* fluidvar_np1, const int Nx, const int Ny, const int Nz)
//     {
//     int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
//     int tidy = threadIdx.y + blockDim.y * blockIdx.y;
//     int tidz = threadIdx.z + blockDim.z * blockIdx.z;

//     int xthreads = blockDim.x * gridDim.x;
//     int ythreads = blockDim.y * gridDim.y;
//     int zthreads = blockDim.z * gridDim.z;

//     int cube_size = Nx * Ny * Nz;
//     for (int k = tidz; k < Nz; k += zthreads){
//         for (int i = tidx; i < Nx; i += xthreads){
//             for (int j = tidy; j < Ny; j += ythreads){
//                 fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz)]; // rho
//                 fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size]; // rhov_x
//                 fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size]; // rhov_y
//                 fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size]; // rhov_z
//                 fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size]; // Bx
//                 fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size]; // By 
//                 fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size]; // Bz
//                 fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = fluidvar_np1[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size]; // e
//             }
//         }
//     }
//     return;
//     }