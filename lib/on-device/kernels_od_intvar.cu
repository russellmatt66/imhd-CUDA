#include <stdio.h>

#include "diffusion.cuh"
#include "kernels_od.cuh"
#include "kernels_od_fluxes.cuh"
#include "kernels_od_intvar.cuh"
#include "helper_functions.cuh"

/* THIS SHOULD BE DEFINED A SINGLE TIME IN A SINGLE PLACE */
#define IDX3D(i, j, k, Nx, Ny, Nz) ((k) * (Nx) * (Ny) + (i) * (Ny) + j) // parentheses are necessary to avoid calculating `i - 1 * Ny` or `k - 1 * (Nx * Ny)`

/* 
REGISTER PRESSURES: (registers per thread)
ComputeIntVarsLocalNoDiff=4 (incomplete)
ComputeIntermediateVariablesNoDiff=42
ComputeIntermediateVariablesStrideNoDiff=80
ComputeIntermediateVariablesStride=74
ComputeIntRhoMicroLocalNoDiff=28
ComputeIntRhoVXMicroLocalNoDiff=64
ComputeIntRhoVYMicroLocalNoDiff=64
ComputeIntRhoVZMicroLocalNoDiff=64
ComputeIntBXMicroLocalNoDiff=48
ComputeIntBYMicroLocalNoDiff=36
ComputeIntBZMicroLocalNoDiff=38
ComputeIntEMicroLocalNoDiff=90

*/
// MEGAKERNELS
/* TODO: Megakernel that uses large amounts of registers to avoid thrashing the cache */
__global__ void ComputeIntVarsSHMEMNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int tidx = threadIdx.x + blockDim.x * blockIdx.x;
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;

        int xthreads = blockDim.x * gridDim.x;
        int ythreads = blockDim.y * gridDim.y;
        int zthreads = blockDim.z * gridDim.z;

        /* 
        COMPUTE THE INTERMEDIATE VARIABLES USING SHARED MEMORY
        */

        return;
    }

// Total Parallelism implementation
// Thrashes the cache
// 42 registers per thread
__global__ void ComputeIntermediateVariablesNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        u_int32_t i = threadIdx.x + blockDim.x * blockIdx.x; 
        u_int32_t j = threadIdx.y + blockDim.y * blockIdx.y;
        u_int32_t k = threadIdx.z + blockDim.z * blockIdx.z;
    
        int cube_size = Nx * Ny * Nz;

        if (i < Nx-1 && j < Ny-1 && k < Nz-1) // Predictor is forward-differenced so leave back alone
        {
            intvar[IDX3D(i, j, k, Nx, Ny, Nz)] = intRho(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = intRhoVX(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = intBX(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = intBY(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);// By
            intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = intBZ(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = intE(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e   
        }

        return;
    }

// Thrashes the cache
// 80 registers per thread
__global__ void ComputeIntermediateVariablesStrideNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;

        int xthreads = blockDim.x * gridDim.x;
        int ythreads = blockDim.y * gridDim.y;
        int zthreads = blockDim.z * gridDim.z;
    
        int cube_size = Nx * Ny * Nz;

        /* Why do these start at 1? */
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){
            for (int i = tidx + 1; i < Nx - 1; i += xthreads){
                for (int j = tidy + 1; j < Ny - 1; j += ythreads){
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz)] = intRho(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = intRhoVX(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = intBX(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = intBY(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);// By
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = intBZ(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = intE(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
                
                }
            }
        }

        return;
    }

// 74 registers per thread
/* DIFFUSION NOT VERIFIED */
__global__ void ComputeIntermediateVariablesStride(const float* fluidvar, float* intvar,
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
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz)] = intRho(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz)
                                                            + dt * numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 0, Nx, Ny, Nz); // rho
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = intRhoVX(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz)
                                                            + dt * numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 1, Nx, Ny, Nz); // rhov_x
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz)
                                                            + dt * numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 2, Nx, Ny, Nz); // rhov_y
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz)
                                                            + dt * numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 3, Nx, Ny, Nz); // rhov_z
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = intBX(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz)
                                                            + dt * numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 4, Nx, Ny, Nz); // Bx
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = intBY(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz)
                                                            + dt * numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 5, Nx, Ny, Nz); // By
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = intBZ(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz)
                                                            + dt * numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 6, Nx, Ny, Nz); // Bz
                    intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = intE(i, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz)
                                                            + dt * numericalDiffusion(i, j, k, fluidvar, D, dx, dy, dz, 7, Nx, Ny, Nz); // e
                
                }
            }
        }


        return;
    }

// MICROKERNELS
/* 
NOTE: 
These DO NOT work for initializing solver. 
I DO NOT know why.
*/
// 28 registers per thread
__global__ void ComputeIntRhoMicroLocalNoDiff(const float* fluidvar, float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;
        unsigned int k = threadIdx.z + blockDim.z * blockIdx.z;

        unsigned int cube_size = Nx * Ny * Nz;

        // DECLARE EVERYTHING 
        // float rho = 0.0, rho_ip1 = 0.0, rho_jp1 = 0.0, rho_kp1 = 0.0;
        float rho = 0.0, rho_ip1 = 0.0, rho_jp1 = 0.0, rho_kp1 = 0.0;

        // float rhovx = 0.0, rhovx_ip1 = 0.0, rhovx_jp1 = 0.0, rhovx_kp1 = 0.0;
        float rhovx = 0.0, rhovx_ip1 = 0.0;

        // float rhovy = 0.0, rhovy_ip1 = 0.0, rhovy_jp1 = 0.0, rhovy_kp1 = 0.0;
        float rhovy = 0.0, rhovy_jp1 = 0.0;

        // float rhovz = 0.0, rhovz_ip1 = 0.0, rhovz_jp1 = 0.0, rhovz_kp1 = 0.0;
        float rhovz = 0.0, rhovz_kp1 = 0.0;

        // float Bx = 0.0, Bx_ip1 = 0.0, Bx_jp1 = 0.0, Bx_kp1 = 0.0;

        // float By = 0.0, By_ip1 = 0.0, By_jp1 = 0.0, By_kp1 = 0.0;

        // float Bz = 0.0, Bz_ip1 = 0.0, Bz_jp1 = 0.0, Bz_kp1 = 0.0;

        // float e = 0.0, e_ip1 = 0.0, e_jp1 = 0.0, e_kp1 = 0.0;

        // float p_ijk = 0.0, p_ip1jk = 0.0, p_ijp1k = 0.0, p_ijkp1 = 0.0;

        // float Bsq_ijk = 0.0, Bsq_ip1jk = 0.0, Bsq_ijp1k = 0.0, Bsq_ijkp1 = 0.0;

        // float KE_ijk = 0.0, KE_ip1jk = 0.0, KE_ijp1k = 0.0, KE_ijkp1 = 0.0;

        // float Bdotu_ijk = 0.0, Bdotu_ip1jk = 0.0, Bdotu_ijp1k = 0.0, Bdotu_ijkp1 = 0.0;

        if (i < Nx-1 && j < Ny-1 && k < Nz-1){
            rho = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)];
            rhovx = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size];
            rhovy = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovz = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size];
            // Bx = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size];
            // By = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size];
            // Bz = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size];
            // e = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size];

            // rho_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz)];
            rhovx_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + cube_size];
            // rhovy_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 2 * cube_size];
            // rhovz_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 3 * cube_size];
            // Bx_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 4 * cube_size];
            // By_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 5 * cube_size];
            // Bz_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 6 * cube_size];
            // e_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 7 * cube_size];

            // rho_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz)];
            // rhovx_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + cube_size];
            rhovy_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 2 * cube_size];
            // rhovz_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 3 * cube_size];
            // Bx_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 4 * cube_size];
            // By_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 5 * cube_size];
            // Bz_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 6 * cube_size];
            // e_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 7 * cube_size];

            // rho_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz)];
            // rhovx_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + cube_size];
            // rhovy_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 3 * cube_size];
            // Bx_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 4 * cube_size];
            // By_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 5 * cube_size];
            // Bz_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 6 * cube_size];
            // e_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 7 * cube_size];

            intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 1 * cube_size] = intRhoLocal(rho, rhovx, rhovy, rhovz, 
                                                                    rhovx_ip1, rhovy_jp1, rhovz_kp1, 
                                                                    dt, dx, dy, dz, Nx, Ny, Nz);
        }
        return;
    }

// 64 registers per thread
__global__ void ComputeIntRhoVXMicroLocalNoDiff(const float* fluidvar, float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
     {
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;
        unsigned int k = threadIdx.z + blockDim.z * blockIdx.z;
        
        unsigned int cube_size = Nx * Ny * Nz;

        // DECLARE EVERYTHING 
        // float rho = 0.0, rho_ip1 = 0.0, rho_jp1 = 0.0, rho_kp1 = 0.0;
        float rho = 0.0, rho_ip1 = 0.0, rho_jp1 = 0.0, rho_kp1 = 0.0;

        // float rhovx = 0.0, rhovx_ip1 = 0.0, rhovx_jp1 = 0.0, rhovx_kp1 = 0.0;
        float rhovx = 0.0, rhovx_ip1 = 0.0, rhovx_jp1 = 0.0, rhovx_kp1 = 0.0;

        // float rhovy = 0.0, rhovy_ip1 = 0.0, rhovy_jp1 = 0.0, rhovy_kp1 = 0.0;
        float rhovy = 0.0, rhovy_ip1 = 0.0, rhovy_jp1 = 0.0;

        // float rhovz = 0.0, rhovz_ip1 = 0.0, rhovz_jp1 = 0.0, rhovz_kp1 = 0.0;
        float rhovz = 0.0, rhovz_ip1 = 0.0, rhovz_kp1 = 0.0;

        // float Bx = 0.0, Bx_ip1 = 0.0, Bx_jp1 = 0.0, Bx_kp1 = 0.0;
        float Bx = 0.0, Bx_ip1 = 0.0, Bx_jp1 = 0.0, Bx_kp1 = 0.0;

        // float By = 0.0, By_ip1 = 0.0, By_jp1 = 0.0, By_kp1 = 0.0;
        float By = 0.0, By_ip1 = 0.0, By_jp1 = 0.0;

        // float Bz = 0.0, Bz_ip1 = 0.0, Bz_jp1 = 0.0, Bz_kp1 = 0.0;
        float Bz = 0.0, Bz_ip1 = 0.0, Bz_kp1 = 0.0;

        // float e = 0.0, e_ip1 = 0.0, e_jp1 = 0.0, e_kp1 = 0.0;
        float e = 0.0, e_ip1 = 0.0;

        // float p_ijk = 0.0, p_ip1jk = 0.0, p_ijp1k = 0.0, p_ijkp1 = 0.0;
        float p_ijk = 0.0, p_ip1jk = 0.0;

        // float Bsq_ijk = 0.0, Bsq_ip1jk = 0.0, Bsq_ijp1k = 0.0, Bsq_ijkp1 = 0.0;
        float Bsq_ijk = 0.0, Bsq_ip1jk = 0.0;

        // float KE_ijk = 0.0, KE_ip1jk = 0.0, KE_ijp1k = 0.0, KE_ijkp1 = 0.0;
        float KE_ijk = 0.0, KE_ip1jk = 0.0;

        // float Bdotu_ijk = 0.0, Bdotu_ip1jk = 0.0, Bdotu_ijp1k = 0.0, Bdotu_ijkp1 = 0.0;

        if (i < Nx-1 && j < Ny-1 && k < Nz-1){
            rho = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)];
            rhovx = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size];
            rhovy = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovz = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size];
            By = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size];
            e = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size];

            rho_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz)];
            rhovx_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + cube_size];
            rhovy_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 4 * cube_size];
            By_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 6 * cube_size];
            e_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 7 * cube_size];

            rho_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz)];
            rhovx_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + cube_size];
            rhovy_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 2 * cube_size];
            // rhovz_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 4 * cube_size];
            By_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 5 * cube_size];
            // Bz_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 6 * cube_size];
            // e_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 7 * cube_size];

            rho_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz)];
            rhovx_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + cube_size];
            // rhovy_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 3 * cube_size];
            Bx_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 4 * cube_size];
            // By_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 5 * cube_size];
            Bz_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 6 * cube_size];
            // e_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 7 * cube_size];
            
            Bsq_ijk = B_sq_local(Bx, By, Bz);
            Bsq_ip1jk = B_sq_local(Bx_ip1, By_ip1, Bz_ip1);
            // Bsq_ijp1k = B_sq_local(Bx_jp1, By_jp1, Bz_jp1);
            // Bsq_ijkp1 = B_sq_local(Bx_kp1, By_kp1, Bz_kp1);

            KE_ijk = KE_local(rho, rhovx, rhovy, rhovz);
            KE_ip1jk = KE_local(rho_ip1, rhovx_ip1, rhovy_ip1, rhovz_ip1);
            // KE_ijp1k = KE_local(rho_jp1, rhovx_jp1, rhovy_jp1, rhovz_jp1);
            // KE_ijkp1 = KE_local(rho_kp1, rhovx_kp1, rhovy_kp1, rhovz_kp1);

            p_ijk = p_local(e, Bsq_ijk, KE_ijk);
            p_ip1jk = p_local(e_ip1, Bsq_ip1jk, KE_ip1jk);
            // p_ijp1k = p_local(e_jp1, Bsq_ijp1k, KE_ijp1k);
            // p_ijkp1 = p_local(e_kp1, Bsq_ijkp1, KE_ijkp1);            

            // Bdotu_ijk = B_dot_u_local(rho, rhovx, rhovy, rhovz, Bx, By, Bz);
            // Bdotu_ip1jk = B_dot_u_local(rho_ip1, rhovx_ip1, rhovy_ip1, rhovz_ip1, Bx_ip1, By_ip1, Bz_ip1);
            // Bdotu_ijp1k = B_dot_u_local(rho_jp1, rhovx_jp1, rhovy_jp1, rhovz_jp1, Bx_jp1, By_jp1, Bz_jp1);
            // Bdotu_ijkp1 = B_dot_u_local(rho_kp1, rhovx_kp1, rhovy_kp1, rhovz_kp1, Bx_kp1, By_kp1, Bz_kp1);

            intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 1 * cube_size] = intRhoVXLocal(rho, rho_ip1, rho_jp1, rho_kp1, 
                                                                    rhovx, rhovx_ip1, rhovx_jp1, rhovx_kp1, 
                                                                    rhovy, rhovy_jp1,  
                                                                    rhovz, rhovz_kp1, 
                                                                    Bx, Bx_ip1, Bx_jp1, Bx_kp1, By, By_jp1, Bz, Bz_kp1, 
                                                                    p_ip1jk, p_ijk, Bsq_ip1jk, Bsq_ijk,
                                                                    dt, dx, dy, dz, Nx, Ny, Nz);
        }
        return;
    }

// 64 registers per thread
__global__ void ComputeIntRhoVYMicroLocalNoDiff(const float* fluidvar, float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
     {
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;
        unsigned int k = threadIdx.z + blockDim.z * blockIdx.z;
        
        unsigned int cube_size = Nx * Ny * Nz;

        // DECLARE EVERYTHING 
        // float rho = 0.0, rho_ip1 = 0.0, rho_jp1 = 0.0, rho_kp1 = 0.0;
        float rho = 0.0, rho_ip1 = 0.0, rho_jp1 = 0.0, rho_kp1 = 0.0;

        // float rhovx = 0.0, rhovx_ip1 = 0.0, rhovx_jp1 = 0.0, rhovx_kp1 = 0.0;
        float rhovx = 0.0, rhovx_ip1 = 0.0, rhovx_jp1 = 0.0;

        // float rhovy = 0.0, rhovy_ip1 = 0.0, rhovy_jp1 = 0.0, rhovy_kp1 = 0.0;
        float rhovy = 0.0, rhovy_ip1 = 0.0, rhovy_jp1 = 0.0, rhovy_kp1 = 0.0;

        // float rhovz = 0.0, rhovz_ip1 = 0.0, rhovz_jp1 = 0.0, rhovz_kp1 = 0.0;
        float rhovz = 0.0, rhovz_jp1 = 0.0, rhovz_kp1 = 0.0;

        // float Bx = 0.0, Bx_ip1 = 0.0, Bx_jp1 = 0.0, Bx_kp1 = 0.0;
        float Bx = 0.0, Bx_ip1 = 0.0, Bx_jp1 = 0.0;

        // float By = 0.0, By_ip1 = 0.0, By_jp1 = 0.0, By_kp1 = 0.0;
        float By = 0.0, By_ip1 = 0.0, By_jp1 = 0.0, By_kp1 = 0.0;

        // float Bz = 0.0, Bz_ip1 = 0.0, Bz_jp1 = 0.0, Bz_kp1 = 0.0;
        float Bz = 0.0, Bz_jp1 = 0.0, Bz_kp1 = 0.0;

        // float e = 0.0, e_ip1 = 0.0, e_jp1 = 0.0, e_kp1 = 0.0;
        float e = 0.0, e_jp1 = 0.0;

        // float p_ijk = 0.0, p_ip1jk = 0.0, p_ijp1k = 0.0, p_ijkp1 = 0.0;
        float p_ijk = 0.0, p_ijp1k = 0.0;

        // float Bsq_ijk = 0.0, Bsq_ip1jk = 0.0, Bsq_ijp1k = 0.0, Bsq_ijkp1 = 0.0;
        float Bsq_ijk = 0.0, Bsq_ijp1k = 0.0;

        // float KE_ijk = 0.0, KE_ip1jk = 0.0, KE_ijp1k = 0.0, KE_ijkp1 = 0.0;
        float KE_ijk = 0.0, KE_ijp1k = 0.0;

        // float Bdotu_ijk = 0.0, Bdotu_ip1jk = 0.0, Bdotu_ijp1k = 0.0, Bdotu_ijkp1 = 0.0;

        if (i < Nx-1 && j < Ny-1 && k < Nz-1){
            rho = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)];
            rhovx = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size];
            rhovy = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovz = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size];
            By = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size];
            e = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size];

            rho_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz)];
            rhovx_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + cube_size];
            rhovy_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 2 * cube_size];
            // rhovz_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 4 * cube_size];
            By_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 5 * cube_size];
            // Bz_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 6 * cube_size];
            // e_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 7 * cube_size];

            rho_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz)];
            rhovx_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + cube_size];
            rhovy_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 4 * cube_size];
            By_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 6 * cube_size];
            e_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 7 * cube_size];

            rho_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz)];
            // rhovx_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + cube_size];
            rhovy_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 3 * cube_size];
            // Bx_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 4 * cube_size];
            By_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 5 * cube_size];
            Bz_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 6 * cube_size];
            // e_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 7 * cube_size];
            
            Bsq_ijk = B_sq_local(Bx, By, Bz);
            // Bsq_ip1jk = B_sq_local(Bx_ip1, By_ip1, Bz_ip1);
            Bsq_ijp1k = B_sq_local(Bx_jp1, By_jp1, Bz_jp1);
            // Bsq_ijkp1 = B_sq_local(Bx_kp1, By_kp1, Bz_kp1);

            KE_ijk = KE_local(rho, rhovx, rhovy, rhovz);
            // KE_ip1jk = KE_local(rho_ip1, rhovx_ip1, rhovy_ip1, rhovz_ip1);
            KE_ijp1k = KE_local(rho_jp1, rhovx_jp1, rhovy_jp1, rhovz_jp1);
            // KE_ijkp1 = KE_local(rho_kp1, rhovx_kp1, rhovy_kp1, rhovz_kp1);

            p_ijk = p_local(e, Bsq_ijk, KE_ijk);
            // p_ip1jk = p_local(e_ip1, Bsq_ip1jk, KE_ip1jk);
            p_ijp1k = p_local(e_jp1, Bsq_ijp1k, KE_ijp1k);
            // p_ijkp1 = p_local(e_kp1, Bsq_ijkp1, KE_ijkp1);            

            // Bdotu_ijk = B_dot_u_local(rho, rhovx, rhovy, rhovz, Bx, By, Bz);
            // Bdotu_ip1jk = B_dot_u_local(rho_ip1, rhovx_ip1, rhovy_ip1, rhovz_ip1, Bx_ip1, By_ip1, Bz_ip1);
            // Bdotu_ijp1k = B_dot_u_local(rho_jp1, rhovx_jp1, rhovy_jp1, rhovz_jp1, Bx_jp1, By_jp1, Bz_jp1);
            // Bdotu_ijkp1 = B_dot_u_local(rho_kp1, rhovx_kp1, rhovy_kp1, rhovz_kp1, Bx_kp1, By_kp1, Bz_kp1);

            intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYLocal(rho, rho_ip1, rho_jp1, rho_kp1, 
                                                                    rhovx, rhovx_ip1, rhovy, rhovy_ip1, rhovy_jp1, rhovy_kp1, rhovz, rhovz_kp1,
                                                                    Bx, Bx_ip1, By, By_ip1, By_jp1, By_kp1, Bz, Bz_kp1,
                                                                    p_ijk, p_ijp1k, Bsq_ijk, Bsq_ijp1k, 
                                                                    dt, dx, dy, dz, Nx, Ny, Nz);
        }
        return;
    }

// 64 registers per thread
__global__ void ComputeIntRhoVZMicroLocalNoDiff(const float* fluidvar, float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
     {
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;
        unsigned int k = threadIdx.z + blockDim.z * blockIdx.z;
        
        unsigned int cube_size = Nx * Ny * Nz;

        // DECLARE EVERYTHING 
        // float rho = 0.0, rho_ip1 = 0.0, rho_jp1 = 0.0, rho_kp1 = 0.0;
        float rho = 0.0, rho_ip1 = 0.0, rho_jp1 = 0.0, rho_kp1 = 0.0;

        // float rhovx = 0.0, rhovx_ip1 = 0.0, rhovx_jp1 = 0.0, rhovx_kp1 = 0.0;
        float rhovx = 0.0, rhovx_ip1 = 0.0, rhovx_kp1 = 0.0;

        // float rhovy = 0.0, rhovy_ip1 = 0.0, rhovy_jp1 = 0.0, rhovy_kp1 = 0.0;
        float rhovy = 0.0, rhovy_jp1 = 0.0, rhovy_kp1 = 0.0;

        // float rhovz = 0.0, rhovz_ip1 = 0.0, rhovz_jp1 = 0.0, rhovz_kp1 = 0.0;
        float rhovz = 0.0, rhovz_ip1 = 0.0, rhovz_jp1 = 0.0, rhovz_kp1 = 0.0;

        // float Bx = 0.0, Bx_ip1 = 0.0, Bx_jp1 = 0.0, Bx_kp1 = 0.0;
        float Bx = 0.0, Bx_ip1 = 0.0, Bx_kp1 = 0.0;

        // float By = 0.0, By_ip1 = 0.0, By_jp1 = 0.0, By_kp1 = 0.0;
        float By = 0.0, By_jp1 = 0.0, By_kp1 = 0.0;

        // float Bz = 0.0, Bz_ip1 = 0.0, Bz_jp1 = 0.0, Bz_kp1 = 0.0;
        float Bz = 0.0, Bz_ip1 = 0.0, Bz_jp1 = 0.0, Bz_kp1 = 0.0;

        // float e = 0.0, e_ip1 = 0.0, e_jp1 = 0.0, e_kp1 = 0.0;
        float e = 0.0, e_kp1 = 0.0;

        // float p_ijk = 0.0, p_ip1jk = 0.0, p_ijp1k = 0.0, p_ijkp1 = 0.0;
        float p_ijk = 0.0, p_ijkp1 = 0.0;

        // float Bsq_ijk = 0.0, Bsq_ip1jk = 0.0, Bsq_ijp1k = 0.0, Bsq_ijkp1 = 0.0;
        float Bsq_ijk = 0.0, Bsq_ijkp1 = 0.0;

        // float KE_ijk = 0.0, KE_ip1jk = 0.0, KE_ijp1k = 0.0, KE_ijkp1 = 0.0;
        float KE_ijk = 0.0, KE_ijkp1 = 0.0;

        // float Bdotu_ijk = 0.0, Bdotu_ip1jk = 0.0, Bdotu_ijp1k = 0.0, Bdotu_ijkp1 = 0.0;

        if (i < Nx-1 && j < Ny-1 && k < Nz-1){
            rho = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)];
            rhovx = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size];
            rhovy = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovz = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size];
            By = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size];
            e = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size];

            rho_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz)];
            rhovx_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + cube_size];
            // rhovy_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 4 * cube_size];
            // By_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 6 * cube_size];
            // e_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 7 * cube_size];

            rho_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz)];
            // rhovx_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + cube_size];
            rhovy_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 3 * cube_size];
            // Bx_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 4 * cube_size];
            By_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 6 * cube_size];
            // e_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 7 * cube_size];

            rho_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz)];
            rhovx_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + cube_size];
            rhovy_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 3 * cube_size];
            Bx_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 4 * cube_size];
            By_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 5 * cube_size];
            Bz_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 6 * cube_size];
            e_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 7 * cube_size];
            
            Bsq_ijk = B_sq_local(Bx, By, Bz);
            // Bsq_ip1jk = B_sq_local(Bx_ip1, By_ip1, Bz_ip1);
            // Bsq_ijp1k = B_sq_local(Bx_jp1, By_jp1, Bz_jp1);
            Bsq_ijkp1 = B_sq_local(Bx_kp1, By_kp1, Bz_kp1);

            KE_ijk = KE_local(rho, rhovx, rhovy, rhovz);
            // KE_ip1jk = KE_local(rho_ip1, rhovx_ip1, rhovy_ip1, rhovz_ip1);
            // KE_ijp1k = KE_local(rho_jp1, rhovx_jp1, rhovy_jp1, rhovz_jp1);
            KE_ijkp1 = KE_local(rho_kp1, rhovx_kp1, rhovy_kp1, rhovz_kp1);

            p_ijk = p_local(e, Bsq_ijk, KE_ijk);
            // p_ip1jk = p_local(e_ip1, Bsq_ip1jk, KE_ip1jk);
            // p_ijp1k = p_local(e_jp1, Bsq_ijp1k, KE_ijp1k);
            p_ijkp1 = p_local(e_kp1, Bsq_ijkp1, KE_ijkp1);            

            // Bdotu_ijk = B_dot_u_local(rho, rhovx, rhovy, rhovz, Bx, By, Bz);
            // Bdotu_ip1jk = B_dot_u_local(rho_ip1, rhovx_ip1, rhovy_ip1, rhovz_ip1, Bx_ip1, By_ip1, Bz_ip1);
            // Bdotu_ijp1k = B_dot_u_local(rho_jp1, rhovx_jp1, rhovy_jp1, rhovz_jp1, Bx_jp1, By_jp1, Bz_jp1);
            // Bdotu_ijkp1 = B_dot_u_local(rho_kp1, rhovx_kp1, rhovy_kp1, rhovz_kp1, Bx_kp1, By_kp1, Bz_kp1);

            intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZLocal(rho, rho_ip1, rho_jp1, rho_kp1,
                                                                    rhovx, rhovx_ip1, rhovy, rhovy_jp1, rhovz, rhovz_ip1, rhovz_jp1, rhovz_kp1, 
                                                                    Bx, Bx_ip1, By, By_jp1, Bz, Bz_ip1, Bz_jp1, Bz_kp1, 
                                                                    p_ijk, p_ijkp1, Bsq_ijk, Bsq_ijkp1, 
                                                                    dt, dx, dy, dz, Nx, Ny, Nz);
        }
        return;
    }

// 48 registers per thread
__global__ void ComputeIntBXMicroLocalNoDiff(const float* fluidvar, float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;
        unsigned int k = threadIdx.z + blockDim.z * blockIdx.z;
        
        unsigned int cube_size = Nx * Ny * Nz;

        // DECLARE EVERYTHING 
        // float rho = 0.0, rho_ip1 = 0.0, rho_jp1 = 0.0, rho_kp1 = 0.0;
        float rho = 0.0, rho_jp1 = 0.0, rho_kp1 = 0.0;

        // float rhovx = 0.0, rhovx_ip1 = 0.0, rhovx_jp1 = 0.0, rhovx_kp1 = 0.0;
        float rhovx = 0.0, rhovx_jp1 = 0.0, rhovx_kp1 = 0.0;

        // float rhovy = 0.0, rhovy_ip1 = 0.0, rhovy_jp1 = 0.0, rhovy_kp1 = 0.0;
        float rhovy = 0.0, rhovy_jp1 = 0.0, rhovy_kp1 = 0.0;

        // float rhovz = 0.0, rhovz_ip1 = 0.0, rhovz_jp1 = 0.0, rhovz_kp1 = 0.0;
        float rhovz = 0.0, rhovz_kp1 = 0.0;

        // float Bx = 0.0, Bx_ip1 = 0.0, Bx_jp1 = 0.0, Bx_kp1 = 0.0;
        float Bx = 0.0, Bx_jp1 = 0.0, Bx_kp1 = 0.0;

        // float By = 0.0, By_ip1 = 0.0, By_jp1 = 0.0, By_kp1 = 0.0;
        float By = 0.0, By_jp1 = 0.0;

        // float Bz = 0.0, Bz_ip1 = 0.0, Bz_jp1 = 0.0, Bz_kp1 = 0.0;
        float Bz = 0.0, Bz_kp1 = 0.0;

        // float e = 0.0, e_ip1 = 0.0, e_jp1 = 0.0, e_kp1 = 0.0;

        // float p_ijk = 0.0, p_ip1jk = 0.0, p_ijp1k = 0.0, p_ijkp1 = 0.0;

        // float Bsq_ijk = 0.0, Bsq_ip1jk = 0.0, Bsq_ijp1k = 0.0, Bsq_ijkp1 = 0.0;

        // float KE_ijk = 0.0, KE_ip1jk = 0.0, KE_ijp1k = 0.0, KE_ijkp1 = 0.0;

        // float Bdotu_ijk = 0.0, Bdotu_ip1jk = 0.0, Bdotu_ijp1k = 0.0, Bdotu_ijkp1 = 0.0;

        if (i < Nx-1 && j < Ny-1 && k < Nz-1){
            rho = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)];
            rhovx = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size];
            rhovy = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovz = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size];
            By = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size];
            // e = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size];

            // rho_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz)];
            // rhovx_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + cube_size];
            // rhovy_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 2 * cube_size];
            // rhovz_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 3 * cube_size];
            // Bx_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 4 * cube_size];
            // By_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 5 * cube_size];
            // Bz_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 6 * cube_size];
            // e_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 7 * cube_size];

            rho_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz)];
            rhovx_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + cube_size];
            rhovy_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 2 * cube_size];
            // rhovz_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 4 * cube_size];
            By_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 5 * cube_size];
            // Bz_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 6 * cube_size];
            // e_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 7 * cube_size];

            rho_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz)];
            rhovx_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + cube_size];
            // rhovy_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 3 * cube_size];
            Bx_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 4 * cube_size];
            // By_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 5 * cube_size];
            Bz_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 6 * cube_size];
            // e_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 7 * cube_size];
            
            // Bsq_ijk = B_sq_local(Bx, By, Bz);
            // Bsq_ip1jk = B_sq_local(Bx_ip1, By_ip1, Bz_ip1);
            // Bsq_ijp1k = B_sq_local(Bx_jp1, By_jp1, Bz_jp1);
            // Bsq_ijkp1 = B_sq_local(Bx_kp1, By_kp1, Bz_kp1);

            // KE_ijk = KE_local(rho, rhovx, rhovy, rhovz);
            // KE_ip1jk = KE_local(rho_ip1, rhovx_ip1, rhovy_ip1, rhovz_ip1);
            // KE_ijp1k = KE_local(rho_jp1, rhovx_jp1, rhovy_jp1, rhovz_jp1);
            // KE_ijkp1 = KE_local(rho_kp1, rhovx_kp1, rhovy_kp1, rhovz_kp1);

            // p_ijk = p_local(e, Bsq_ijk, KE_ijk);
            // p_ip1jk = p_local(e_ip1, Bsq_ip1jk, KE_ip1jk);
            // p_ijp1k = p_local(e_jp1, Bsq_ijp1k, KE_ijp1k);
            // p_ijkp1 = p_local(e_kp1, Bsq_ijkp1, KE_ijkp1);            

            // Bdotu_ijk = B_dot_u_local(rho, rhovx, rhovy, rhovz, Bx, By, Bz);
            // Bdotu_ip1jk = B_dot_u_local(rho_ip1, rhovx_ip1, rhovy_ip1, rhovz_ip1, Bx_ip1, By_ip1, Bz_ip1);
            // Bdotu_ijp1k = B_dot_u_local(rho_jp1, rhovx_jp1, rhovy_jp1, rhovz_jp1, Bx_jp1, By_jp1, Bz_jp1);
            // Bdotu_ijkp1 = B_dot_u_local(rho_kp1, rhovx_kp1, rhovy_kp1, rhovz_kp1, Bx_kp1, By_kp1, Bz_kp1);

            intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = intBXLocal(rho, rho_jp1, rho_kp1, 
                                                                    rhovx, rhovx_jp1, rhovx_kp1, rhovy, rhovy_jp1, rhovz, rhovz_kp1,
                                                                    Bx, Bx_jp1, Bx_kp1, By, By_jp1, Bz, Bz_kp1, 
                                                                    dt, dx, dy, dz, Nx, Ny, Nz);
        }
        return;
    }

// 36 registers per thread
__global__ void ComputeIntBYMicroLocalNoDiff(const float* fluidvar, float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;
        unsigned int k = threadIdx.z + blockDim.z * blockIdx.z;
        
        unsigned int cube_size = Nx * Ny * Nz;

        // DECLARE EVERYTHING 
        // float rho = 0.0, rho_ip1 = 0.0, rho_jp1 = 0.0, rho_kp1 = 0.0;
        float rho = 0.0, rho_ip1 = 0.0, rho_kp1 = 0.0;

        // float rhovx = 0.0, rhovx_ip1 = 0.0, rhovx_jp1 = 0.0, rhovx_kp1 = 0.0;
        float rhovx = 0.0, rhovx_ip1 = 0.0;

        // float rhovy = 0.0, rhovy_ip1 = 0.0, rhovy_jp1 = 0.0, rhovy_kp1 = 0.0;
        float rhovy = 0.0, rhovy_ip1 = 0.0, rhovy_kp1 = 0.0;

        // float rhovz = 0.0, rhovz_ip1 = 0.0, rhovz_jp1 = 0.0, rhovz_kp1 = 0.0;
        float rhovz = 0.0, rhovz_kp1 = 0.0;

        // float Bx = 0.0, Bx_ip1 = 0.0, Bx_jp1 = 0.0, Bx_kp1 = 0.0;
        float Bx = 0.0, Bx_ip1 = 0.0;

        // float By = 0.0, By_ip1 = 0.0, By_jp1 = 0.0, By_kp1 = 0.0;
        float By = 0.0, By_ip1 = 0.0, By_jp1 = 0.0, By_kp1 = 0.0;

        // float Bz = 0.0, Bz_ip1 = 0.0, Bz_jp1 = 0.0, Bz_kp1 = 0.0;
        float Bz = 0.0, Bz_kp1 = 0.0;

        // float e = 0.0, e_ip1 = 0.0, e_jp1 = 0.0, e_kp1 = 0.0;

        // float p_ijk = 0.0, p_ip1jk = 0.0, p_ijp1k = 0.0, p_ijkp1 = 0.0;

        // float Bsq_ijk = 0.0, Bsq_ip1jk = 0.0, Bsq_ijp1k = 0.0, Bsq_ijkp1 = 0.0;

        // float KE_ijk = 0.0, KE_ip1jk = 0.0, KE_ijp1k = 0.0, KE_ijkp1 = 0.0;

        // float Bdotu_ijk = 0.0, Bdotu_ip1jk = 0.0, Bdotu_ijp1k = 0.0, Bdotu_ijkp1 = 0.0;

        if (i < Nx-1 && j < Ny-1 && k < Nz-1){
            rho = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)];
            rhovx = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size];
            rhovy = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovz = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size];
            By = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size];
            // e = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size];

            rho_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz)];
            rhovx_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + cube_size];
            rhovy_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 2 * cube_size];
            // rhovz_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 4 * cube_size];
            By_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 5 * cube_size];
            // Bz_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 6 * cube_size];
            // e_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 7 * cube_size];

            // rho_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz)];
            // rhovx_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + cube_size];
            // rhovy_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 2 * cube_size];
            // rhovz_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 3 * cube_size];
            // Bx_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 4 * cube_size];
            By_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 5 * cube_size];
            // Bz_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 6 * cube_size];
            // e_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 7 * cube_size];

            rho_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz)];
            // rhovx_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + cube_size];
            rhovy_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 3 * cube_size];
            // Bx_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 4 * cube_size];
            By_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 5 * cube_size];
            Bz_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 6 * cube_size];
            // e_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 7 * cube_size];
            
            // Bsq_ijk = B_sq_local(Bx, By, Bz);
            // Bsq_ip1jk = B_sq_local(Bx_ip1, By_ip1, Bz_ip1);
            // Bsq_ijp1k = B_sq_local(Bx_jp1, By_jp1, Bz_jp1);
            // Bsq_ijkp1 = B_sq_local(Bx_kp1, By_kp1, Bz_kp1);

            // KE_ijk = KE_local(rho, rhovx, rhovy, rhovz);
            // KE_ip1jk = KE_local(rho_ip1, rhovx_ip1, rhovy_ip1, rhovz_ip1);
            // KE_ijp1k = KE_local(rho_jp1, rhovx_jp1, rhovy_jp1, rhovz_jp1);
            // KE_ijkp1 = KE_local(rho_kp1, rhovx_kp1, rhovy_kp1, rhovz_kp1);

            // p_ijk = p_local(e, Bsq_ijk, KE_ijk);
            // p_ip1jk = p_local(e_ip1, Bsq_ip1jk, KE_ip1jk);
            // p_ijp1k = p_local(e_jp1, Bsq_ijp1k, KE_ijp1k);
            // p_ijkp1 = p_local(e_kp1, Bsq_ijkp1, KE_ijkp1);            

            // Bdotu_ijk = B_dot_u_local(rho, rhovx, rhovy, rhovz, Bx, By, Bz);
            // Bdotu_ip1jk = B_dot_u_local(rho_ip1, rhovx_ip1, rhovy_ip1, rhovz_ip1, Bx_ip1, By_ip1, Bz_ip1);
            // Bdotu_ijp1k = B_dot_u_local(rho_jp1, rhovx_jp1, rhovy_jp1, rhovz_jp1, Bx_jp1, By_jp1, Bz_jp1);
            // Bdotu_ijkp1 = B_dot_u_local(rho_kp1, rhovx_kp1, rhovy_kp1, rhovz_kp1, Bx_kp1, By_kp1, Bz_kp1);

            intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = intBYLocal(rho, rho_ip1, rho_kp1, 
                                                                    rhovx, rhovx_ip1, rhovy, rhovy_ip1, rhovy_kp1, rhovz, rhovz_kp1, 
                                                                    Bx, Bx_ip1, By, By_ip1, By_kp1, Bz, Bz_kp1,
                                                                    dt, dx, dy, dz, Nx, Ny, Nz);
        }
        return;
    }

// 38 registers per thread
__global__ void ComputeIntBZMicroLocalNoDiff(const float* fluidvar, float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;
        unsigned int k = threadIdx.z + blockDim.z * blockIdx.z;
        
        unsigned int cube_size = Nx * Ny * Nz;

        // DECLARE EVERYTHING 
        // float rho = 0.0, rho_ip1 = 0.0, rho_jp1 = 0.0, rho_kp1 = 0.0;
        float rho = 0.0, rho_ip1 = 0.0, rho_jp1 = 0.0, rho_kp1 = 0.0;

        // float rhovx = 0.0, rhovx_ip1 = 0.0, rhovx_jp1 = 0.0, rhovx_kp1 = 0.0;
        float rhovx = 0.0, rhovx_ip1 = 0.0;

        // float rhovy = 0.0, rhovy_ip1 = 0.0, rhovy_jp1 = 0.0, rhovy_kp1 = 0.0;
        float rhovy = 0.0, rhovy_jp1 = 0.0;

        // float rhovz = 0.0, rhovz_ip1 = 0.0, rhovz_jp1 = 0.0, rhovz_kp1 = 0.0;
        float rhovz = 0.0, rhovz_ip1 = 0.0;

        // float Bx = 0.0, Bx_ip1 = 0.0, Bx_jp1 = 0.0, Bx_kp1 = 0.0;
        float Bx = 0.0, Bx_ip1 = 0.0;

        // float By = 0.0, By_ip1 = 0.0, By_jp1 = 0.0, By_kp1 = 0.0;
        float By = 0.0, By_jp1 = 0.0;

        // float Bz = 0.0, Bz_ip1 = 0.0, Bz_jp1 = 0.0, Bz_kp1 = 0.0;
        float Bz = 0.0, Bz_ip1 = 0.0, Bz_jp1 = 0.0;

        // float e = 0.0, e_ip1 = 0.0, e_jp1 = 0.0, e_kp1 = 0.0;

        // float p_ijk = 0.0, p_ip1jk = 0.0, p_ijp1k = 0.0, p_ijkp1 = 0.0;

        // float Bsq_ijk = 0.0, Bsq_ip1jk = 0.0, Bsq_ijp1k = 0.0, Bsq_ijkp1 = 0.0;

        // float KE_ijk = 0.0, KE_ip1jk = 0.0, KE_ijp1k = 0.0, KE_ijkp1 = 0.0;

        // float Bdotu_ijk = 0.0, Bdotu_ip1jk = 0.0, Bdotu_ijp1k = 0.0, Bdotu_ijkp1 = 0.0;

        if (i < Nx-1 && j < Ny-1 && k < Nz-1){
            rho = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)];
            rhovx = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size];
            rhovy = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovz = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size];
            By = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size];
            // e = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size];

            rho_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz)];
            rhovx_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + cube_size];
            // rhovy_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 4 * cube_size];
            // By_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 6 * cube_size];
            // e_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 7 * cube_size];

            rho_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz)];
            // rhovx_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + cube_size];
            rhovy_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 2 * cube_size];
            // rhovz_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 3 * cube_size];
            // Bx_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 4 * cube_size];
            By_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 6 * cube_size];
            // e_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 7 * cube_size];

            rho_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz)];
            // rhovx_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + cube_size];
            // rhovy_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 2 * cube_size];
            // rhovz_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 3 * cube_size];
            // Bx_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 4 * cube_size];
            // By_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 5 * cube_size];
            // Bz_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 6 * cube_size];
            // e_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 7 * cube_size];
            
            // Bsq_ijk = B_sq_local(Bx, By, Bz);
            // Bsq_ip1jk = B_sq_local(Bx_ip1, By_ip1, Bz_ip1);
            // Bsq_ijp1k = B_sq_local(Bx_jp1, By_jp1, Bz_jp1);
            // Bsq_ijkp1 = B_sq_local(Bx_kp1, By_kp1, Bz_kp1);

            // KE_ijk = KE_local(rho, rhovx, rhovy, rhovz);
            // KE_ip1jk = KE_local(rho_ip1, rhovx_ip1, rhovy_ip1, rhovz_ip1);
            // KE_ijp1k = KE_local(rho_jp1, rhovx_jp1, rhovy_jp1, rhovz_jp1);
            // KE_ijkp1 = KE_local(rho_kp1, rhovx_kp1, rhovy_kp1, rhovz_kp1);

            // p_ijk = p_local(e, Bsq_ijk, KE_ijk);
            // p_ip1jk = p_local(e_ip1, Bsq_ip1jk, KE_ip1jk);
            // p_ijp1k = p_local(e_jp1, Bsq_ijp1k, KE_ijp1k);
            // p_ijkp1 = p_local(e_kp1, Bsq_ijkp1, KE_ijkp1);            

            // Bdotu_ijk = B_dot_u_local(rho, rhovx, rhovy, rhovz, Bx, By, Bz);
            // Bdotu_ip1jk = B_dot_u_local(rho_ip1, rhovx_ip1, rhovy_ip1, rhovz_ip1, Bx_ip1, By_ip1, Bz_ip1);
            // Bdotu_ijp1k = B_dot_u_local(rho_jp1, rhovx_jp1, rhovy_jp1, rhovz_jp1, Bx_jp1, By_jp1, Bz_jp1);
            // Bdotu_ijkp1 = B_dot_u_local(rho_kp1, rhovx_kp1, rhovy_kp1, rhovz_kp1, Bx_kp1, By_kp1, Bz_kp1);

            intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = intBZLocal(rho, rho_ip1, rho_jp1, rho_kp1,
                                                                    rhovx, rhovx_ip1, rhovy, rhovy_jp1, rhovz, rhovz_ip1,
                                                                    Bx, Bx_ip1, By, By_jp1, Bz, Bz_ip1, Bz_jp1,
                                                                    dt, dx, dy, dz, Nx, Ny, Nz);
        }
        return;
    }

// 90 registers per thread
__global__ void ComputeIntEMicroLocalNoDiff(const float* fluidvar, float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;
        unsigned int k = threadIdx.z + blockDim.z * blockIdx.z;
        
        unsigned int cube_size = Nx * Ny * Nz;

        // DECLARE EVERYTHING 
        float rho = 0.0, rho_ip1 = 0.0, rho_jp1 = 0.0, rho_kp1 = 0.0;

        float rhovx = 0.0, rhovx_ip1 = 0.0, rhovx_jp1 = 0.0, rhovx_kp1 = 0.0;

        float rhovy = 0.0, rhovy_ip1 = 0.0, rhovy_jp1 = 0.0, rhovy_kp1 = 0.0;

        float rhovz = 0.0, rhovz_ip1 = 0.0, rhovz_jp1 = 0.0, rhovz_kp1 = 0.0;

        float Bx = 0.0, Bx_ip1 = 0.0, Bx_jp1 = 0.0, Bx_kp1 = 0.0;

        float By = 0.0, By_ip1 = 0.0, By_jp1 = 0.0, By_kp1 = 0.0;

        float Bz = 0.0, Bz_ip1 = 0.0, Bz_jp1 = 0.0, Bz_kp1 = 0.0;

        float e = 0.0, e_ip1 = 0.0, e_jp1 = 0.0, e_kp1 = 0.0;

        float p_ijk = 0.0, p_ip1jk = 0.0, p_ijp1k = 0.0, p_ijkp1 = 0.0;

        float Bsq_ijk = 0.0, Bsq_ip1jk = 0.0, Bsq_ijp1k = 0.0, Bsq_ijkp1 = 0.0;

        float KE_ijk = 0.0, KE_ip1jk = 0.0, KE_ijp1k = 0.0, KE_ijkp1 = 0.0;

        float Bdotu_ijk = 0.0, Bdotu_ip1jk = 0.0, Bdotu_ijp1k = 0.0, Bdotu_ijkp1 = 0.0;

        if (i < Nx-1 && j < Ny-1 && k < Nz-1){
            rho = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)];
            rhovx = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size];
            rhovy = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovz = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size];
            By = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size];
            e = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size];

            rho_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz)];
            rhovx_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + cube_size];
            rhovy_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 4 * cube_size];
            By_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 6 * cube_size];
            e_ip1 = fluidvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 7 * cube_size];

            rho_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz)];
            rhovx_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + cube_size];
            rhovy_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 4 * cube_size];
            By_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 6 * cube_size];
            e_jp1 = fluidvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 7 * cube_size];

            rho_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz)];
            rhovx_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + cube_size];
            rhovy_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 3 * cube_size];
            Bx_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 4 * cube_size];
            By_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 5 * cube_size];
            Bz_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 6 * cube_size];
            e_kp1 = fluidvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 7 * cube_size];
            
            Bsq_ijk = B_sq_local(Bx, By, Bz);
            Bsq_ip1jk = B_sq_local(Bx_ip1, By_ip1, Bz_ip1);
            Bsq_ijp1k = B_sq_local(Bx_jp1, By_jp1, Bz_jp1);
            Bsq_ijkp1 = B_sq_local(Bx_kp1, By_kp1, Bz_kp1);

            KE_ijk = KE_local(rho, rhovx, rhovy, rhovz);
            KE_ip1jk = KE_local(rho_ip1, rhovx_ip1, rhovy_ip1, rhovz_ip1);
            KE_ijp1k = KE_local(rho_jp1, rhovx_jp1, rhovy_jp1, rhovz_jp1);
            KE_ijkp1 = KE_local(rho_kp1, rhovx_kp1, rhovy_kp1, rhovz_kp1);

            p_ijk = p_local(e, Bsq_ijk, KE_ijk);
            p_ip1jk = p_local(e_ip1, Bsq_ip1jk, KE_ip1jk);
            p_ijp1k = p_local(e_jp1, Bsq_ijp1k, KE_ijp1k);
            p_ijkp1 = p_local(e_kp1, Bsq_ijkp1, KE_ijkp1);            

            Bdotu_ijk = B_dot_u_local(rho, rhovx, rhovy, rhovz, Bx, By, Bz);
            Bdotu_ip1jk = B_dot_u_local(rho_ip1, rhovx_ip1, rhovy_ip1, rhovz_ip1, Bx_ip1, By_ip1, Bz_ip1);
            Bdotu_ijp1k = B_dot_u_local(rho_jp1, rhovx_jp1, rhovy_jp1, rhovz_jp1, Bx_jp1, By_jp1, Bz_jp1);
            Bdotu_ijkp1 = B_dot_u_local(rho_kp1, rhovx_kp1, rhovy_kp1, rhovz_kp1, Bx_kp1, By_kp1, Bz_kp1);

            intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = intELocal(rho, rho_ip1, rho_jp1, rho_kp1,
                                                                    rhovx, rhovx_ip1, rhovy, rhovy_jp1, rhovz, rhovz_kp1,
                                                                    Bx, Bx_ip1, By, By_jp1, Bz, Bz_kp1,
                                                                    e, e_ip1, e_jp1, e_kp1, 
                                                                    p_ijk, p_ip1jk, p_ijp1k, p_ijkp1,
                                                                    Bsq_ijk, Bsq_ip1jk, Bsq_ijp1k, Bsq_ijkp1, 
                                                                    Bdotu_ijk, Bdotu_ip1jk, Bdotu_ijp1k, Bdotu_ijkp1,
                                                                    dt, dx, dy, dz, Nx, Ny, Nz);
        }
        return;
    }

// Device kernels
// These kernels implement the Predictor step on the interior of the computational volume: {[0, Nx-1), [0, Ny-1), [0, Nz-1)}
// Accept local variables => high amount of register pressure, low number of memory accesses
__device__ float intRhoLocal(const float rho, 
    const float rhovx, const float rhovy, const float rhovz,
    const float rhovx_ip1, const float rhovy_jp1, const float rhovz_kp1,
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
    {
        return rho
            - (dt / dx) * (XFluxRho(rhovx_ip1) - XFluxRho(rhovx))
            - (dt / dy) * (YFluxRho(rhovy_jp1) - YFluxRho(rhovy))
            - (dt / dz) * (ZFluxRho(rhovz_kp1) - ZFluxRho(rhovz));
    }

__device__ float intRhoVXLocal(const float rho, const float rho_ip1, const float rho_jp1, const float rho_kp1,
    const float rhovx, const float rhovx_ip1, const float rhovx_jp1, const float rhovx_kp1, 
    const float rhovy, const float rhovy_jp1, 
    const float rhovz, const float rhovz_kp1,
    const float Bx, const float Bx_ip1, const float Bx_jp1, const float Bx_kp1,
    const float By, const float By_jp1, 
    const float Bz, const float Bz_kp1,
    const float p_ip1jk, const float p_ijk,
    const float Bsq_ip1jk, const float Bsq_ijk,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {   
        int cube_size = Nx * Ny * Nz;
        return rhovx
        - (dt / dx) * (XFluxRhoVX(rho_ip1, rhovx_ip1, Bx_ip1, p_ip1jk, Bsq_ip1jk) - XFluxRhoVX(rho, rhovx, Bx, p_ijk, Bsq_ijk))
        - (dt / dy) * (YFluxRhoVX(rho_jp1, rhovx_jp1, rhovy_jp1, Bx_jp1, By_jp1) - YFluxRhoVX(rho, rhovx, rhovy, Bx, By))
        - (dt / dz) * (ZFluxRhoVX(rho_kp1, rhovx_kp1, rhovz_kp1, Bx_kp1, Bz_kp1) - ZFluxRhoVX(rho, rhovx, rhovz, Bx, Bz));
    }

__device__ float intRhoVYLocal(const float rho, const float rho_ip1, const float rho_jp1, const float rho_kp1,
    const float rhovx, const float rhovx_ip1, 
    const float rhovy, const float rhovy_ip1, const float rhovy_jp1, const float rhovy_kp1,
    const float rhovz, const float rhovz_kp1,
    const float Bx, const float Bx_ip1,
    const float By, const float By_ip1, const float By_jp1, const float By_kp1,
    const float Bz, const float Bz_kp1,
    const float p_ijk, const float p_ijp1k,
    const float Bsq_ijk, const float Bsq_ijp1k,  
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return rhovy
        - (dt / dx) * (XFluxRhoVY(rho_ip1, rhovx_ip1, rhovy_ip1, Bx_ip1, By_ip1) - XFluxRhoVY(rho, rhovx, rhovy, Bx, By))
        - (dt / dy) * (YFluxRhoVY(rho_jp1, rhovy_jp1, By_jp1, p_ijp1k, Bsq_ijp1k) - YFluxRhoVY(rho, rhovy, By, p_ijk, Bsq_ijk))
        - (dt / dz) * (ZFluxRhoVY(rho_kp1, rhovy_kp1, rhovz_kp1, By_kp1, Bz_kp1) - ZFluxRhoVY(rho, rhovy, rhovz, By, Bz));
    }

__device__ float intRhoVZLocal(const float rho, const float rho_ip1, const float rho_jp1, const float rho_kp1,
    const float rhovx, const float rhovx_ip1, 
    const float rhovy, const float rhovy_jp1, 
    const float rhovz, const float rhovz_ip1, const float rhovz_jp1, const float rhovz_kp1,
    const float Bx, const float Bx_ip1, 
    const float By, const float By_jp1,
    const float Bz, const float Bz_ip1, const float Bz_jp1, const float Bz_kp1,
    const float p_ijk, const float p_ijkp1, 
    const float Bsq_ijk, const float Bsq_ijkp1,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return rhovz
        - (dt / dx) * (XFluxRhoVZ(rho_ip1, rhovx_ip1, rhovz_ip1, Bx_ip1, Bz_ip1) - XFluxRhoVZ(rho, rhovx, rhovz, Bx, Bz))
        - (dt / dy) * (YFluxRhoVZ(rho_jp1, rhovy_jp1, rhovz_jp1, By_jp1, Bz_jp1) - YFluxRhoVZ(rho, rhovy, rhovz, By, Bz))
        - (dt / dz) * (ZFluxRhoVZ(rho_kp1, rhovz_kp1, Bz_kp1, p_ijkp1, Bsq_ijkp1) - ZFluxRhoVZ(rho, rhovz, Bz, p_ijk, Bsq_ijk)); 
    }

__device__ float intBXLocal(const float rho, const float rho_jp1, const float rho_kp1,
    const float rhovx, const float rhovx_jp1, const float rhovx_kp1, 
    const float rhovy, const float rhovy_jp1,
    const float rhovz, const float rhovz_kp1, 
    const float Bx, const float Bx_jp1, const float Bx_kp1,
    const float By, const float By_jp1, 
    const float Bz, const float Bz_kp1,
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return Bx
        - (dt / dx) * (XFluxBX() - XFluxBX())
        - (dt / dy) * (YFluxBX(rho_jp1, rhovx_jp1, rhovy_jp1, Bx_jp1, By_jp1) - YFluxBX(rho, rhovx, rhovy, Bx, By))
        - (dt / dz) * (ZFluxBX(rho_kp1, rhovx_kp1, rhovz_kp1, Bx_kp1, Bz_kp1) - ZFluxBX(rho, rhovx, rhovz, Bx, Bz));
    }

__device__ float intBYLocal(const float rho, const float rho_ip1, const float rho_kp1,
    const float rhovx, const float rhovx_ip1, 
    const float rhovy, const float rhovy_ip1, const float rhovy_kp1, 
    const float rhovz, const float rhovz_kp1, 
    const float Bx, const float Bx_ip1, 
    const float By, const float By_ip1, const float By_kp1,
    const float Bz, const float Bz_kp1,  
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return By
        - (dt / dx) * (XFluxBY(rho_ip1, rhovx_ip1, rhovy_ip1, Bx_ip1, By_ip1) - XFluxBY(rho, rhovx, rhovy, Bx, By))
        - (dt / dy) * (YFluxBY() - YFluxBY())
        - (dt / dz) * (ZFluxBY(rho_kp1, rhovy_kp1, rhovz_kp1, By_kp1, Bz_kp1) - ZFluxBY(rho, rhovy, rhovz, By, Bz));
    }

__device__ float intBZLocal(const float rho, const float rho_ip1, const float rho_jp1, 
    const float rhovx, const float rhovx_ip1, 
    const float rhovy, const float rhovy_jp1,
    const float rhovz, const float rhovz_ip1, const float rhovz_jp1, 
    const float Bx, const float Bx_ip1, 
    const float By, const float By_jp1,
    const float Bz, const float Bz_ip1, const float Bz_jp1, 
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return Bz
        - (dt / dx) * (XFluxBZ(rho_ip1, rhovx_ip1, rhovz_ip1, Bx_ip1, Bz_ip1) - XFluxBZ(rho, rhovx, rhovz, Bx, Bz))
        - (dt / dy) * (YFluxBZ(rho_jp1, rhovy_jp1, rhovz_jp1, By_jp1, Bz_jp1) - YFluxBZ(rho, rhovy, rhovz, By, Bz))
        - (dt / dz) * (ZFluxBZ() - ZFluxBZ());
    }

__device__ float intELocal(const float rho, const float rho_ip1, const float rho_jp1, const float rho_kp1,
    const float rhovx, const float rhovx_ip1, 
    const float rhovy, const float rhovy_jp1, 
    const float rhovz, const float rhovz_kp1,
    const float Bx, const float Bx_ip1, 
    const float By, const float By_jp1, 
    const float Bz, const float Bz_kp1, 
    const float e, const float e_ip1, const float e_jp1, const float e_kp1,
    const float p_ijk, const float p_ip1jk, const float p_ijp1k, const float p_ijkp1,
    const float Bsq_ijk, const float Bsq_ip1jk, const float Bsq_ijp1k, const float Bsq_ijkp1, 
    const float Bdotu_ijk, const float Bdotu_ip1jk, const float Bdotu_ijp1k, const float Bdotu_ijkp1, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return e
        - (dt / dx) * (XFluxE(rho_ip1, rhovx_ip1, Bx_ip1, e_ip1, p_ip1jk, Bsq_ip1jk, Bdotu_ip1jk) - XFluxE(rho, rhovx, Bx, e, p_ijk, Bsq_ijk, Bdotu_ijk))
        - (dt / dy) * (YFluxE(rho_jp1, rhovy_jp1, By_jp1, e_jp1, p_ijp1k, Bsq_ijp1k, Bdotu_ijp1k) - YFluxE(rho, rhovy, By, e, p_ijk, Bsq_ijk, Bdotu_ijk))
        - (dt / dz) * (ZFluxE(rho_kp1, rhovz_kp1, Bz_kp1, e_kp1, p_ijkp1, Bsq_ijkp1, Bdotu_ijkp1) - ZFluxE(rho, rhovz, Bz, e, p_ijk, Bsq_ijk, Bdotu_ijk));
    }

// These thrash the cache instead of accepting local variables
__device__ float intRho(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]
            - (dt / dx) * (XFluxRho(i+1, j, k, fluidvar, Nx, Ny, Nz) - XFluxRho(i, j, k, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxRho(i, j+1, k, fluidvar, Nx, Ny, Nz) - YFluxRho(i, j, k, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRho(i, j, k+1, fluidvar, Nx, Ny, Nz) - ZFluxRho(i, j, k, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVX(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {   
        int cube_size = Nx * Ny * Nz;
        return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size]
        - (dt / dx) * (XFluxRhoVX(i+1, j, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVX(i, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dy) * (YFluxRhoVX(i, j+1, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVX(i, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dz) * (ZFluxRhoVX(i, j, k+1, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(i, j, k, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVY(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size]
        - (dt / dx) * (XFluxRhoVY(i+1, j, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVY(i, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dy) * (YFluxRhoVY(i, j+1, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVY(i, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dz) * (ZFluxRhoVY(i, j, k+1, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(i, j, k, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVZ(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size]
        - (dt / dx) * (XFluxRhoVZ(i+1, j, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVZ(i, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dy) * (YFluxRhoVZ(i, j+1, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVZ(i, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dz) * (ZFluxRhoVZ(i, j, k+1, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, j, k, fluidvar, Nx, Ny, Nz)); 
    }

__device__ float intBX(const int i, const int j, const int k,
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size]
        - (dt / dx) * (XFluxBX(i+1, j, k, fluidvar, Nx, Ny, Nz) - XFluxBX(i, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dy) * (YFluxBX(i, j+1, k, fluidvar, Nx, Ny, Nz) - YFluxBX(i, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dz) * (ZFluxBX(i, j, k+1, fluidvar, Nx, Ny, Nz) - ZFluxBX(i, j, k, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBY(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size]
        - (dt / dx) * (XFluxBY(i+1, j, k, fluidvar, Nx, Ny, Nz) - XFluxBY(i, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dy) * (YFluxBY(i, j+1, k, fluidvar, Nx, Ny, Nz) - YFluxBY(i, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dz) * (ZFluxBY(i, j, k+1, fluidvar, Nx, Ny, Nz) - ZFluxBY(i, j, k, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBZ(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz, 
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size]
        - (dt / dx) * (XFluxBZ(i+1, j, k, fluidvar, Nx, Ny, Nz) - XFluxBZ(i, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dy) * (YFluxBZ(i, j+1, k, fluidvar, Nx, Ny, Nz) - YFluxBZ(i, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dz) * (ZFluxBZ(i, j, k+1, fluidvar, Nx, Ny, Nz) - ZFluxBZ(i, j, k, fluidvar, Nx, Ny, Nz));
    }

__device__ float intE(const int i, const int j, const int k, 
    const float* fluidvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size]
        - (dt / dx) * (XFluxE(i+1, j, k, fluidvar, Nx, Ny, Nz) - XFluxE(i, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dy) * (YFluxE(i, j+1, k, fluidvar, Nx, Ny, Nz) - YFluxE(i, j, k, fluidvar, Nx, Ny, Nz))
        - (dt / dz) * (ZFluxE(i, j, k+1, fluidvar, Nx, Ny, Nz) - ZFluxE(i, j, k, fluidvar, Nx, Ny, Nz));
    }
    
// I think that's everything!