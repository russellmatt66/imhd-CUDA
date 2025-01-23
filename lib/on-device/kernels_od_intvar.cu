#include <stdio.h>

#include "diffusion.cuh"
#include "kernels_od.cuh"
#include "kernels_od_fluxes.cuh"
#include "kernels_od_intvar.cuh"
#include "helper_functions.cuh"

/* THIS SHOULD BE DEFINED A SINGLE TIME IN A SINGLE PLACE */
#define IDX3D(i, j, k, Nx, Ny, Nz) ((k) * (Nx) * (Ny) + (i) * (Ny) + j) // parentheses are necessary to avoid calculating `i - 1 * Ny` or `k - 1 * (Nx * Ny)`

// MEGAKERNELS
// Megakernel that uses large amounts of registers to avoid thrashing the cache
__global__ void ComputeIntVarsLocalNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz, const float D,
    const int Nx, const int Ny, const int Nz)
    {
        int tidx = threadIdx.x + blockDim.x * blockIdx.x;
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;

        int xthreads = blockDim.x * gridDim.x;
        int ythreads = blockDim.y * gridDim.y;
        int zthreads = blockDim.z * gridDim.z;

        /* 
        COMPUTE THE INTERMEDIATE VARIABLES USING LOCAL (SHARED) MEMORY
        */

        return;
    }

// Thrashes the cache and uses loops
// Uses loops
__global__ void ComputeIntermediateVariablesNoDiff(const float* fluidvar, float* intvar,
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
// Uses loops
// Adds diffusion
/* DIFFUSION NOT VERIFIED */
__global__ void ComputeIntermediateVariables(const float* fluidvar, float* intvar,
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
// These kernels deal with the interior of the computational volume

// Accept local variables
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

// These also thrash the cache
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



// Kernels that deal with the interior of the k = 0 plane: (i, j, 0)
/* 
THESE ARE WRONG 
THEY IMPLEMENT CORRECTOR STEP
WRONG AND NO REASON TO BE HERE - CONCERN MOVED TO `kernels_intvarbcs.cu` 
*/
// __device__ float intRhoFront(const int i, const int j, 
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz)]
//             - (dt / dx) * (XFluxRho(i+1, j, 0, fluidvar, Nx, Ny, Nz) - XFluxRho(i, j, 0, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxRho(i, j+1, 0, fluidvar, Nx, Ny, Nz) - YFluxRho(i, j, 0, fluidvar, Nx, Ny, Nz))
//             - (dt / dz) * (ZFluxRho(i, j, 1, fluidvar, Nx, Ny, Nz) - ZFluxRho(i, j, 0, fluidvar, Nx, Ny, Nz));        
//     }

// __device__ float intRhoVXFront(const int i, const int j,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz) + Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVX(i, j, 0, fluidvar, Nx, Ny, Nz) - XFluxRhoVX(i-1, j, 0, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxRhoVX(i, j, 0, fluidvar, Nx, Ny, Nz) - YFluxRhoVX(i, j-1, 0, fluidvar, Nx, Ny, Nz))
//             - (dt / dz) * (ZFluxRhoVX(i, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(i, j, Nz-2, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intRhoVYFront(const int i, const int j,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVY(i, j, 0, fluidvar, Nx, Ny, Nz) - XFluxRhoVY(i-1, j, 0, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxRhoVY(i, j, 0, fluidvar, Nx, Ny, Nz) - YFluxRhoVY(i, j-1, 0, fluidvar, Nx, Ny, Nz))
//             - (dt / dz) * (ZFluxRhoVY(i, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(i, j, Nz-2, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intRhoVZFront(const int i, const int j,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 3 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVZ(i, j, 0, fluidvar, Nx, Ny, Nz) - XFluxRhoVZ(i-1, j, 0, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxRhoVZ(i, j, 0, fluidvar, Nx, Ny, Nz) - YFluxRhoVZ(i, j-1, 0, fluidvar, Nx, Ny, Nz))
//             - (dt / dz) * (ZFluxRhoVZ(i, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, j, Nz-2, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intBXFront(const int i, const int j,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBX(i, j, 0, fluidvar, Nx, Ny, Nz) - XFluxBX(i-1, j, 0, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxBX(i, j, 0, fluidvar, Nx, Ny, Nz) - YFluxBX(i, j-1, 0, fluidvar, Nx, Ny, Nz))
//             - (dt / dz) * (ZFluxBX(i, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxBX(i, j, Nz-2, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intBYFront(const int i, const int j,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBY(i, j, 0, fluidvar, Nx, Ny, Nz) - XFluxBY(i-1, j, 0, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxBY(i, j, 0, fluidvar, Nx, Ny, Nz) - YFluxBY(i, j-1, 0, fluidvar, Nx, Ny, Nz))
//             - (dt / dz) * (ZFluxBY(i, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxBY(i, j, Nz-2, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intBZFront(const int i, const int j, 
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBZ(i, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxBZ(i, j, 0, fluidvar, Nx, Ny, Nz) - YFluxBZ(i, j-1, 0, fluidvar, Nx, Ny, Nz))
//             - (dt / dz) * (ZFluxBZ(i, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxBZ(i, j, Nz-2, fluidvar, Nx, Ny, Nz));  
//     }

// __device__ float intEFront(const int i, const int j,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxE(i, j, 0, fluidvar, Nx, Ny, Nz) - XFluxE(i-1, j, 0, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxE(i, j, 0, fluidvar, Nx, Ny, Nz) - YFluxE(i, j-1, 0, fluidvar, Nx, Ny, Nz))
//             - (dt / dz) * (ZFluxE(i, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxE(i, j, Nz-2, fluidvar, Nx, Ny, Nz));
//     }

// // Kernels that deal with the (0, j, 0) line
// __device__ float intRhoFrontTop(const int j, 
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz)]
//             - (dt / dx) * (XFluxRho(0, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxRho(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxRho(0, j-1, 0, fluidvar, Nx, Ny, Nz))
//             - (dt / dz) * (ZFluxRho(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxRho(0, j, Nz-2, fluidvar, Nx, Ny, Nz));        
//     }

// __device__ float intRhoVXFrontTop(const int j,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVX(0, j, 0, fluidvar, Nx, Ny, Nz)) // i = -1 flux is zero => rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxRhoVX(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxRhoVX(0, j-1, 0, fluidvar, Nx, Ny, Nz))
//             - (dt / dz) * (ZFluxRhoVX(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(0, j, Nx-2, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intRhoVYFrontTop(const int j,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVY(0, j, 0, fluidvar, Nx, Ny, Nz)) // i = -1 flux is zero => rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxRhoVY(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxRhoVY(0, j-1, 0, fluidvar, Nx, Ny, Nz))
//             - (dt / dz) * (ZFluxRhoVY(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(0, j, Nz-2, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intRhoVZFrontTop(const int j, 
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 3 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVZ(0, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxRhoVZ(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxRhoVZ(0, j-1, 0, fluidvar, Nx, Ny, Nz))
//             - (dt / dz) * (ZFluxRhoVZ(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(0, j, Nz-2, fluidvar, Nx, Ny, Nz));  
//     }

// __device__ float intBXFrontTop(const int j, 
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBX(0, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxBX(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxBX(0, j-1, 0, fluidvar, Nx, Ny, Nz))
//             - (dt / dz) * (ZFluxBX(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxBX(0, j, Nz-2, fluidvar, Nx, Ny, Nz));  
//     }

// __device__ float intBYFrontTop(const int j, 
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBY(0, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxBY(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxBY(0, j-1, 0, fluidvar, Nx, Ny, Nz))
//             - (dt / dz) * (ZFluxBY(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxBY(0, j, Nz-2, fluidvar, Nx, Ny, Nz));  
//     }

// __device__ float intBZFrontTop(const int j,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBZ(0, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxBZ(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxBZ(0, j-1, 0, fluidvar, Nx, Ny, Nz))
//             - (dt / dz) * (ZFluxBZ(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxBZ(0, j, Nz-2, fluidvar, Nx, Ny, Nz));  
//     }

// __device__ float intEFrontTop(const int j, 
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxE(0, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxE(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxE(0, j-1, 0, fluidvar, Nx, Ny, Nz))
//             - (dt / dz) * (ZFluxE(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxE(0, j, Nz-2, fluidvar, Nx, Ny, Nz));  
//     }

// // Kernels that deal with Front Left Line
// // i \in [1, Nx-1]
// // j = 0
// // k = 0
// __device__ float intRhoFrontLeft(const int i, 
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz)]
//             - (dt / dx) * (XFluxRho(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxRho(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxRho(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxRho(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall       
//     }

// __device__ float intRhoVXFrontLeft(const int i,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVX(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxRhoVX(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxRhoVX(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxRhoVX(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
//     }

// __device__ float intRhoVYFrontLeft(const int i,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVY(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxRhoVY(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxRhoVY(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxRhoVY(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
//     }

// __device__ float intRhoVZFrontLeft(const int i,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 3 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVZ(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxRhoVZ(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxRhoVZ(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxRhoVZ(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
//     }

// __device__ float intBXFrontLeft(const int i,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBX(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxBX(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxBX(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxBX(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
//     }

// __device__ float intBYFrontLeft(const int i,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBY(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxBY(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxBY(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxBY(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
//     }

// __device__ float intBZFrontLeft(const int i, 
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBZ(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxBZ(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxBZ(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxBZ(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall  
//     }

// __device__ float intEFrontLeft(const int i,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxE(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxE(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxE(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxE(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
//     }

// // Kernels that deal with the Left Face 
// // j = 0 
// // i \in [1, Nx-1]
// // k \in [1, Nz-2]
// __device__ float intRhoLeft(const int i, const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz)]
//             - (dt / dx) * (XFluxRho(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRho(i-1, 0, k, fluidvar, Nx, Ny, Nz))
//             - (dt / dy) * (YFluxRho(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxRho(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRho(i, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intRhoVXLeft(const int i, const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVX(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVX(i-1, 0, k, fluidvar, Nx, Ny, Nz))
//             - (dt / dy) * (YFluxRhoVX(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxRhoVX(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(i, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intRhoVYLeft(const int i, const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVY(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVY(i-1, 0, k, fluidvar, Nx, Ny, Nz))
//             - (dt / dy) * (YFluxRhoVY(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxRhoVY(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(i, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intRhoVZLeft(const int i, const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVZ(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVZ(i-1, 0, k, fluidvar, Nx, Ny, Nz))
//             - (dt / dy) * (YFluxRhoVZ(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxRhoVZ(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intBXLeft(const int i, const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBX(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxBX(i-1, 0, k, fluidvar, Nx, Ny, Nz))
//             - (dt / dy) * (YFluxBX(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxBX(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBX(i, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intBYLeft(const int i, const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBY(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxBY(i-1, 0, k, fluidvar, Nx, Ny, Nz))
//             - (dt / dy) * (YFluxBY(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxBY(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBY(i, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intBZLeft(const int i, const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBZ(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxBZ(i-1, 0, k, fluidvar, Nx, Ny, Nz))
//             - (dt / dy) * (YFluxBZ(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxBZ(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBZ(i, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intELeft(const int i, const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxE(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxE(i-1, 0, k, fluidvar, Nx, Ny, Nz))
//             - (dt / dy) * (YFluxE(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxE(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxE(i, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// // Kernels that deal with Top Face
// // i = 0
// // j \in [1,Ny-1]
// // k \in [1,Nz-2] - leave the front / back faces alone
// __device__ float intRhoTop(const int j, const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz)]
//             - (dt / dx) * (XFluxRho(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxRho(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxRho(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dz) * (ZFluxRho(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRho(0, j, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intRhoVXTop(const int j, const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVX(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxRhoVX(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVX(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dz) * (ZFluxRhoVX(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(0, j, k-1, fluidvar, Nx, Ny, Nz));
//     }
    
// __device__ float intRhoVYTop(const int j, const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVY(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxRhoVY(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVY(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dz) * (ZFluxRhoVY(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(0, j, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intRhoVZTop(const int j, const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 3 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVZ(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxRhoVZ(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVZ(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dz) * (ZFluxRhoVZ(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(0, j, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intBXTop(const int j, const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBX(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxBX(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxBX(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dz) * (ZFluxBX(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxBX(0, j, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intBYTop(const int j, const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBY(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxBY(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxBY(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dz) * (ZFluxBY(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxBY(0, j, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intBZTop(const int j, const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBZ(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxBZ(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxBZ(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dz) * (ZFluxBZ(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxBZ(0, j, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intETop(const int j, const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxE(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxE(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxE(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dz) * (ZFluxE(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxE(0, j, k-1, fluidvar, Nx, Ny, Nz));
//     }

// // Kernels that deal with the Top Left Line
// // i = 0
// // j = 0
// // k \in [1, Nz-1]
// __device__ float intRhoTopLeft(const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz)]
//             - (dt / dx) * (XFluxRho(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxRho(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxRho(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRho(0, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intRhoVXTopLeft(const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVX(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxRhoVX(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxRhoVX(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(0, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intRhoVYTopLeft(const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVY(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxRhoVY(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxRhoVY(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(0, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intRhoVZTopLeft(const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVZ(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxRhoVZ(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxRhoVZ(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(0, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intBXTopLeft(const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBX(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxBX(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxBX(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBX(0, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intBYTopLeft(const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBY(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxBY(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxBY(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBY(0, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intBZTopLeft(const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBZ(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxBZ(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxBZ(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBZ(0, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intETopLeft(const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxE(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dy) * (YFluxE(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxE(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxE(0, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// // Kernels that deal with the Bottom Left Line
// // i = Nx-1
// // j = 0
// // k \in [1, Nz-2]
// __device__ float intRhoBottomLeft(const int k, 
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz)]
//             - (dt / dx) * (XFluxRho(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRho(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxRho(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxRho(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRho(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));        
//     }

// __device__ float intRhoVXBottomLeft(const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVX(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVX(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxRhoVX(Nx-1, 0, k, fluidvar, Nx, Ny, Nz))  // No flux on opposite site of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxRhoVX(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intRhoVYBottomLeft(const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVY(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVY(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxRhoVY(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxRhoVY(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intRhoVZBottomLeft(const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 3 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxRhoVZ(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVZ(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxRhoVZ(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxRhoVZ(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intBXBottomLeft(const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBX(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxBX(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxBX(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxBX(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBX(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intBYBottomLeft(const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBY(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxBY(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxBY(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxBY(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBY(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// __device__ float intBZBottomLeft(const int k, 
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxBZ(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxBZ(Nx-2, 0, k, fluidvar, Nx, Ny, Nz))
//             - (dt / dy) * (YFluxBZ(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxBZ(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBZ(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));  
//     }

// __device__ float intEBottomLeft(const int k,
//     const float* fluidvar,
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
//             - (dt / dx) * (XFluxE(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxE(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
//             - (dt / dy) * (YFluxE(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
//             - (dt / dz) * (ZFluxE(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxE(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));
//     }

// I think that's everything! 