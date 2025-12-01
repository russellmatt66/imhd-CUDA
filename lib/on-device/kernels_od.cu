#include <math.h>
#include <stdio.h>

#include "kernels_od.cuh"
#include "kernels_od_fluxes.cuh"
#include "helper_functions.cuh"
#include "diffusion.cuh"

/* THIS SHOULD BE DEFINED A SINGLE TIME IN A SINGLE PLACE */
// row-major, column-minor order
#define IDX3D(i, j, k, Nx, Ny, Nz) ((k) * (Nx) * (Ny) + (i) * (Ny) + j) // parentheses are necessary to avoid calculating `i - 1 * Ny` or `k - 1 * (Nx * Ny)`

/* 
NOTE:
Based on the above macro, this is what the storage pattern looks like:
fluidvar -> [rho_{000}, rho_{010}, rho_{020}, ..., rho_{0,Ny-1,0}, rho_{100}, ..., rho_{Nx-1,Ny-1,Nz-1}, rhov_x_{000}, rhov_x_{010}, ... , e_{Nx-1,Ny-1,Nz-1}]
*/

/* 
REGISTER PRESSURES: (registers per thread)
FluidAdvance=88
FluidAdvanceLocal=255
FluidAdvanceLocalNoDiff=160
FluidAdvanceMicroRhoLocalNoDiff=32
FluidAdvanceMicroRhoVXLocalNoDiff=66
FluidAdvanceMicroRhoVYLocalNoDiff=72
FluidAdvanceMicroRhoVZLocalNoDiff=72
FluidAdvanceMicroBXLocalNoDiff=56
FluidAdvanceMicroBYLocalNoDiff=50
FluidAdvanceMicroBZLocalNoDiff=50
FluidAdvanceMicroELocalNoDiff=90


*/

// Megakernels
// FluidAdvance uses kernels that thrash the cache, very badly
// FluidAdvanceLocal uses kernels that do not thrash the cache, but it puts a lot of pressure on the registers

// 88 registers per thread
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

// 255 registers per thread 
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

        // Tradeoff cache-thrashing for high register pressure
        float rho_int = 0.0, rhovx_int = 0.0, rhovy_int = 0.0, rhovz_int = 0.0, Bx_int = 0.0, By_int = 0.0, Bz_int = 0.0, e_int = 0.0;
        float rho_int_im1 = 0.0, rhovx_int_im1 = 0.0, rhovy_int_im1 = 0.0, rhovz_int_im1 = 0.0, Bx_int_im1 = 0.0, By_int_im1 = 0.0, Bz_int_im1 = 0.0, e_int_im1 = 0.0;
        float rho_int_jm1 = 0.0, rhovx_int_jm1 = 0.0, rhovy_int_jm1 = 0.0, rhovz_int_jm1 = 0.0, Bx_int_jm1 = 0.0, By_int_jm1 = 0.0, Bz_int_jm1 = 0.0, e_int_jm1 = 0.0;
        float rho_int_km1 = 0.0, rhovx_int_km1 = 0.0, rhovy_int_km1 = 0.0, rhovz_int_km1 = 0.0, Bx_int_km1 = 0.0, By_int_km1 = 0.0, Bz_int_km1 = 0.0, e_int_km1 = 0.0;
        float rho_int_ip1 = 0.0, rhovx_int_ip1 = 0.0, rhovy_int_ip1 = 0.0, rhovz_int_ip1 = 0.0, Bx_int_ip1 = 0.0, By_int_ip1 = 0.0, Bz_int_ip1 = 0.0, e_int_ip1 = 0.0;
        float rho_int_jp1 = 0.0, rhovx_int_jp1 = 0.0, rhovy_int_jp1 = 0.0, rhovz_int_jp1 = 0.0, Bx_int_jp1 = 0.0, By_int_jp1 = 0.0, Bz_int_jp1 = 0.0, e_int_jp1 = 0.0;
        float rho_int_kp1 = 0.0, rhovx_int_kp1 = 0.0, rhovy_int_kp1 = 0.0, rhovz_int_kp1 = 0.0, Bx_int_kp1 = 0.0, By_int_kp1 = 0.0, Bz_int_kp1 = 0.0, e_int_kp1 = 0.0;

        float KE_ijk = 0.0, KE_im1jk = 0.0, KE_ijm1k = 0.0, KE_ijkm1 = 0.0;
        float p_ijk = 0.0, p_im1jk = 0.0, p_ijm1k = 0.0, p_ijkm1 = 0.0;
        float Bsq_ijk = 0.0, Bsq_im1jk = 0.0, Bsq_ijm1k = 0.0, Bsq_ijkm1 = 0.0;
        float Bdotu_ijk = 0.0, Bdotu_im1jk = 0.0, Bdotu_ijm1k = 0.0, Bdotu_ijkm1 = 0.0;

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
                    rho_int_ip1 = intvar[IDX3D(i+1, j, k, Nx, Ny, Nz)];
                    rhovx_int_ip1 = intvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + cube_size];
                    rhovy_int_ip1 = intvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 2 * cube_size];
                    rhovz_int_ip1 = intvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 3 * cube_size];
                    Bx_int_ip1 = intvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 4 * cube_size];
                    By_int_ip1 = intvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 5 * cube_size];
                    Bz_int_ip1 = intvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 6 * cube_size];
                    e_int_ip1 = intvar[IDX3D(i+1, j, k, Nx, Ny, Nz) + 7 * cube_size];
                    
                    rho_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz)];
                    rhovx_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + cube_size];
                    rhovy_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 2 * cube_size];
                    rhovz_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 3 * cube_size];
                    Bx_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 4 * cube_size];
                    By_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 5 * cube_size];
                    Bz_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 6 * cube_size];
                    e_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 7 * cube_size];

                    rho_int_jp1 = intvar[IDX3D(i, j+1, k, Nx, Ny, Nz)];
                    rhovx_int_jp1 = intvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + cube_size];
                    rhovy_int_jp1 = intvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 2 * cube_size];
                    rhovz_int_jp1 = intvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 3 * cube_size];
                    Bx_int_jp1 = intvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 4 * cube_size];
                    By_int_jp1 = intvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 5 * cube_size];
                    Bz_int_jp1 = intvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 6 * cube_size];
                    e_int_jp1 = intvar[IDX3D(i, j+1, k, Nx, Ny, Nz) + 7 * cube_size];
                    
                    rho_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz)];
                    rhovx_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + cube_size];
                    rhovy_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 2 * cube_size];
                    rhovz_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 3 * cube_size];
                    Bx_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 4 * cube_size];
                    By_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 5 * cube_size];
                    Bz_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 6 * cube_size];
                    e_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 7 * cube_size];

                    rho_int_kp1 = intvar[IDX3D(i, j, k+1, Nx, Ny, Nz)];
                    rhovx_int_kp1 = intvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + cube_size];
                    rhovy_int_kp1 = intvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 2 * cube_size];
                    rhovz_int_kp1 = intvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 3 * cube_size];
                    Bx_int_kp1 = intvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 4 * cube_size];
                    By_int_kp1 = intvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 5 * cube_size];
                    Bz_int_kp1 = intvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 6 * cube_size];
                    e_int_kp1 = intvar[IDX3D(i, j, k+1, Nx, Ny, Nz) + 7 * cube_size];

                    // KE_ijk = KE(i, j, k, intvar, Nx, Ny, Nz);
                    // KE_im1jk = KE(i - 1, j, k, intvar, Nx, Ny, Nz);
                    // KE_ijm1k = KE(i, j-1, k, intvar, Nx, Ny, Nz);
                    // KE_ijkm1 = KE(i, j, k-1, intvar, Nx, Ny, Nz);

                    KE_ijk = KE_local(rho_int, rhovx_int, rhovy_int, rhovz_int);
                    KE_im1jk = KE_local(rho_int_im1, rhovx_int_im1, rhovy_int_im1, rhovz_int_im1);
                    KE_ijm1k = KE_local(rho_int_jm1, rhovx_int_jm1, rhovy_int_jm1, rhovz_int_jm1);
                    KE_ijkm1 = KE_local(rho_int_km1, rhovx_int_km1, rhovy_int_km1, rhovz_int_km1);

                    // Bsq_ijk = B_sq(i, j, k, intvar, Nx, Ny, Nz);
                    // Bsq_im1jk = B_sq(i-1, j, k, intvar, Nx, Ny, Nz);
                    // Bsq_ijm1k = B_sq(i, j-1, k, intvar, Nx, Ny, Nz);
                    // Bsq_ijkm1 = B_sq(i, j, k-1, intvar, Nx, Ny, Nz);

                    Bsq_ijk = B_sq_local(Bx_int, By_int, Bz_int);
                    Bsq_im1jk = B_sq_local(Bx_int_im1, By_int_im1, Bz_int_im1);
                    Bsq_ijm1k = B_sq_local(Bx_int_jm1, By_int_jm1, Bz_int_jm1);
                    Bsq_ijkm1 = B_sq_local(Bx_int_im1, By_int_jm1, Bz_int_km1);

                    // p_ijk = p(i, j, k, intvar, Bsq_ijk, KE_ijk, Nx, Ny, Nz);
                    // p_im1jk = p(i-1, j, k, intvar, Bsq_ijk, KE_ijk, Nx, Ny, Nz);
                    // p_ijm1k = p(i, j-1, k, intvar, Bsq_ijm1k, KE_ijm1k, Nx, Ny, Nz);
                    // p_ijkm1 = p(i, j, k-1, intvar, Bsq_ijkm1, KE_ijkm1, Nx, Ny, Nz);

                    p_ijk = p_local(e_int, Bsq_ijk, KE_ijk);
                    p_im1jk = p_local(e_int_im1, Bsq_im1jk, KE_im1jk);
                    p_ijm1k = p_local(e_int_jm1, Bsq_ijm1k, KE_ijm1k);
                    p_ijkm1 = p_local(e_int_km1, Bsq_ijkm1, KE_ijkm1);

                    // Bdot_ijk = B_dot_u(i, j, k, intvar, Nx, Ny, Nz);
                    // Bdot_im1jk = B_dot_u(i-1, j, k, intvar, Nx, Ny, Nz);
                    // Bdot_ijm1k = B_dot_u(i, j-1, k, intvar, Nx, Ny, Nz);
                    // Bdot_ijkm1 = B_dot_u(i, j, k-1, intvar, Nx, Ny, Nz);

                    Bdotu_ijk = B_dot_u_local(rho_int, rhovx_int, rhovy_int, rhovz_int, Bx_int, By_int, Bz_int);
                    Bdotu_im1jk = B_dot_u_local(rho_int_im1, rhovx_int_im1, rhovy_int_im1, rhovz_int_im1, Bx_int_im1, By_int_im1, Bz_int_im1);
                    Bdotu_ijm1k = B_dot_u_local(rho_int_jm1, rhovx_int_jm1, rhovy_int_jm1, rhovz_int_jm1, Bx_int_jm1, By_int_jm1, Bz_int_jm1);
                    Bdotu_ijkm1 = B_dot_u_local(rho_int_km1, rhovx_int_km1, rhovy_int_jm1, rhovz_int_km1, Bx_int_km1, By_int_km1, Bz_int_km1);

                    // Update and store fluidvars
                    // rho
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvRhoLocal(rho, rho_int, 
                                                                rhovx_int, rho_int_im1, 
                                                                rhovy_int, rhovy_int_jm1, 
                                                                rhovz_int, rhovz_int_km1, 
                                                                dt, dx, dy, dz) 
                                                            + dt * numericalDiffusionLocal(rho_int, 
                                                                    rho_int_ip1, rho_int_jp1, rho_int_kp1, 
                                                                    rho_int_im1, rho_int_jm1, rho_int_km1, 
                                                                    D, dx, dy, dz); 
                    // rhovx
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] =  LaxWendroffAdvRhoVXLocal(rho_int, rho_int_im1, rho_int_jm1, rho_int_km1, 
                                                                            rhovx, rhovx_int, rhovx_int_im1, rhovx_int_jm1, rhovx_int_km1, 
                                                                            rhovy_int, rhovy_int_jm1, 
                                                                            rhovz_int, rhovz_int_km1, 
                                                                            Bx_int, Bx_int_im1, Bx_int_jm1, Bx_int_km1, 
                                                                            By_int, By_int_jm1, 
                                                                            Bz_int, Bz_int_km1, 
                                                                            p_ijk, p_im1jk, 
                                                                            Bsq_ijk, Bsq_im1jk, 
                                                                            dt, dx, dy, dz) 
                                                                        + dt * numericalDiffusionLocal(rhovx_int, 
                                                                            rhovx_int_ip1, rhovx_int_jp1, rhovx_int_kp1, 
                                                                            rhovx_int_im1, rhovx_int_jm1, rhovx_int_km1, 
                                                                            D, dx, dy, dz); 
                    // rhovy
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] =  LaxWendroffAdvRhoVYLocal(rho_int, rho_int_im1, rho_int_jm1, rho_int_km1, 
                                                                                rhovx_int, rhovx_int_im1, 
                                                                                rhovy, rhovy_int, rhovy_int_im1, rhovy_int_jm1, rhovy_int_km1, 
                                                                                rhovz_int, rhovz_int_km1, 
                                                                                Bx_int, Bx_int_im1, 
                                                                                By_int, By_int_im1, By_int_jm1, By_int_km1, 
                                                                                Bz_int, Bz_int_km1, 
                                                                                Bsq_ijk, Bsq_ijm1k, 
                                                                                p_ijk, p_ijm1k, 
                                                                                dt, dx, dy, dz) 
                                                                            + dt * numericalDiffusionLocal(rhovy_int, 
                                                                                rhovy_int_ip1, rhovy_int_jp1, rhovy_int_kp1, 
                                                                                rhovy_int_im1, rhovy_int_jm1, rhovy_int_km1, 
                                                                                D, dx, dy, dz); 
                    // rhovz
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] =  LaxWendroffAdvRhoVZLocal(rho_int, rho_int_im1, rho_int_jm1, rho_int_km1, 
                                                                                rhovx_int, rhovx_int_im1, 
                                                                                rhovy_int, rhovy_int_jm1, 
                                                                                rhovz, rhovz_int, rhovz_int_im1, rhovz_int_jm1, rhovz_int_km1,
                                                                                Bx_int, Bx_int_im1, 
                                                                                By_int, By_int_jm1, 
                                                                                Bz_int, Bz_int_im1, Bz_int_jm1, Bz_int_km1, 
                                                                                p_ijk, p_ijkm1,
                                                                                Bsq_ijk, Bsq_ijkm1,
                                                                                dt, dx, dy, dz) 
                                                                            + dt * numericalDiffusionLocal(rhovz_int, 
                                                                                rhovz_int_ip1, rhovz_int_jp1, rhovz_int_kp1, 
                                                                                rhovz_int_im1, rhovz_int_jm1, rhovz_int_km1, 
                                                                                D, dx, dy, dz); 
                    // Bx
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] =  LaxWendroffAdvBXLocal(rho_int, rho_int_jm1, rho_int_km1, 
                                                                                rhovx_int, rhovx_int_jm1, rhovx_int_km1, 
                                                                                rhovy_int, rhovy_int_jm1, 
                                                                                rhovz_int, rhovz_int_km1, 
                                                                                Bx, Bx_int, Bx_int_jm1, Bx_int_km1, 
                                                                                By_int, By_int_jm1, 
                                                                                Bz_int, Bz_int_km1, 
                                                                                dt, dx, dy, dz) 
                                                                            + dt * numericalDiffusionLocal(Bx_int, 
                                                                                Bx_int_ip1, Bx_int_jp1, Bx_int_kp1, 
                                                                                Bx_int_im1, Bx_int_jm1, Bx_int_km1, 
                                                                                D, dx, dy, dz); 
                    // By
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] =  LaxWendroffAdvBYLocal(rho_int, rho_int_im1, rho_int_km1, 
                                                                                rhovx_int, rhovx_int_im1, 
                                                                                rhovy_int, rhovy_int_im1, rhovy_int_km1, 
                                                                                rhovz_int, rhovz_int_km1, 
                                                                                Bx_int, Bx_int_im1, 
                                                                                By, By_int, By_int_im1, By_int_km1, 
                                                                                Bz_int, Bz_int_km1, 
                                                                                dt, dx, dy, dz)
                                                                            + dt * numericalDiffusionLocal(By_int, 
                                                                                By_int_ip1, By_int_jp1, By_int_kp1, 
                                                                                By_int_im1, By_int_jm1, By_int_km1, 
                                                                                D, dx, dy, dz); 
                    // Bz
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] =  LaxWendroffAdvBZLocal(rho_int, rho_int_im1, rho_int_jm1, 
                                                                                rhovx_int, rhovx_int_im1, 
                                                                                rhovy_int, rhovy_int_jm1, 
                                                                                rhovz_int, rhovz_int_im1, rhovz_int_jm1, 
                                                                                Bx_int, Bx_int_im1, 
                                                                                By_int, By_int_jm1, 
                                                                                Bz, Bz_int, Bz_int_im1, Bz_int_jm1, 
                                                                                dt, dx, dy, dz) 
                                                                            + dt * numericalDiffusionLocal(Bz_int, 
                                                                                Bz_int_ip1, Bz_int_jp1, Bz_int_kp1, 
                                                                                Bz_int_im1, Bz_int_jm1, Bz_int_km1, 
                                                                                D, dx, dy, dz); 
                    // e
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] =  LaxWendroffAdvELocal(rho_int, rho_int_im1, rho_int_jm1, rho_int_km1, 
                                                                                rhovx_int, rhovx_int_im1, 
                                                                                rhovy_int, rhovy_int_jm1, 
                                                                                rhovz_int, rhovz_int_km1,
                                                                                Bx_int, Bx_int_im1, 
                                                                                By_int, By_int_jm1, 
                                                                                Bz_int, Bz_int_km1, 
                                                                                e, e_int, e_int_im1, e_int_jm1, e_int_km1,
                                                                                p_ijk, p_im1jk, p_ijm1k, p_ijkm1, 
                                                                                Bsq_ijk, Bsq_im1jk, Bsq_ijm1k, Bsq_ijkm1, 
                                                                                Bdotu_ijk, Bdotu_im1jk, Bdotu_ijm1k, Bdotu_ijkm1,  
                                                                                dt, dx, dy, dz) 
                                                                            + dt * numericalDiffusionLocal(e_int, 
                                                                                e_int_ip1, e_int_jp1, e_int_kp1, 
                                                                                e_int_im1, e_int_jm1, e_int_km1, 
                                                                                D, dx, dy, dz); 
                }
            }
        }
        return;
    }

// 160 registers per thread
__global__ void FluidAdvanceLocalNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        int k = threadIdx.z + blockDim.z * blockIdx.z;

        int cube_size = Nx * Ny * Nz;

        float rho = 0.0, rhovx = 0.0, rhovy = 0.0, rhovz = 0.0, Bx = 0.0, By = 0.0, Bz = 0.0, e = 0.0;

        // Tradeoff cache-thrashing for high register pressure
        float rho_int = 0.0, rhovx_int = 0.0, rhovy_int = 0.0, rhovz_int = 0.0, Bx_int = 0.0, By_int = 0.0, Bz_int = 0.0, e_int = 0.0;
        float rho_int_im1 = 0.0, rhovx_int_im1 = 0.0, rhovy_int_im1 = 0.0, rhovz_int_im1 = 0.0, Bx_int_im1 = 0.0, By_int_im1 = 0.0, Bz_int_im1 = 0.0, e_int_im1 = 0.0;
        float rho_int_jm1 = 0.0, rhovx_int_jm1 = 0.0, rhovy_int_jm1 = 0.0, rhovz_int_jm1 = 0.0, Bx_int_jm1 = 0.0, By_int_jm1 = 0.0, Bz_int_jm1 = 0.0, e_int_jm1 = 0.0;
        float rho_int_km1 = 0.0, rhovx_int_km1 = 0.0, rhovy_int_km1 = 0.0, rhovz_int_km1 = 0.0, Bx_int_km1 = 0.0, By_int_km1 = 0.0, Bz_int_km1 = 0.0, e_int_km1 = 0.0;

        float KE_ijk = 0.0, KE_im1jk = 0.0, KE_ijm1k = 0.0, KE_ijkm1 = 0.0;
        float p_ijk = 0.0, p_im1jk = 0.0, p_ijm1k = 0.0, p_ijkm1 = 0.0;
        float Bsq_ijk = 0.0, Bsq_im1jk = 0.0, Bsq_ijm1k = 0.0, Bsq_ijkm1 = 0.0;
        float Bdotu_ijk = 0.0, Bdotu_im1jk = 0.0, Bdotu_ijm1k = 0.0, Bdotu_ijkm1 = 0.0;

        // The analogue to the `2` __device__ kernels here were the parts that were thrashing the cache in `FluidAdvance`
        if (i > 0 && i < Nx && j > 0 && j < Ny && k > 0 && k < Nz) {
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

            KE_ijk = KE_local(rho_int, rhovx_int, rhovy_int, rhovz_int);
            KE_im1jk = KE_local(rho_int_im1, rhovx_int_im1, rhovy_int_im1, rhovz_int_im1);
            KE_ijm1k = KE_local(rho_int_jm1, rhovx_int_jm1, rhovy_int_jm1, rhovz_int_jm1);
            KE_ijkm1 = KE_local(rho_int_km1, rhovx_int_km1, rhovy_int_km1, rhovz_int_km1);

            Bsq_ijk = B_sq_local(Bx_int, By_int, Bz_int);
            Bsq_im1jk = B_sq_local(Bx_int_im1, By_int_im1, Bz_int_im1);
            Bsq_ijm1k = B_sq_local(Bx_int_jm1, By_int_jm1, Bz_int_jm1);
            Bsq_ijkm1 = B_sq_local(Bx_int_im1, By_int_jm1, Bz_int_km1);

            p_ijk = p_local(e_int, Bsq_ijk, KE_ijk);
            p_im1jk = p_local(e_int_im1, Bsq_im1jk, KE_im1jk);
            p_ijm1k = p_local(e_int_jm1, Bsq_ijm1k, KE_ijm1k);
            p_ijkm1 = p_local(e_int_km1, Bsq_ijkm1, KE_ijkm1);

            Bdotu_ijk = B_dot_u_local(rho_int, rhovx_int, rhovy_int, rhovz_int, Bx_int, By_int, Bz_int);
            Bdotu_im1jk = B_dot_u_local(rho_int_im1, rhovx_int_im1, rhovy_int_im1, rhovz_int_im1, Bx_int_im1, By_int_im1, Bz_int_im1);
            Bdotu_ijm1k = B_dot_u_local(rho_int_jm1, rhovx_int_jm1, rhovy_int_jm1, rhovz_int_jm1, Bx_int_jm1, By_int_jm1, Bz_int_jm1);
            Bdotu_ijkm1 = B_dot_u_local(rho_int_km1, rhovx_int_km1, rhovy_int_jm1, rhovz_int_km1, Bx_int_km1, By_int_km1, Bz_int_km1);

            // Update and store fluidvars
            // rho
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvRhoLocal(rho, rho_int, 
                                                        rhovx_int, rhovx_int_im1, 
                                                        rhovy_int, rhovy_int_jm1, 
                                                        rhovz_int, rhovz_int_km1, 
                                                        dt, dx, dy, dz); 
            // rhovx
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] =  LaxWendroffAdvRhoVXLocal(rho_int, rho_int_im1, rho_int_jm1, rho_int_km1, 
                                                                    rhovx, rhovx_int, rhovx_int_im1, rhovx_int_jm1, rhovx_int_km1, 
                                                                    rhovy_int, rhovy_int_jm1, 
                                                                    rhovz_int, rhovz_int_km1, 
                                                                    Bx_int, Bx_int_im1, Bx_int_jm1, Bx_int_km1, 
                                                                    By_int, By_int_jm1, 
                                                                    Bz_int, Bz_int_km1, 
                                                                    p_ijk, p_im1jk, 
                                                                    Bsq_ijk, Bsq_im1jk, 
                                                                    dt, dx, dy, dz); 
            // rhovy
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] =  LaxWendroffAdvRhoVYLocal(rho_int, rho_int_im1, rho_int_jm1, rho_int_km1, 
                                                                        rhovx_int, rhovx_int_im1, 
                                                                        rhovy, rhovy_int, rhovy_int_im1, rhovy_int_jm1, rhovy_int_km1, 
                                                                        rhovz_int, rhovz_int_km1, 
                                                                        Bx_int, Bx_int_im1, 
                                                                        By_int, By_int_im1, By_int_jm1, By_int_km1, 
                                                                        Bz_int, Bz_int_km1, 
                                                                        Bsq_ijk, Bsq_ijm1k, 
                                                                        p_ijk, p_ijm1k, 
                                                                        dt, dx, dy, dz); 
            // rhovz
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] =  LaxWendroffAdvRhoVZLocal(rho_int, rho_int_im1, rho_int_jm1, rho_int_km1, 
                                                                        rhovx_int, rhovx_int_im1, 
                                                                        rhovy_int, rhovy_int_jm1, 
                                                                        rhovz, rhovz_int, rhovz_int_im1, rhovz_int_jm1, rhovz_int_km1,
                                                                        Bx_int, Bx_int_im1, 
                                                                        By_int, By_int_jm1, 
                                                                        Bz_int, Bz_int_im1, Bz_int_jm1, Bz_int_km1, 
                                                                        p_ijk, p_ijkm1,
                                                                        Bsq_ijk, Bsq_ijkm1,
                                                                        dt, dx, dy, dz); 
            // Bx
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] =  LaxWendroffAdvBXLocal(rho_int, rho_int_jm1, rho_int_km1, 
                                                                        rhovx_int, rhovx_int_jm1, rhovx_int_km1, 
                                                                        rhovy_int, rhovy_int_jm1, 
                                                                        rhovz_int, rhovz_int_km1, 
                                                                        Bx, Bx_int, Bx_int_jm1, Bx_int_km1, 
                                                                        By_int, By_int_jm1, 
                                                                        Bz_int, Bz_int_km1, 
                                                                        dt, dx, dy, dz); 
            // By
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] =  LaxWendroffAdvBYLocal(rho_int, rho_int_im1, rho_int_km1, 
                                                                        rhovx_int, rhovx_int_im1, 
                                                                        rhovy_int, rhovy_int_im1, rhovy_int_km1, 
                                                                        rhovz_int, rhovz_int_km1, 
                                                                        Bx_int, Bx_int_im1, 
                                                                        By, By_int, By_int_im1, By_int_km1, 
                                                                        Bz_int, Bz_int_km1, 
                                                                        dt, dx, dy, dz);
            // Bz
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] =  LaxWendroffAdvBZLocal(rho_int, rho_int_im1, rho_int_jm1, 
                                                                        rhovx_int, rhovx_int_im1, 
                                                                        rhovy_int, rhovy_int_jm1, 
                                                                        rhovz_int, rhovz_int_im1, rhovz_int_jm1, 
                                                                        Bx_int, Bx_int_im1, 
                                                                        By_int, By_int_jm1, 
                                                                        Bz, Bz_int, Bz_int_im1, Bz_int_jm1, 
                                                                        dt, dx, dy, dz); 
            // e
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] =  LaxWendroffAdvELocal(rho_int, rho_int_im1, rho_int_jm1, rho_int_km1, 
                                                                        rhovx_int, rhovx_int_im1, 
                                                                        rhovy_int, rhovy_int_jm1, 
                                                                        rhovz_int, rhovz_int_km1,
                                                                        Bx_int, Bx_int_im1, 
                                                                        By_int, By_int_jm1, 
                                                                        Bz_int, Bz_int_km1, 
                                                                        e, e_int, e_int_im1, e_int_jm1, e_int_km1,
                                                                        p_ijk, p_im1jk, p_ijm1k, p_ijkm1, 
                                                                        Bsq_ijk, Bsq_im1jk, Bsq_ijm1k, Bsq_ijkm1, 
                                                                        Bdotu_ijk, Bdotu_im1jk, Bdotu_ijm1k, Bdotu_ijkm1,  
                                                                        dt, dx, dy, dz);
        }
        return;
    }

void LaunchFluidAdvanceLocalNoDiff(float* fluidvar, const float* intvar, const KernelConfig& kcfg)
{
    FluidAdvanceLocalNoDiff<<<kcfg.gridDim, kcfg.blockDim>>>(fluidvar, intvar, kcfg.dt, kcfg.dx, kcfg.dy, kcfg.dz, kcfg.Nx, kcfg.Ny, kcfg.Nz);
    return;
}

__global__ void FluidAdvanceNoDiffSHMEM(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        int k = threadIdx.z + blockDim.z * blockIdx.z;

        int cube_size = Nx * Ny * Nz;
        
        /* NEED TO READ DATA INTO SHARED MEMORY */
        // __shared__ float shmem_intvar[8 * ((blockDim.x * blockDim.y * blockDim.z) + 2 * (blockDim.x * blockDim.y + blockDim.y * blockDim.z + blockDim.x * blockDim.z))];
        
        __shared__ float shmem_intvar[8 * (blockDim.x + 2) * (blockDim.y +2) * (blockDim.z + 2)];
        
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int tz = threadIdx.z;

        /* Every thread deals completely with a single location - how to deal with halos? */
        for (int ivf = 0; ivf < 8; ivf++) { 
            shmem_intvar[IDX3D(tx, ty, tz, blockDim.x, blockDim.y, blockDim.z) + ivf * cube_size] = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + ivf * cube_size];
        }
        __syncthreads();

        /* A better idea than to do all this microwork with the halos is just to read in the (blockDim.x + 2) * (blockDim.y + 2) * (blockDim.z + 2) cube */
        // __shared__ float shmem_halo_right[blockDim.x * blockDim.z * 8];
        // __shared__ float shmem_halo_left[blockDim.x * blockDim.z * 8];
        // __shared__ float shmem_halo_top[blockDim.y * blockDim.z * 8];
        // __shared__ float shmem_halo_bottom[blockDim.y * blockDim.z * 8];
        // __shared__ float shmem_halo_front[blockDim.x * blockDim.y * 8];
        // __shared__ float shmem_halo_back[blockDim.x * blockDim.y * 8];

        return;
    }

/* WRITE */
__global__ void FluidAdvanceStrideNoDiffSHMEM(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        int k = threadIdx.z + blockDim.z * blockIdx.z;

        int cube_size = Nx * Ny * Nz;
        
        // Placeholder for future shared memory implementation
        

        return;
    }

// Microkernels 
// 32 registers per thread
__global__ void FluidAdvanceMicroRhoLocalNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        int k = threadIdx.z + blockDim.z * blockIdx.z;

        int cube_size = Nx * Ny * Nz;

        float rho = 0.0;

        float rho_int = 0.0;
        
        float rhovx_int = 0.0, rhovx_int_im1 = 0.0;
        
        float rhovy_int = 0.0, rhovy_int_jm1 = 0.0;
        
        float rhovz_int = 0.0, rhovz_int_km1 = 0.0;

        if (i > 0 && i < Nx-1 && j > 0 && j < Ny-1 && k > 0 && k < Nz){ // Ignore front and sides
            rho = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)];

            rho_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz)];
            
            rhovx_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size];
            rhovx_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + cube_size];
            
            rhovy_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovy_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 2 * cube_size];
            
            rhovz_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size];
            rhovz_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 3 * cube_size];

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = LaxWendroffAdvRhoLocal(rho, rho_int, 
                                                        rhovx_int, rhovx_int_im1, 
                                                        rhovy_int, rhovy_int_jm1, 
                                                        rhovz_int, rhovz_int_km1, 
                                                        dt, dx, dy, dz); 
        }
        return;
    }

// 66 registers per thread
__global__ void FluidAdvanceMicroRhoVXLocalNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        int k = threadIdx.z + blockDim.z * blockIdx.z;

        int cube_size = Nx * Ny * Nz;

        float rhovx = 0.0;

        float rho_int = 0.0, rho_int_im1 = 0.0, rho_int_jm1 = 0.0, rho_int_km1 = 0.0;
        float rhovx_int = 0.0, rhovx_int_im1 = 0.0, rhovx_int_jm1 = 0.0, rhovx_int_km1 = 0.0; 
        float rhovy_int = 0.0, rhovy_int_im1 = 0.0, rhovy_int_jm1 = 0.0;
        float rhovz_int = 0.0, rhovz_int_im1 = 0.0, rhovz_int_km1 = 0.0;
        float Bx_int = 0.0, Bx_int_im1 = 0.0, Bx_int_jm1 = 0.0, Bx_int_km1 = 0.0;
        float By_int = 0.0, By_int_im1 = 0.0, By_int_jm1 = 0.0;
        float Bz_int = 0.0, Bz_int_im1 = 0.0, Bz_int_km1 = 0.0;

        float e_int = 0.0, e_int_im1 = 0.0;
        
        float p_ijk = 0.0, p_im1jk = 0.0, KE_ijk = 0.0, KE_im1jk = 0.0, Bsq_ijk = 0.0, Bsq_im1jk = 0.0;

        if (i > 0 && i < Nx-1 && j > 0 && j < Ny-1 && k > 0 && k < Nz){ // Ignore front and sides
            rhovx = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size];

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
            Bx_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 4 * cube_size];
            By_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 5 * cube_size];

            rho_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz)];
            rhovx_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + cube_size];
            rhovz_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 3 * cube_size];
            Bx_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 4 * cube_size];
            Bz_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 6 * cube_size];

            Bsq_ijk = B_sq_local(Bx_int, By_int, Bz_int);
            Bsq_im1jk = B_sq_local(Bx_int_im1, By_int_im1, Bz_int_im1);

            KE_ijk = KE_local(rho_int, rhovx_int, rhovy_int, rhovz_int);
            KE_im1jk = KE_local(rho_int_im1, rhovx_int_im1, rhovy_int_im1, rhovz_int_im1);

            p_ijk = p_local(e_int, Bsq_ijk, KE_ijk);
            p_im1jk = p_local(e_int_im1, Bsq_im1jk, KE_im1jk);

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] =  LaxWendroffAdvRhoVXLocal(rho_int, rho_int_im1, rho_int_jm1, rho_int_km1, 
                                                                    rhovx, rhovx_int, rhovx_int_im1, rhovx_int_jm1, rhovx_int_km1, 
                                                                    rhovy_int, rhovy_int_jm1, 
                                                                    rhovz_int, rhovz_int_km1, 
                                                                    Bx_int, Bx_int_im1, Bx_int_jm1, Bx_int_km1, 
                                                                    By_int, By_int_jm1, 
                                                                    Bz_int, Bz_int_km1, 
                                                                    p_ijk, p_im1jk, 
                                                                    Bsq_ijk, Bsq_im1jk, 
                                                                    dt, dx, dy, dz);
        }
        return;
    }

// 72 registers per thread
__global__ void FluidAdvanceMicroRhoVYLocalNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        int k = threadIdx.z + blockDim.z * blockIdx.z;

        int cube_size = Nx * Ny * Nz;

        float rhovy = 0.0;

        float rho_int = 0.0, rho_int_im1 = 0.0, rho_int_jm1 = 0.0, rho_int_km1 = 0.0;
        // float rhovx_int = 0.0, rhovx_int_im1 = 0.0, rhovx_int_jm1 = 0.0, rhovx_int_km1 = 0.0; 
        float rhovx_int = 0.0, rhovx_int_im1 = 0.0, rhovx_int_jm1 = 0.0; 
        float rhovy_int = 0.0, rhovy_int_im1 = 0.0, rhovy_int_jm1 = 0.0, rhovy_int_km1 = 0.0;
        float rhovz_int = 0.0, rhovz_int_jm1 = 0.0, rhovz_int_km1 = 0.0;
        // float Bx_int = 0.0, Bx_int_im1 = 0.0, Bx_int_jm1 = 0.0, Bx_int_km1 = 0.0;
        float Bx_int = 0.0, Bx_int_im1 = 0.0, Bx_int_jm1 = 0.0;
        float By_int = 0.0, By_int_im1 = 0.0, By_int_jm1 = 0.0, By_int_km1 = 0.0;
        float Bz_int = 0.0, Bz_int_jm1 = 0.0, Bz_int_km1 = 0.0;

        float e_int = 0.0, e_int_jm1 = 0.0;
        
        float p_ijk = 0.0, p_ijm1k = 0.0; 
        float KE_ijk = 0.0, KE_ijm1k = 0.0;
        float Bsq_ijk = 0.0, Bsq_ijm1k = 0.0;

        if (i > 0 && i < Nx-1 && j > 0 && j < Ny-1 && k > 0 && k < Nz){ // Ignore front and sides
            rhovy = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size];

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
            Bx_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + 4 * cube_size];
            By_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + 5 * cube_size];
            
            rho_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz)];
            rhovx_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + cube_size];
            rhovy_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 3 * cube_size];
            By_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 6 * cube_size];
            e_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 7 * cube_size];

            rho_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz)];
            rhovy_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 3 * cube_size];
            By_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 5 * cube_size];
            Bz_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 6 * cube_size];

            Bsq_ijk = B_sq_local(Bx_int, By_int, Bz_int);
            Bsq_ijm1k = B_sq_local(Bx_int_jm1, By_int_jm1, Bz_int_jm1);

            KE_ijk = KE_local(rho_int, rhovx_int, rhovy_int, rhovz_int);
            KE_ijm1k = KE_local(rho_int_jm1, rhovx_int_jm1, rhovy_int_jm1, rhovz_int_jm1);

            p_ijk = p_local(e_int, Bsq_ijk, KE_ijk);
            p_ijm1k = p_local(e_int_jm1, Bsq_ijm1k, KE_ijm1k);

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] =  LaxWendroffAdvRhoVYLocal(rho_int, rho_int_im1, rho_int_jm1, rho_int_km1, 
                                                                        rhovx_int, rhovx_int_im1, 
                                                                        rhovy, rhovy_int, rhovy_int_im1, rhovy_int_jm1, rhovy_int_km1, 
                                                                        rhovz_int, rhovz_int_km1, 
                                                                        Bx_int, Bx_int_im1, 
                                                                        By_int, By_int_im1, By_int_jm1, By_int_km1, 
                                                                        Bz_int, Bz_int_km1, 
                                                                        Bsq_ijk, Bsq_ijm1k, 
                                                                        p_ijk, p_ijm1k, 
                                                                        dt, dx, dy, dz); 
        }
        return;
    }

// 72 registers per thread
__global__ void FluidAdvanceMicroRhoVZLocalNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        int k = threadIdx.z + blockDim.z * blockIdx.z;

        int cube_size = Nx * Ny * Nz;

        float rhovz = 0.0;

        float rho_int = 0.0, rho_int_im1 = 0.0, rho_int_jm1 = 0.0, rho_int_km1 = 0.0;
        
        // float rhovx_int = 0.0, rhovx_int_im1 = 0.0, rhovx_int_jm1 = 0.0, rhovx_int_km1 = 0.0; 
        float rhovx_int = 0.0, rhovx_int_im1 = 0.0, rhovx_int_km1 = 0.0; 
        
        // float rhovy_int = 0.0, rhovy_int_im1 = 0.0, rhovy_int_jm1 = 0.0, rhovy_int_km1 = 0.0;
        float rhovy_int = 0.0, rhovy_int_jm1 = 0.0, rhovy_int_km1 = 0.0;
        
        // float rhovz_int = 0.0, rhovz_int_im1 = 0.0, rhovz_int_jm1 = 0.0, rhovz_int_km1 = 0.0;
        float rhovz_int = 0.0, rhovz_int_im1 = 0.0, rhovz_int_jm1 = 0.0, rhovz_int_km1 = 0.0;

        // float Bx_int = 0.0, Bx_int_im1 = 0.0, Bx_int_jm1 = 0.0, Bx_int_km1 = 0.0;
        float Bx_int = 0.0, Bx_int_im1 = 0.0, Bx_int_km1 = 0.0;
        
        // float By_int = 0.0, By_int_im1 = 0.0, By_int_jm1 = 0.0, By_int_km1 = 0.0;
        float By_int = 0.0, By_int_jm1 = 0.0, By_int_km1 = 0.0;
        
        // float Bz_int = 0.0, Bz_int_im1 = 0.0, Bz_int_jm1 = 0.0, Bz_int_km1 = 0.0;
        float Bz_int = 0.0, Bz_int_im1 = 0.0, Bz_int_jm1 = 0.0, Bz_int_km1 = 0.0;

        // float e_int_ijk = 0.0, e_int_im1jk = 0.0, e_int_ijm1k = 0.0, e_int_ijkm1 = 0.0;
        float e_int = 0.0, e_int_km1 = 0.0;
        
        // float p_ijk = 0.0, p_im1jk = 0.0, p_ijm1k = 0.0, p_ijm1k = 0.0; 
        float p_ijk = 0.0, p_ijkm1 = 0.0; 

        // float KE_ijk = 0.0, KE_im1jk = 0.0, KE_ijm1k = 0.0, KE_ijkm1 = 0.0; 
        float KE_ijk = 0.0, KE_ijkm1 = 0.0;
        
        // float Bsq_ijk = 0.0, Bsq_im1jk = 0.0, Bsq_ijm1k = 0.0, Bsq_ijkm1 = 0.0; 
        float Bsq_ijk = 0.0, Bsq_ijkm1 = 0.0;

        if (i > 0 && i < Nx-1 && j > 0 && j < Ny-1 && k > 0 && k < Nz){ // Ignore front and sides
            rhovz = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size];

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
            rhovz_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + 4 * cube_size];
            Bz_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + 6 * cube_size];

            rho_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz)];
            rhovy_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 2 * cube_size];
            By_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 6 * cube_size];

            rho_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz)];
            rhovx_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + cube_size];
            rhovy_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 3 * cube_size];
            Bx_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 4 * cube_size];
            By_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 5 * cube_size];
            Bz_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 6 * cube_size];
            e_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 7 * cube_size];

            Bsq_ijk = B_sq_local(Bx_int, By_int, Bz_int);
            Bsq_ijkm1 = B_sq_local(Bx_int_km1, By_int_km1, Bz_int_km1);

            KE_ijk = KE_local(rho_int, rhovx_int, rhovy_int, rhovz_int);
            KE_ijkm1 = KE_local(rho_int_km1, rhovx_int_km1, rhovy_int_km1, rhovz_int_km1);

            p_ijk = p_local(e_int, Bsq_ijk, KE_ijk);
            p_ijkm1 = p_local(e_int_km1, Bsq_ijkm1, KE_ijkm1);

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] =  LaxWendroffAdvRhoVZLocal(rho_int, rho_int_im1, rho_int_jm1, rho_int_km1, 
                                                                        rhovx_int, rhovx_int_im1, 
                                                                        rhovy_int, rhovy_int_jm1, 
                                                                        rhovz, rhovz_int, rhovz_int_im1, rhovz_int_jm1, rhovz_int_km1,
                                                                        Bx_int, Bx_int_im1, 
                                                                        By_int, By_int_jm1, 
                                                                        Bz_int, Bz_int_im1, Bz_int_jm1, Bz_int_km1, 
                                                                        p_ijk, p_ijkm1,
                                                                        Bsq_ijk, Bsq_ijkm1,
                                                                        dt, dx, dy, dz); 
        }
        return;
    }

// 56 registers per thread
__global__ void FluidAdvanceMicroBXLocalNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        int k = threadIdx.z + blockDim.z * blockIdx.z;

        int cube_size = Nx * Ny * Nz;

        float Bx = 0.0;

        // float rho_int = 0.0, rho_int_im1 = 0.0, rho_int_jm1 = 0.0, rho_int_km1 = 0.0;
        float rho_int = 0.0, rho_int_jm1 = 0.0, rho_int_km1 = 0.0;
        
        // float rhovx_int = 0.0, rhovx_int_im1 = 0.0, rhovx_int_jm1 = 0.0, rhovx_int_km1 = 0.0; 
        float rhovx_int = 0.0, rhovx_int_jm1 = 0.0, rhovx_int_km1 = 0.0; 
        
        // float rhovy_int = 0.0, rhovy_int_im1 = 0.0, rhovy_int_jm1 = 0.0, rhovy_int_km1 = 0.0;
        float rhovy_int = 0.0, rhovy_int_jm1 = 0.0;
        
        // float rhovz_int = 0.0, rhovz_int_im1 = 0.0, rhovz_int_jm1 = 0.0, rhovz_int_km1 = 0.0;
        float rhovz_int = 0.0, rhovz_int_km1 = 0.0;

        // float Bx_int = 0.0, Bx_int_im1 = 0.0, Bx_int_jm1 = 0.0, Bx_int_km1 = 0.0;
        float Bx_int = 0.0, Bx_int_jm1 = 0.0, Bx_int_km1 = 0.0;
        
        // float By_int = 0.0, By_int_im1 = 0.0, By_int_jm1 = 0.0, By_int_km1 = 0.0;
        float By_int = 0.0, By_int_jm1 = 0.0;
        
        // float Bz_int = 0.0, Bz_int_im1 = 0.0, Bz_int_jm1 = 0.0, Bz_int_km1 = 0.0;
        float Bz_int = 0.0, Bz_int_km1 = 0.0;

        // float e_int_ijk = 0.0, e_int_im1jk = 0.0, e_int_ijm1k = 0.0, e_int_ijkm1 = 0.0;
        // float e_int_ijk = 0.0, e_int_ijkm1 = 0.0;
        
        // float p_ijk = 0.0, p_im1jk = 0.0, p_ijm1k = 0.0, p_ijm1k = 0.0; 
        // float p_ijk = 0.0, p_ijkm1 = 0.0; 

        // float KE_ijk = 0.0, KE_im1jk = 0.0, KE_ijm1k = 0.0, KE_ijkm1 = 0.0; 
        // float KE_ijk = 0.0, KE_ijkm1 = 0.0;
        
        // float Bsq_ijk = 0.0, Bsq_im1jk = 0.0, Bsq_ijm1k = 0.0, Bsq_ijkm1 = 0.0; 
        // float Bsq_ijk = 0.0, Bsq_ijkm1 = 0.0;

        if (i > 0 && i < Nx-1 && j > 0 && j < Ny-1 && k > 0 && k < Nz){ // Ignore front and sides
            Bx = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size];

            rho_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz)];
            rhovx_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size];
            rhovy_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size];
            By_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size];

            rho_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz)];
            rhovx_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + cube_size];
            rhovy_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 2 * cube_size];
            Bx_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 4 * cube_size];
            By_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 5 * cube_size];
            
            rho_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz)];
            rhovx_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + cube_size];
            rhovz_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 3 * cube_size];
            Bx_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 4 * cube_size];
            Bz_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 6 * cube_size];

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] =  LaxWendroffAdvBXLocal(rho_int, rho_int_jm1, rho_int_km1, 
                                                                        rhovx_int, rhovx_int_jm1, rhovx_int_km1, 
                                                                        rhovy_int, rhovy_int_jm1, 
                                                                        rhovz_int, rhovz_int_km1, 
                                                                        Bx, Bx_int, Bx_int_jm1, Bx_int_km1, 
                                                                        By_int, By_int_jm1, 
                                                                        Bz_int, Bz_int_km1, 
                                                                        dt, dx, dy, dz); 
        }
        return;
    }

// 50 registers per thread
__global__ void FluidAdvanceMicroBYLocalNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        int k = threadIdx.z + blockDim.z * blockIdx.z;

        int cube_size = Nx * Ny * Nz;

        float By = 0.0;

        // float rho_int = 0.0, rho_int_im1 = 0.0, rho_int_jm1 = 0.0, rho_int_km1 = 0.0;
        float rho_int = 0.0, rho_int_im1 = 0.0, rho_int_km1 = 0.0;
        
        // float rhovx_int = 0.0, rhovx_int_im1 = 0.0, rhovx_int_jm1 = 0.0, rhovx_int_km1 = 0.0; 
        float rhovx_int = 0.0, rhovx_int_im1 = 0.0; 
        
        // float rhovy_int = 0.0, rhovy_int_im1 = 0.0, rhovy_int_jm1 = 0.0, rhovy_int_km1 = 0.0;
        float rhovy_int = 0.0, rhovy_int_im1 = 0.0, rhovy_int_km1 = 0.0;
        
        // float rhovz_int = 0.0, rhovz_int_im1 = 0.0, rhovz_int_jm1 = 0.0, rhovz_int_km1 = 0.0;
        float rhovz_int = 0.0, rhovz_int_km1 = 0.0;

        // float Bx_int = 0.0, Bx_int_im1 = 0.0, Bx_int_jm1 = 0.0, Bx_int_km1 = 0.0;
        float Bx_int = 0.0, Bx_int_im1 = 0.0;
        
        // float By_int = 0.0, By_int_im1 = 0.0, By_int_jm1 = 0.0, By_int_km1 = 0.0;
        float By_int = 0.0, By_int_im1 = 0.0, By_int_km1 = 0.0;
        
        // float Bz_int = 0.0, Bz_int_im1 = 0.0, Bz_int_jm1 = 0.0, Bz_int_km1 = 0.0;
        float Bz_int = 0.0, Bz_int_km1 = 0.0;

        // float e_int_ijk = 0.0, e_int_im1jk = 0.0, e_int_ijm1k = 0.0, e_int_ijkm1 = 0.0;
        // float e_int_ijk = 0.0, e_int_ijkm1 = 0.0;
        
        // float p_ijk = 0.0, p_im1jk = 0.0, p_ijm1k = 0.0, p_ijm1k = 0.0; 
        // float p_ijk = 0.0, p_ijkm1 = 0.0; 

        // float KE_ijk = 0.0, KE_im1jk = 0.0, KE_ijm1k = 0.0, KE_ijkm1 = 0.0; 
        // float KE_ijk = 0.0, KE_ijkm1 = 0.0;
        
        // float Bsq_ijk = 0.0, Bsq_im1jk = 0.0, Bsq_ijm1k = 0.0, Bsq_ijkm1 = 0.0; 
        // float Bsq_ijk = 0.0, Bsq_ijkm1 = 0.0;

        if (i > 0 && i < Nx-1 && j > 0 && j < Ny-1 && k > 0 && k < Nz){ // Ignore front and sides
            By = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size];

            rho_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz)];
            rhovx_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size];
            rhovy_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size];
            By_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size];

            rho_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz)];
            rhovx_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + cube_size];
            rhovy_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + 2 * cube_size];
            Bx_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + 4 * cube_size];
            By_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + 5 * cube_size];
            
            rho_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz)];
            rhovy_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 3 * cube_size];
            By_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 5 * cube_size];
            Bz_int_km1 = intvar[IDX3D(i, j, k-1, Nx, Ny, Nz) + 6 * cube_size];

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] =  LaxWendroffAdvBYLocal(rho_int, rho_int_im1, rho_int_km1, 
                                                                        rhovx_int, rhovx_int_im1, 
                                                                        rhovy_int, rhovy_int_im1, rhovy_int_km1, 
                                                                        rhovz_int, rhovz_int_km1, 
                                                                        Bx_int, Bx_int_im1, 
                                                                        By, By_int, By_int_im1, By_int_km1, 
                                                                        Bz_int, Bz_int_km1, 
                                                                        dt, dx, dy, dz);
        }
        return;
    }

// 50 registers per thread
__global__ void FluidAdvanceMicroBZLocalNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        int k = threadIdx.z + blockDim.z * blockIdx.z;

        int cube_size = Nx * Ny * Nz;

        float Bz = 0.0;

        // float rho_int = 0.0, rho_int_im1 = 0.0, rho_int_jm1 = 0.0, rho_int_km1 = 0.0;
        float rho_int = 0.0, rho_int_im1 = 0.0, rho_int_jm1 = 0.0;
        
        // float rhovx_int = 0.0, rhovx_int_im1 = 0.0, rhovx_int_jm1 = 0.0, rhovx_int_km1 = 0.0; 
        float rhovx_int = 0.0, rhovx_int_im1 = 0.0; 
        
        // float rhovy_int = 0.0, rhovy_int_im1 = 0.0, rhovy_int_jm1 = 0.0, rhovy_int_km1 = 0.0;
        float rhovy_int = 0.0, rhovy_int_jm1 = 0.0;
        
        // float rhovz_int = 0.0, rhovz_int_im1 = 0.0, rhovz_int_jm1 = 0.0, rhovz_int_km1 = 0.0;
        float rhovz_int = 0.0, rhovz_int_im1 = 0.0, rhovz_int_jm1 = 0.0;

        // float Bx_int = 0.0, Bx_int_im1 = 0.0, Bx_int_jm1 = 0.0, Bx_int_km1 = 0.0;
        float Bx_int = 0.0, Bx_int_im1 = 0.0;
        
        // float By_int = 0.0, By_int_im1 = 0.0, By_int_jm1 = 0.0, By_int_km1 = 0.0;
        float By_int = 0.0, By_int_jm1 = 0.0;
        
        // float Bz_int = 0.0, Bz_int_im1 = 0.0, Bz_int_jm1 = 0.0, Bz_int_km1 = 0.0;
        float Bz_int = 0.0, Bz_int_im1 = 0.0, Bz_int_jm1 = 0.0;

        // float e_int_ijk = 0.0, e_int_im1jk = 0.0, e_int_ijm1k = 0.0, e_int_ijkm1 = 0.0;
        // float e_int_ijk = 0.0, e_int_ijkm1 = 0.0;
        
        // float p_ijk = 0.0, p_im1jk = 0.0, p_ijm1k = 0.0, p_ijm1k = 0.0; 
        // float p_ijk = 0.0, p_ijkm1 = 0.0; 

        // float KE_ijk = 0.0, KE_im1jk = 0.0, KE_ijm1k = 0.0, KE_ijkm1 = 0.0; 
        // float KE_ijk = 0.0, KE_ijkm1 = 0.0;
        
        // float Bsq_ijk = 0.0, Bsq_im1jk = 0.0, Bsq_ijm1k = 0.0, Bsq_ijkm1 = 0.0; 
        // float Bsq_ijk = 0.0, Bsq_ijkm1 = 0.0;

        if (i > 0 && i < Nx-1 && j > 0 && j < Ny-1 && k > 0 && k < Nz){ // Ignore front and sides
            Bz = fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size];

            rho_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz)];
            rhovx_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size];
            rhovy_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size];
            By_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz_int = intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size];

            rho_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz)];
            rhovx_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + cube_size];
            rhovz_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + 3 * cube_size];
            Bx_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + 4 * cube_size];
            Bz_int_im1 = intvar[IDX3D(i-1, j, k, Nx, Ny, Nz) + 6 * cube_size];
            
            rho_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz)];
            rhovy_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 2 * cube_size];
            rhovz_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 3 * cube_size];
            By_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 5 * cube_size];
            Bz_int_jm1 = intvar[IDX3D(i, j-1, k, Nx, Ny, Nz) + 6 * cube_size];

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] =  LaxWendroffAdvBZLocal(rho_int, rho_int_im1, rho_int_jm1, 
                                                                        rhovx_int, rhovx_int_im1, 
                                                                        rhovy_int, rhovy_int_jm1, 
                                                                        rhovz_int, rhovz_int_im1, rhovz_int_jm1, 
                                                                        Bx_int, Bx_int_im1, 
                                                                        By_int, By_int_jm1, 
                                                                        Bz, Bz_int, Bz_int_im1, Bz_int_jm1, 
                                                                        dt, dx, dy, dz); 
        }
        return;
    }

// 90 registers per thread
__global__ void FluidAdvanceMicroELocalNoDiff(float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        int k = threadIdx.z + blockDim.z * blockIdx.z;

        int cube_size = Nx * Ny * Nz;

        float e = 0.0;

        float rho_int = 0.0, rho_int_im1 = 0.0, rho_int_jm1 = 0.0, rho_int_km1 = 0.0;
        // float rho_int = 0.0, rho_int_im1 = 0.0, rho_int_jm1 = 0.0;
        
        float rhovx_int = 0.0, rhovx_int_im1 = 0.0, rhovx_int_jm1 = 0.0, rhovx_int_km1 = 0.0; 
        // float rhovx_int = 0.0, rhovx_int_im1 = 0.0; 
        
        float rhovy_int = 0.0, rhovy_int_im1 = 0.0, rhovy_int_jm1 = 0.0, rhovy_int_km1 = 0.0;
        // float rhovy_int = 0.0, rhovy_int_jm1 = 0.0;
        
        float rhovz_int = 0.0, rhovz_int_im1 = 0.0, rhovz_int_jm1 = 0.0, rhovz_int_km1 = 0.0;
        // float rhovz_int = 0.0, rhovz_int_im1 = 0.0, rhovz_int_jm1 = 0.0;

        float Bx_int = 0.0, Bx_int_im1 = 0.0, Bx_int_jm1 = 0.0, Bx_int_km1 = 0.0;
        // float Bx_int = 0.0, Bx_int_im1 = 0.0;
        
        float By_int = 0.0, By_int_im1 = 0.0, By_int_jm1 = 0.0, By_int_km1 = 0.0;
        // float By_int = 0.0, By_int_jm1 = 0.0;
        
        float Bz_int = 0.0, Bz_int_im1 = 0.0, Bz_int_jm1 = 0.0, Bz_int_km1 = 0.0;
        // float Bz_int = 0.0, Bz_int_im1 = 0.0, Bz_int_jm1 = 0.0;

        float e_int = 0.0, e_int_im1 = 0.0, e_int_jm1 = 0.0, e_int_km1 = 0.0;
        // float e_int_ijk = 0.0, e_int_ijkm1 = 0.0;
        
        float p_ijk = 0.0, p_im1jk = 0.0, p_ijm1k = 0.0, p_ijkm1 = 0.0; 
        // float p_ijk = 0.0, p_ijkm1 = 0.0; 

        float KE_ijk = 0.0, KE_im1jk = 0.0, KE_ijm1k = 0.0, KE_ijkm1 = 0.0; 
        // float KE_ijk = 0.0, KE_ijkm1 = 0.0;
        
        float Bdotu_ijk = 0.0, Bdotu_im1jk = 0.0, Bdotu_ijm1k = 0.0, Bdotu_ijkm1 = 0.0; 

        float Bsq_ijk = 0.0, Bsq_im1jk = 0.0, Bsq_ijm1k = 0.0, Bsq_ijkm1 = 0.0; 
        // float Bsq_ijk = 0.0, Bsq_ijkm1 = 0.0;

        if (i > 0 && i < Nx-1 && j > 0 && j < Ny-1 && k > 0 && k < Nz){ // Ignore front and sides
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

            Bsq_ijk = B_sq_local(Bx_int, By_int, Bz_int);
            Bsq_im1jk = B_sq_local(Bx_int_im1, By_int_im1, Bz_int_im1);
            Bsq_ijm1k = B_sq_local(Bx_int_jm1, By_int_jm1, Bz_int_jm1);
            Bsq_ijkm1 = B_sq_local(Bx_int_km1, By_int_km1, Bz_int_km1);

            Bdotu_ijk = B_dot_u_local(rho_int, rhovx_int, rhovy_int, rhovz_int, Bx_int, By_int, Bz_int);
            Bdotu_im1jk = B_dot_u_local(rho_int_im1, rhovx_int_im1, rhovy_int_im1, rhovz_int_im1, Bx_int_im1, By_int_im1, Bz_int_im1);
            Bdotu_ijm1k = B_dot_u_local(rho_int_jm1, rhovx_int_jm1, rhovy_int_jm1, rhovz_int_jm1, Bx_int_jm1, By_int_jm1, Bz_int_jm1);
            Bdotu_ijkm1 = B_dot_u_local(rho_int_km1, rhovx_int_km1, rhovy_int_km1, rhovz_int_km1, Bx_int_km1, By_int_km1, Bz_int_km1);

            KE_ijk = KE_local(rho_int, rhovx_int, rhovy_int, rhovz_int);
            KE_im1jk = KE_local(rho_int_im1, rhovx_int_im1, rhovy_int_im1, rhovz_int_im1);
            KE_ijm1k = KE_local(rho_int_jm1, rhovx_int_jm1, rhovy_int_jm1, rhovz_int_jm1);
            KE_ijkm1 = KE_local(rho_int_km1, rhovx_int_km1, rhovy_int_km1, rhovz_int_km1);

            p_ijk = p_local(e_int, Bsq_ijk, KE_ijk);
            p_im1jk = p_local(e_int_im1, Bsq_im1jk, KE_im1jk);
            p_ijm1k = p_local(e_int_jm1, Bsq_ijm1k, KE_ijm1k);
            p_ijkm1 = p_local(e_int_km1, Bsq_ijkm1, KE_ijkm1);

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] =  LaxWendroffAdvELocal(rho_int, rho_int_im1, rho_int_jm1, rho_int_km1, 
                                                                        rhovx_int, rhovx_int_im1, 
                                                                        rhovy_int, rhovy_int_jm1, 
                                                                        rhovz_int, rhovz_int_km1,
                                                                        Bx_int, Bx_int_im1, 
                                                                        By_int, By_int_jm1, 
                                                                        Bz_int, Bz_int_km1, 
                                                                        e, e_int, e_int_im1, e_int_jm1, e_int_km1,
                                                                        p_ijk, p_im1jk, p_ijm1k, p_ijkm1, 
                                                                        Bsq_ijk, Bsq_im1jk, Bsq_ijm1k, Bsq_ijkm1, 
                                                                        Bdotu_ijk, Bdotu_im1jk, Bdotu_ijm1k, Bdotu_ijkm1,  
                                                                        dt, dx, dy, dz);
        }
        return;
    }

// Device kernels
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

__device__ float LaxWendroffAdvRhoVXLocal(const float rho_int, const float rho_int_im1, const float rho_int_jm1, const float rho_int_km1,
    const float rhovx, const float rhovx_int, const float rhovx_int_im1, const float rhovx_int_jm1, const float rhovx_int_km1,
    const float rhovy_int, const float rhovy_int_jm1, 
    const float rhovz_int, const float rhovz_int_km1,
    const float Bx_int, const float Bx_int_im1, const float Bx_int_jm1, const float Bx_int_km1,
    const float By_int, const float By_int_jm1, 
    const float Bz_int, const float Bz_int_km1, 
    const float p_ijk, const float p_im1jk, 
    const float Bsq_ijk, const float Bsq_im1jk, 
    const float dt, const float dx, const float dy, const float dz)
    {
        return 0.5 * (rhovx + rhovx_int)
        - 0.5 * (dt / dx) * (XFluxRhoVX(rho_int, rhovx_int, Bx_int, p_ijk, Bsq_ijk) - XFluxRhoVX(rho_int_im1, rhovx_int_im1, Bx_int_im1, p_im1jk, Bsq_im1jk))
        - 0.5 * (dt / dy) * (YFluxRhoVX(rho_int, rhovx_int, rhovy_int, Bx_int, By_int) - YFluxRhoVX(rho_int_jm1, rhovx_int_jm1, rhovy_int_jm1, Bx_int_jm1, By_int_jm1))
        - 0.5 * (dt / dz) * (ZFluxRhoVX(rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int) - ZFluxRhoVX(rho_int_km1, rhovx_int_km1, rhovz_int_km1, Bx_int_km1, Bz_int_km1));
    }

__device__ float LaxWendroffAdvRhoVYLocal(const float rho_int, const float rho_int_im1, const float rho_int_jm1, const float rho_int_km1,
    const float rhovx_int, const float rhovx_int_im1,
    const float rhovy, const float rhovy_int, const float rhovy_int_im1, const float rhovy_int_jm1, const float rhovy_int_km1,
    const float rhovz_int, const float rhovz_int_km1,
    const float Bx_int, const float Bx_int_im1, 
    const float By_int, const float By_int_im1, const float By_int_jm1, const float By_int_km1,
    const float Bz_int, const float Bz_int_km1, 
    const float Bsq_ijk, const float Bsq_ijm1k,
    const float p_ijk, const float p_ijm1k,
    const float dt, const float dx, const float dy, const float dz)
    {
        return 0.5 * (rhovy + rhovy_int)
        - 0.5 * (dt / dx) * (XFluxRhoVY(rho_int, rhovx_int, rhovy_int, Bx_int, By_int) - XFluxRhoVY(rho_int_im1, rhovx_int_im1, rhovy_int_im1, Bx_int_im1, By_int_im1))
        - 0.5 * (dt / dy) * (YFluxRhoVY(rho_int, rhovy_int, By_int, p_ijk, Bsq_ijk) - YFluxRhoVY(rho_int_jm1, rhovy_int_jm1, By_int_jm1, p_ijm1k, Bsq_ijm1k))
        - 0.5 * (dt / dz) * (ZFluxRhoVY(rho_int, rhovy_int, rhovz_int, By_int, Bz_int) - ZFluxRhoVY(rho_int_km1, rhovy_int_km1, rhovz_int_km1, By_int_km1, Bz_int_km1));
    }

__device__ float LaxWendroffAdvRhoVZLocal(const float rho_int, const float rho_int_im1, const float rho_int_jm1, const float rho_int_km1,
    const float rhovx_int, const float rhovx_int_im1,  
    const float rhovy_int, const float rhovy_int_jm1, 
    const float rhovz, const float rhovz_int, const float rhovz_int_im1, const float rhovz_int_jm1, const float rhovz_int_km1, 
    const float Bx_int, const float Bx_int_im1,
    const float By_int, const float By_int_jm1, 
    const float Bz_int, const float Bz_int_im1, const float Bz_int_jm1, const float Bz_int_km1,  
    const float p_ijk, const float p_ijkm1, 
    const float Bsq_ijk, const float Bsq_ijkm1,
    const float dt, const float dx, const float dy, const float dz)
    {
        return 0.5 * (rhovz + rhovz_int)
        - 0.5 * (dt / dx) * (XFluxRhoVZ(rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int) - XFluxRhoVZ(rho_int_im1, rhovx_int_im1, rhovz_int_im1, Bx_int_im1, Bz_int_im1))
        - 0.5 * (dt / dy) * (YFluxRhoVZ(rho_int, rhovy_int, rhovz_int, By_int, Bz_int) - YFluxRhoVZ(rho_int_jm1, rhovy_int_jm1, rhovz_int_jm1, By_int_jm1, Bz_int_jm1))
        - 0.5 * (dt / dz) * (ZFluxRhoVZ(rho_int, rhovz_int, Bz_int, p_ijk, Bsq_ijk) - ZFluxRhoVZ(rho_int_km1, rhovz_int_km1, Bz_int_km1, p_ijkm1, Bsq_ijkm1));
    }

__device__ float LaxWendroffAdvBXLocal(const float rho_int, const float rho_int_jm1, const float rho_int_km1, 
    const float rhovx_int, const float rhovx_int_jm1, const float rhovx_int_km1,  
    const float rhovy_int, const float rhovy_int_jm1, 
    const float rhovz_int, const float rhovz_int_km1, 
    const float Bx, const float Bx_int, const float Bx_int_jm1, const float Bx_int_km1,
    const float By_int, const float By_int_jm1, 
    const float Bz_int, const float Bz_int_km1, 
    const float dt, const float dx, const float dy, const float dz)
    {
        return 0.5 * (Bx + Bx_int)
        - 0.5 * (dt / dx) * (XFluxBX() - XFluxBX())
        - 0.5 * (dt / dy) * (YFluxBX(rho_int, rhovx_int, rhovy_int, Bx_int, By_int) - YFluxBX(rho_int_jm1, rhovx_int_jm1, rhovy_int_jm1, Bx_int_jm1, By_int_jm1))
        - 0.5 * (dt / dz) * (ZFluxBX(rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int) - ZFluxBX(rho_int_km1, rhovx_int_km1, rhovz_int_km1, Bx_int_km1, Bz_int_km1));
    }

__device__ float LaxWendroffAdvBYLocal(const float rho_int, const float rho_int_im1, const float rho_int_km1,  
    const float rhovx_int, const float rhovx_int_im1,  
    const float rhovy_int, const float rhovy_int_im1, const float rhovy_int_km1, 
    const float rhovz_int, const float rhovz_int_km1, 
    const float Bx_int, const float Bx_int_im1,
    const float By, const float By_int, const float By_int_im1, const float By_int_km1, 
    const float Bz_int, const float Bz_int_km1, 
    const float dt, const float dx, const float dy, const float dz)
    {
        return 0.5 * (By + By_int)
        - 0.5 * (dt / dx) * (XFluxBY(rho_int, rhovx_int, rhovy_int, Bx_int, By_int) - XFluxBY(rho_int_im1, rhovx_int_im1, rhovy_int_im1, Bx_int_im1, By_int_im1))
        - 0.5 * (dt / dy) * (YFluxBY() - YFluxBY())
        - 0.5 * (dt / dz) * (ZFluxBY(rho_int, rhovy_int, rhovz_int, By_int, Bz_int) - ZFluxBY(rho_int_km1, rhovy_int_km1, rhovz_int_km1, By_int_km1, Bz_int_km1));
    }

__device__ float LaxWendroffAdvBZLocal(const float rho_int, const float rho_int_im1, const float rho_int_jm1, 
    const float rhovx_int, const float rhovx_int_im1,
    const float rhovy_int, const float rhovy_int_jm1,    
    const float rhovz_int, const float rhovz_int_im1, const float rhovz_int_jm1, 
    const float Bx_int, const float Bx_int_im1,  
    const float By_int, const float By_int_jm1, 
    const float Bz, const float Bz_int, const float Bz_int_im1, const float Bz_int_jm1, 
    const float dt, const float dx, const float dy, const float dz)
    {
        return 0.5 * (Bz + Bz_int)
        - 0.5 * (dt / dx) * (XFluxBZ(rho_int, rhovx_int, rhovz_int, Bx_int, Bz_int) - XFluxBZ(rho_int_im1, rhovx_int_im1, rhovz_int_im1, Bx_int_im1, Bz_int_im1))
        - 0.5 * (dt / dy) * (YFluxBZ(rho_int, rhovy_int, rhovz_int, By_int, Bz_int) - YFluxBZ(rho_int_jm1, rhovy_int_jm1, rhovz_int_jm1, By_int_jm1, Bz_int_jm1))
        - 0.5 * (dt / dz) * (ZFluxBZ() - ZFluxBZ());
    }

__device__ float LaxWendroffAdvELocal(const float rho_int, const float rho_int_im1, const float rho_int_jm1, const float rho_int_km1, 
    const float rhovx_int, const float rhovx_int_im1,
    const float rhovy_int, const float rhovy_int_jm1, 
    const float rhovz_int, const float rhovz_int_km1, 
    const float Bx_int, const float Bx_int_im1, 
    const float By_int, const float By_int_jm1, 
    const float Bz_int, const float Bz_int_km1, 
    const float e, const float e_int, const float e_int_im1, const float e_int_jm1, const float e_int_km1, 
    const float p_ijk, const float p_im1jk, const float p_ijm1k, const float p_ijkm1, 
    const float Bsq_ijk, const float Bsq_im1jk, const float Bsq_ijm1k, const float Bsq_ijkm1, 
    const float Bdotu_ijk, const float Bdotu_im1jk, const float Bdotu_ijm1k, const float Bdotu_ijkm1, 
    const float dt, const float dx, const float dy, const float dz)
    {
        return 0.5 * (e + e_int)
        - 0.5 * (dt / dx) * (XFluxE(rho_int, rhovx_int, Bx_int, e_int, p_ijk, Bsq_ijk, Bdotu_ijk) - XFluxE(rho_int_im1, rhovx_int_im1, Bx_int_im1, e_int_im1, p_im1jk, Bsq_im1jk, Bdotu_im1jk))
        - 0.5 * (dt / dy) * (YFluxE(rho_int, rhovy_int, By_int, e_int, p_ijk, Bsq_ijk, Bdotu_ijk) - YFluxE(rho_int_jm1, rhovy_int_jm1, By_int_jm1, e_int_jm1, p_ijm1k, Bsq_ijm1k, Bdotu_ijm1k))
        - 0.5 * (dt / dz) * (ZFluxE(rho_int, rhovz_int, Bz_int, e_int, p_ijk, Bsq_ijk, Bdotu_ijk) - ZFluxE(rho_int_km1, rhovz_int_km1, Bz_int_km1, e_int_km1, p_ijkm1, Bsq_ijkm1, Bdotu_ijkm1));
    }

// Implement corrector step
// THESE KERNELS THRASH THE CACHE
__device__ float LaxWendroffAdvRho(const int i, const int j, const int k, 
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] + intvar[IDX3D(i, j, k, Nx, Ny, Nz)])
        - 0.5 * (dt / dx) * (XFluxRho(i, j, k, intvar, Nx, Ny, Nz) - XFluxRho(i-1, j, k, intvar, Nx, Ny, Nz))
        - 0.5 * (dt / dy) * (YFluxRho(i, j, k, intvar, Nx, Ny, Nz) - YFluxRho(i, j-1, k, intvar, Nx, Ny, Nz))
        - 0.5 * (dt / dz) * (ZFluxRho(i, j, k, intvar, Nx, Ny, Nz) - ZFluxRho(i, j, k-1, intvar, Nx, Ny, Nz));
    }

__device__ float LaxWendroffAdvRhoVX(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size])
                - 0.5 * (dt / dx) * (XFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVX(i-1, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz) - YFluxRhoVX(i, j-1, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxRhoVX(i, j, k, intvar, Nx, Ny, Nz) - ZFluxRhoVX(i, j, k-1, intvar, Nx, Ny, Nz));  
    }

__device__ float LaxWendroffAdvRhoVY(const int i, const int j, const int k, 
    const float* fluidvar, const float* intvar,  
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size])
                - 0.5 * (dt / dx) * (XFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVY(i-1, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz) - YFluxRhoVY(i, j-1, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxRhoVY(i, j, k, intvar, Nx, Ny, Nz) - ZFluxRhoVY(i, j, k-1, intvar, Nx, Ny, Nz)); 
    }

__device__ float LaxWendroffAdvRhoVZ(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size])
                - 0.5 * (dt / dx) * (XFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz) - XFluxRhoVZ(i-1, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz) - YFluxRhoVZ(i, j-1, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxRhoVZ(i, j, k, intvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, j, k-1, intvar, Nx, Ny, Nz)); 
    }

__device__ float LaxWendroffAdvBX(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size])
                - 0.5 * (dt / dx) * (XFluxBX(i, j, k, intvar, Nx, Ny, Nz) - XFluxBX(i-1, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxBX(i, j, k, intvar, Nx, Ny, Nz) - YFluxBX(i, j-1, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxBX(i, j, k, intvar, Nx, Ny, Nz) - ZFluxBX(i, j, k-1, intvar, Nx, Ny, Nz)); 
    }

__device__ float LaxWendroffAdvBY(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size])
                - 0.5 * (dt / dx) * (XFluxBY(i, j, k, intvar, Nx, Ny, Nz) - XFluxBY(i-1, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxBY(i, j, k, intvar, Nx, Ny, Nz) - YFluxBY(i, j-1, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxBY(i, j, k, intvar, Nx, Ny, Nz) - ZFluxBY(i, j, k-1, intvar, Nx, Ny, Nz)); 
    }

__device__ float LaxWendroffAdvBZ(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size])
                - 0.5 * (dt / dx) * (XFluxBZ(i, j, k, intvar, Nx, Ny, Nz) - XFluxBZ(i-1, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxBZ(i, j, k, intvar, Nx, Ny, Nz) - YFluxBZ(i, j-1, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxBZ(i, j, k, intvar, Nx, Ny, Nz) - ZFluxBZ(i, j, k-1, intvar, Nx, Ny, Nz)); 
    }

__device__ float LaxWendroffAdvE(const int i, const int j, const int k,
    const float* fluidvar, const float* intvar, 
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int cube_size = Nx * Ny * Nz;
        return 0.5 * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] + intvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size])
                - 0.5 * (dt / dx) * (XFluxE(i, j, k, intvar, Nx, Ny, Nz) - XFluxE(i-1, j, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dy) * (YFluxE(i, j, k, intvar, Nx, Ny, Nz) - YFluxE(i, j-1, k, intvar, Nx, Ny, Nz))
                - 0.5 * (dt / dz) * (ZFluxE(i, j, k, intvar, Nx, Ny, Nz) - ZFluxE(i, j, k-1, intvar, Nx, Ny, Nz)); 
    }

/* 
LaxWendroffAdvVAR2 functions use flux functions that do not access global/shared memory, but they themselves DO
Therefore, it is likely that an implementation which leverages them will thrash the cache
*/
// __device__ float LaxWendroffAdvRho2(const int i, const int j, const int k,
//     const float* fluidvar, const float* intvar, 
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     { 
//         /* Put the code for advancing the fluid state here */
//     }

// __device__ float LaxWendroffAdvRhoVX2(const int i, const int j, const int k,
//     const float* fluidvar, const float* intvar, 
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         /* Put the code for advancing the fluid state here */ 
//     }

// __device__ float LaxWendroffAdvRhoVY2(const int i, const int j, const int k, 
//     const float* fluidvar, const float* intvar,  
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//           /* Put the code for advancing the fluid state here */
//     }

// __device__ float LaxWendroffAdvRhoVZ2(const int i, const int j, const int k,
//     const float* fluidvar, const float* intvar, 
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         /* Put the code for advancing the fluid state here */
//     }

// __device__ float LaxWendroffAdvBX2(const int i, const int j, const int k,
//     const float* fluidvar, const float* intvar, 
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         /* Put the code for advancing the fluid state here */
//     }

// __device__ float LaxWendroffAdvBY2(const int i, const int j, const int k,
//     const float* fluidvar, const float* intvar, 
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         /* Put the code for advancing the fluid state here */
//     }

// __device__ float LaxWendroffAdvBZ2(const int i, const int j, const int k,
//     const float* fluidvar, const float* intvar, 
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         /* Put the code for advancing the fluid state here */
//     }

// __device__ float LaxWendroffAdvE2(const int i, const int j, const int k,
//     const float* fluidvar, const float* intvar, 
//     const float dt, const float dx, const float dy, const float dz,
//     const int Nx, const int Ny, const int Nz)
//     {
//         /* Put the code for advancing the fluid state here */
//     }