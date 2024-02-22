#ifndef KERNELS_OD_DECL
#define KERNELS_OD_DECL

#include <math.h>

#include "kernels_od.cuh"

// row-major, column-minor order
#define IDX3D(i, j, k, Nx, Ny, Nz) (k * (Nx * Ny) + i * Nx + j)

/* DONT FORGET NUMERICAL DIFFUSION */
__global__ void FluidAdvance(float* rho_np1, float* rhovx_np1, float* rhovy_np1, float* rhovz_np1, float* Bx_np1, float* By_np1, float* Bz_np1, float* e_np1,
     const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, const float* Bx, const float* By, const float* Bz, const float* e,
     const float D, const float dt, const float dx, const float dy, const float dz, 
     const int Nx, const int Ny, const int Nz)
    {
        // Execution configuration boilerplate
        int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;
        int xthreads = gridDim.x * blockDim.x;
        int ythreads = gridDim.y * blockDim.y;
        int zthreads = gridDim.z * blockDim.z;

        /* 
        This all is getting re-declared every timestep 
        */
        // Hoist fluid variables
        float t_rho = 0.0;
        float t_rhov_x = 0.0, t_rhov_y = 0.0, t_rhov_z = 0.0;
        float t_Bx = 0.0, t_By = 0.0, t_Bz = 0.0;
        float t_p = 0.0, t_e = 0.0; 
        float t_KE = 0.0, t_B_sq = 0.0, t_B_dot_u = 0.0;

        // Hoist Fluid Fluxes 
        float t_xflux_rho = 0.0, t_yflux_rho = 0.0, t_zflux_rho = 0.0;
        float t_xflux_rhovx = 0.0, t_yflux_rhovx = 0.0, t_zflux_rhovx = 0.0;
        float t_xflux_rhovy = 0.0, t_yflux_rhovy = 0.0, t_zflux_rhovy = 0.0;
        float t_xflux_rhovz = 0.0, t_yflux_rhovz = 0.0, t_zflux_rhovz = 0.0;
        float t_xflux_Bx = 0.0, t_yflux_Bx = 0.0, t_zflux_Bx = 0.0;
        float t_xflux_By = 0.0, t_yflux_By = 0.0, t_zflux_By = 0.0;
        float t_xflux_Bz = 0.0, t_yflux_Bz = 0.0, t_zflux_Bz = 0.0;
        float t_xflux_e = 0.0, t_yflux_e = 0.0, t_zflux_e = 0.0;
        
        // Hoist Intermediate Variables
        float t_int_rho = 0.0;
        float t_int_rhov_x = 0.0, t_int_rhov_y = 0.0, t_int_rhov_z = 0.0;
        float t_int_Bx = 0.0, t_int_By = 0.0, t_int_Bz = 0.0;
        float t_int_p = 0.0, t_int_e = 0.0; 
        float t_int_KE = 0.0, t_int_B_sq = 0.0;

        //Hoist Intermediate Fluxes
        float t_int_xflux_rho = 0.0, t_int_yflux_rho = 0.0, t_int_zflux_rho = 0.0;
        float t_int_xflux_rhovx = 0.0, t_int_yflux_rhovx = 0.0, t_int_zflux_rhovx = 0.0;
        float t_int_xflux_rhovy = 0.0, t_int_yflux_rhovy = 0.0, t_int_zflux_rhovy = 0.0;
        float t_int_xflux_rhovz = 0.0, t_int_yflux_rhovz = 0.0, t_int_zflux_rhovz = 0.0;
        float t_int_xflux_Bx = 0.0, t_int_yflux_Bx = 0.0, t_int_zflux_Bx = 0.0;
        float t_int_xflux_By = 0.0, t_int_yflux_By = 0.0, t_int_zflux_By = 0.0;
        float t_int_xflux_Bz = 0.0, t_int_yflux_Bz = 0.0, t_int_zflux_Bz = 0.0;
        float t_int_xflux_e = 0.0, t_int_yflux_e = 0.0, t_int_zflux_e = 0.0;

        // Handle B.Cs separately
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){ // THIS LOOP ORDER IS FOR CONTIGUOUS MEMORY ACCESS
            for (int i = tidx + 1; i < Nx - 1; i += xthreads){ 
                for (int j = tidy + 1; j < Ny - 1; j += ythreads){
                    /* Compute p, B^2, \vec{B}\dot\vec{u}, and the hoisted fluid variables */
                    t_rho = rho[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_rhov_x = rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_rhov_y = rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_rhov_z = rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_KE = KE(i, j, k, rho, rhov_x, rhov_y, rhov_z, Nx, Ny, Nz); // I know this is overly verbose, current concern is get it working and correct 
                    
                    t_Bx = Bx[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_By = By[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_Bz = Bz[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_B_sq = B_sq(i, j, k, Bx, By, Bz, Nx, Ny, Nz);
                    
                    t_e = e[IDX3D(i, j, k, Nx, Ny, Nz)];
                    t_p = p(i, j, k, e, t_B_sq, t_KE, Nx, Ny, Nz);
                    t_B_dot_u = B_dot_u(i, j, k, rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, Nx, Ny, Nz);
                    
                    /* Compute fluid fluxes */
                    t_xflux_rho = XFluxRho(i, j, k, rhov_x, Nx, Ny, Nz);
                    t_yflux_rho = YFluxRho(i, j, k, rhov_y, Nx, Ny, Nz);
                    t_zflux_rho = ZFluxRho(i, j, k, rhov_z, Nx, Ny, Nz);

                    t_xflux_rhovx = XFluxRhoVX(t_rho, t_rhov_x, t_Bx, t_B_sq, t_p);
                    t_yflux_rhovx = YFluxRhoVX(t_rho, t_rhov_x, t_rhov_y, t_Bx, t_By);
                    t_zflux_rhovx = ZFluxRhoVX(t_rho, t_rhov_x, t_rhov_z, t_Bx, t_Bz);

                    t_xflux_rhovy = XFluxRhoVY(t_rho, t_rhov_x, t_rhov_y, t_Bx, t_By);
                    t_yflux_rhovy = YFluxRhoVY(t_rho, t_rhov_y, t_By, t_B_sq, t_p);
                    t_zflux_rhovy = ZFluxRhoVY(t_rho, t_rhov_y, t_rhov_z, t_By, t_Bz);

                    t_xflux_rhovz = XFluxRhoVZ(t_rho, t_rhov_x, t_rhov_z, t_Bx, t_Bz);
                    t_yflux_rhovz = YFluxRhoVZ(t_rho, t_rhov_y, t_rhov_z, t_By, t_Bz);
                    t_zflux_rhovz = ZFluxRhoVZ(t_rho, t_rhov_z, t_Bz, t_B_sq, t_p);

                    t_xflux_Bx = XFluxBX();
                    t_yflux_Bx = YFluxBX(t_rho, t_rhov_x, t_rhov_y, t_Bx, t_By); 
                    t_zflux_Bx = ZFluxBX(t_rho, t_rhov_x, t_rhov_z, t_Bx, t_Bz); 

                    t_xflux_By = XFluxBY(t_rho, t_rhov_x, t_rhov_y, t_Bx, t_By); 
                    t_yflux_By = YFluxBY(); 
                    t_zflux_By = ZFluxBY(t_rho, t_rhov_y, t_rhov_z, t_By, t_Bz); 

                    t_xflux_Bz = XFluxBZ(t_rho, t_rhov_x, t_rhov_z, t_Bx, t_Bz); 
                    t_yflux_Bz = YFluxBZ(t_rho, t_rhov_y, t_rhov_z, t_By, t_Bz); 
                    t_zflux_Bz = ZFluxBZ(); 

                    t_xflux_e = XFluxE(t_rho, t_rhov_x, t_Bx, t_e, t_p, t_B_sq, t_B_dot_u); 
                    t_yflux_e = YFluxE(t_rho, t_rhov_y, t_By, t_e, t_p, t_B_sq, t_B_dot_u); 
                    t_zflux_e = ZFluxE(t_rho, t_rhov_z, t_Bz, t_e, t_p, t_B_sq, t_B_dot_u); 

                    /* 
                    Compute intermediate variables 
                    \bar{Q}_{ijk} = Q^{n}_{ijk} - (dt / dx) * (F^{n}_{ijk} - F^{n}_{i-1,j,k}) 
                                        - (dt / dy) * (G^{n}_{ijk} - G^{n}_{i,j-1,k})
                                        - (dt / dz) * (H^{n}_{ijk} - H^{n}_{i,j,k-1})
                    */
                    t_int_rho = t_rho 
                        - (dt / dx) * (t_xflux_rho - XFluxRho(i-1, j, k, rhov_x, Nx, Ny, Nz))
                        - (dt / dy) * (t_yflux_rho - YFluxRho(i, j-1, k, rhov_y, Nx, Ny, Nz))   
                        - (dt / dz) * (t_zflux_rho - ZFluxRho(i, j, k-1, rhov_z, Nx, Ny, Nz));
                    
                    t_int_rhov_x = t_rhov_x
                        - (dt / dx) * (t_xflux_rhovx - XFluxRhoVX()) /* IMPLEMENT OVERLOADED FUNCTIONS FOR THESE */
                    // I fully understand how awful this looks, and it's going to happen for 7 more variables 
                    // t_int_rho = t_rho 
                    //     - (dt / dx) * (t_xflux_rho - 
                    //     XFluxRho(
                    //         rho[IDX3D(i-1, j, k, Nx, Ny, Nz)], rhov_x[IDX3D(i-1, j, k, Nx, Ny, Nz)], 
                    //         Bx[IDX3D(i-1, j, k, Nx, Ny, Nz)], B_sq(i-1, j, k, Bx, By, Bz, Nx, Ny, Nz), 
                    //         p(i-1, j, k, e, 
                    //             B_sq(i-1, j, k, Bx, By, Bz, Nx, Ny, Nz), 
                    //             KE(i, j, k, rho, rhov_x, rhov_y, rhov_z, Nx, Ny, Nz),
                    //             Nx, Ny, Nz
                    //             )
                    //     )
                    //     )
                    //     - (dt / dy) * (t_yflux_rho -
                    //     YFluxRho(
                    //         rho[IDX3D(i, j-1, k, Nx, Ny, Nz)], rhov_y[IDX3D(i-1, j, k, Nx, Ny, Nz)], 
                    //         Bx[IDX3D(i-1, j, k, Nx, Ny, Nz)], B_sq(i-1, j, k, Bx, By, Bz, Nx, Ny, Nz), 
                    //         p(i-1, j, k, e, 
                    //             B_sq(i-1, j, k, Bx, By, Bz, Nx, Ny, Nz), 
                    //             KE(i, j, k, rho, rhov_x, rhov_y, rhov_z, Nx, Ny, Nz),
                    //             Nx, Ny, Nz
                    //         )
                    //     )
                    //     )   
                    //     - (dt / dz) * (t_zflux_rho -
                    //     ZFluxRho(

                    //         )
                    //     );
                    /* TODO - Compute intermediate fluxes */

                    /* TODO - Update fluid variables */
                }
            }
        } 
        return;
    }

/* 
Boundary Conditions are:
(1) Rigid, Perfectly-Conducting wall
(2) Periodic in z
*/
__global__ void BoundaryConditions(float* rho_np1, float* rhovx_np1, float* rhovy_np1, float* rhovz_np1, float* Bx_np1, float* By_np1, float* Bz_np1, float* e_np1,
     const float* rho, const float* rhov_x, const float *rhov_y, const float* rhov_z, const float* Bx, const float* By, const float* Bz, const float* e, 
     const int Nx, const int Ny, const int Nz)
     {
        // Execution configuration boilerplate
        int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;
        int xthreads = gridDim.x * blockDim.x;
        int ythreads = gridDim.y * blockDim.y;
        int zthreads = gridDim.z * blockDim.z;

        /* B.Cs on (i, j, k = 0) */

        /* B.Cs on (i, j, k = N-1) */

        /* B.Cs on (i = 0, j, k) */
        
        /* B.Cs on (i = N-1, j, k) */

        /* B.Cs on (i, j = 0, k) */

        /* B.Cs on (i, j = N-1, k) */

        return;
     }

/* Flux declarations */
// Rho
/* THESE CAN BE REFACTORED TO MINIMIZE VERBOSITY */
__device__ float XFluxRho(const int i, const int j, const int k, const float* rhov_x, const int Nx, const int Ny, const int Nz)
    {
        return rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)];
    }
__device__ float YFluxRho(const int i, const int j, const int k, const float* rhov_y, const int Nx, const int Ny, const int Nz)
    {
        return rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)];
    }
__device__ float ZFluxRho(const int i, const int j, const int k, const float* rhov_z, const int Nx, const int Ny, const int Nz)
    {
        return rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)];
    }

// RhoVX
__device__ float XFluxRhoVX(const float rho, const float rhov_x, const float Bx, const float B_sq, const float p)
    {
        return (1.0 / rho) * pow(rhov_x, 2) - pow(Bx, 2) + p + B_sq / 2.0;
    }

__device__ float YFluxRhoVX(const float rho, const float rhov_x, const float rhov_y, const float Bx, const float By)
    {
        return (1.0 / rho) * rhov_x * rhov_y - Bx * By;
    }

__device__ float ZFluxRhoVX(const float rho, const float rhov_x, const float rhov_z, const float Bx, const float Bz)
    {
        return (1.0 / rho) * rhov_x * rhov_z - Bx * Bz;
    }

// RhoVY
__device__ float XFluxRhoVY(const float rho, const float rhov_x, const float rhov_y, const float Bx, const float By)
    {
        return (1.0 / rho) * rhov_x * rhov_y - Bx * By;
    }

__device__ float YFluxRhoVY(const float rho, const float rhov_y, const float By, const float B_sq, const float p)
    {
        return (1.0 / rho) * pow(rhov_y, 2) - pow(By, 2) + p + B_sq / 2.0;
    }

__device__ float ZFluxRhoVY(const float rho, const float rhov_y, const float rhov_z, const float By, const float Bz)
    {
        return (1.0 / rho) * rhov_y * rhov_z - By * Bz;
    }

// RhoVZ
__device__ float XFluxRhoVZ(const float rho, const float rhov_x, const float rhov_z, const float Bx, const float Bz)
    {
        return (1.0 / rho) * rhov_x * rhov_z - Bx * Bz;
    }

__device__ float YFluxRhoVZ(const float rho, const float rhov_y, const float rhov_z, const float By, const float Bz)
    {
        return (1.0 / rho) * rhov_y * rhov_z - By * Bz;
    }

__device__ float ZFluxRhoVZ(const float rho, const float rhov_z, const float Bz, const float B_sq, const float p)
    {
        return (1.0 / rho) * pow(rhov_z, 2) - pow(Bz, 2) + p + B_sq / 2.0;
    }

// Bx
__device__ float XFluxBX()
    {
        return 0.0;
    }

__device__ float YFluxBX(const float rho, const float rhov_x, const float rhov_y, const float Bx, const float By)
    {
        return (1.0 / rho) * (rhov_x * By - Bx * rhov_y);
    }

__device__ float ZFluxBX(const float rho, const float rhov_x, const float rhov_z, const float Bx, const float Bz)
    {
        return (1.0 / rho) * (rhov_x * Bz - Bx * rhov_z);
    }

// By
__device__ float XFluxBY(const float rho, const float rhov_x, const float rhov_y, const float Bx, const float By)
    {
        return (1.0 / rho) * (rhov_y * Bx - By * rhov_x); 
    }

__device__ float YFluxBY()
    {
        return 0.0;
    }

__device__ float ZFluxBY(const float rho, const float rhov_y, const float rhov_z, const float By, const float Bz)
    {
        return (1.0 / rho) * (rhov_y * Bz - By * rhov_z);
    }

// Bz
__device__ float XFluxBZ(const float rho, const float rhov_x, const float rhov_z, const float Bx, const float Bz)
    {
        return (1.0 / rho) * (rhov_z * Bx - Bz * rhov_x);
    }

__device__ float YFluxBZ(const float rho, const float rhov_y, const float rhov_z, const float By, const float Bz)
    {
        return (1.0 / rho) * (rhov_z * By - Bz * rhov_y);
    }

__device__ float ZFluxBZ()
    {
        return 0.0;
    }

// e
__device__ float XFluxE(const float rho, const float rhov_x, const float Bx, const float e, const float p, const float B_sq, const float B_dot_u)
    {
        return e + p + B_sq * (rhov_x / rho) - B_dot_u * Bx; 
    }

__device__ float YFluxE(const float rho, const float rhov_y, const float By, const float e, const float p, const float B_sq, const float B_dot_u)
    {
        return e + p + B_sq * (rhov_y / rho) - B_dot_u * By; 
    }

__device__ float ZFluxE(const float rho, const float rhov_z, const float Bz, const float e, const float p, const float B_sq, const float B_dot_u)
    {
        return e + p + B_sq * (rhov_z / rho) - B_dot_u * Bz; 
    }

/* 
Intermediate Flux declarations 
(Aren't these just flux functions w / intermediate variables?)
*/


/* B-squared, etc. */
__device__ float B_sq(const int i, const int j, const int k, const float* Bx, const float* By, const float* Bz, 
    const int Nx, const int Ny, const int Nz)
    {
        return pow(Bx[IDX3D(i, j, k, Nx, Ny, Nz)], 2) + pow(By[IDX3D(i, j, k, Nx, Ny, Nz)], 2) + pow(Bz[IDX3D(i, j, k, Nx, Ny, Nz)], 2);
    }

__device__ float p(const int i, const int j, const int k, const float* e, const float B_sq, const float KE, 
    const int Nx, const int Ny, const int Nz)
    {
        return (gamma - 1.0) * (e[IDX3D(i, j, k, Nx, Ny, Nz)] - KE - B_sq / 2.0);
    }

__device__ float KE(const int i, const int j, const int k, const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
    const int Nx, const int Ny, const int Nz)
    {
        float KE = (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * (
            pow(rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)], 2) + pow(rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)], 2) + pow(rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)], 2));
        return KE;
    }

__device__ float B_dot_u(const int i, const int j, const int k, const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z,
    const float* Bx, const float* By, const float* Bz, const int Nx, const int Ny, const int Nz)
    {
        float B_dot_u = 0.0;
        B_dot_u = (1.0 / rho[IDX3D(i, j, k, Nx, Ny, Nz)]) * (rhov_x[IDX3D(i, j, k, Nx, Ny, Nz)] * Bx[IDX3D(i, j, k, Nx, Ny, Nz)]
            + rhov_y[IDX3D(i, j, k, Nx, Ny, Nz)] * By[IDX3D(i, j, k, Nx, Ny, Nz)] + rhov_z[IDX3D(i, j, k, Nx, Ny, Nz)] * Bz[IDX3D(i, j, k, Nx, Ny, Nz)]);
        return B_dot_u;
    }
#endif