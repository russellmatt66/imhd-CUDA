#include "kernels_od_fluxes.cuh"
#include "kernels_intvarbcs.cuh"
#include "kernels_od_intvar.cuh"

__global__ void ComputeIntermediateVariablesBoundary(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz, const float D,
    const int Nx, const int Ny, const int Nz)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x; 
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        int k = threadIdx.z + blockDim.z * blockIdx.z;

        int xthreads = blockDim.x * gridDim.x;
        int ythreads = blockDim.y * gridDim.y;
        int zthreads = blockDim.z * gridDim.z;
    
        int cube_size = Nx * Ny * Nz;

        // Front Face
        // k = 0
        // i \in [1,Nx-1]
        // j \in [1,Ny-1]
        if (i > 0 && i < Nx && j > 0 && j < Ny)
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz)] = intRho(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + cube_size] = intRhoVX(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 4 * cube_size] = intBX(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 5 * cube_size] = intBY(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 6 * cube_size] = intBZ(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 7 * cube_size] = intE(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
        }

        // Right Face
        // j = Ny - 1
        // i \in [1,Nx-2] 
        // k \in [1,Nz-2]
        if (i > 0 && i < Nx-1 && k > 0 && k < Nz-1){
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz)] = intRhoRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + cube_size] = intRhoVXRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 4 * cube_size] = intBXRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 5 * cube_size] = intBYRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 6 * cube_size] = intBZRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 7 * cube_size] = intERight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // Top Face
        // i = 0
        // j \in [1,Ny-1]
        // k \in [1,Nz-2]
        if (j > 0 && j < Ny && k > 0 && k < Nz-1){
            intvar[IDX3D(0, j, k, Nx, Ny, Nz)] = intRho(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + cube_size] = intRhoVX(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 4 * cube_size] = intBX(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 5 * cube_size] = intBY(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 6 * cube_size] = intBZ(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 7 * cube_size] = intE(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // Left Face
        // j = 0
        // i \in [1, Nx-2]
        // k \in [1, Nz-2]
        if (i > 0 && i < Nx-1 && k > 0 && k < Nz-1){
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz)] = intRho(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + cube_size] = intRhoVX(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 4 * cube_size] = intBX(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 5 * cube_size] = intBY(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 6 * cube_size] = intBZ(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 7 * cube_size] = intE(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // Bottom Face
        // i = Nx - 1
        // j \in [1,Ny-1]
        // k \in [1,Nz-2]
        if (j > 0 && j < Ny && k > 0 && k < Nz-1){
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz)] = intRhoBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + cube_size] = intRhoVXBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 4 * cube_size] = intBXBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 5 * cube_size] = intBYBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 6 * cube_size] = intBZBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 7 * cube_size] = intEBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // Back Face
        // k = Nz-1
        // i \in [1,Nx-1]
        // j \in [1,Ny-1]
        if (i > 0 && i < Nx && j > 0 && j < Ny){
            intvar[IDX3D(i, j, Nz-1, Nx, Ny, Nz)] = intRhoBack(i, j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + cube_size] = intRhoVXBack(i, j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYBack(i, j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZBack(i, j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 4 * cube_size] = intBXBack(i, j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 5 * cube_size] = intBYBack(i, j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 6 * cube_size] = intBZBack(i, j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 7 * cube_size] = intEBack(i, j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
        }

        /* BELOW NEEDS WORK */
        // After the above is done, there are still SIX lines where data has not been specified
        // {(0,j,0), (0,j,Nz-1), (0,0,k), (Nx-1,0,k), (i,0,0), (i,0,Nz-1)}
        // (0, j, 0) - "FrontTop"
        for (int j = tidy + 1; j < Ny; j += ythreads){
                intvar[IDX3D(0, j, 0, Nx, Ny, Nz)] = intRhoFrontTop(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
                intvar[IDX3D(0, j, 0, Nx, Ny, Nz) + cube_size] = intRhoVXFrontTop(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
                intvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYFrontTop(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
                intvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZFrontTop(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
                intvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 4 * cube_size] = intBXFrontTop(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
                intvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 5 * cube_size] = intBYFrontTop(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
                intvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 6 * cube_size] = intBZFrontTop(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
                intvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 7 * cube_size] = intEFrontTop(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // (i, 0, 0) - "FrontLeft"
        for (int i = tidx + 1; i < Nx; i += xthreads){
            intvar[IDX3D(i, 0, 0, Nx, Ny, Nz)] = intRhoFrontLeft(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + cube_size] = intRhoVXFrontLeft(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYFrontLeft(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZFrontLeft(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 4 * cube_size] = intBXFrontLeft(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 5 * cube_size] = intBYFrontLeft(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 6 * cube_size] = intBZFrontLeft(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 7 * cube_size] = intEFrontLeft(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // (0, 0, k) - "TopLeft"
        for (int k = tidz + 1; k < Nz ; k += zthreads){
            intvar[IDX3D(0, 0, k, Nx, Ny, Nz)] = intRhoTopLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(0, 0, k, Nx, Ny, Nz) + cube_size] = intRhoVXTopLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYTopLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZTopLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 4 * cube_size] = intBXTopLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 5 * cube_size] = intBYTopLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 6 * cube_size] = intBZTopLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 7 * cube_size] = intETopLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // (0, j, Nz-1) - "TopBack"
        for (int j = tidy + 1; j < Ny; j += ythreads){
                intvar[IDX3D(0, j, Nz-1, Nx, Ny, Nz)] = intRho(0, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
                intvar[IDX3D(0, j, Nz-1, Nx, Ny, Nz) + cube_size] = intRhoVX(0, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
                intvar[IDX3D(0, j, Nz-1, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(0, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
                intvar[IDX3D(0, j, Nz-1, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(0, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
                intvar[IDX3D(0, j, Nz-1, Nx, Ny, Nz) + 4 * cube_size] = intBX(0, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
                intvar[IDX3D(0, j, Nz-1, Nx, Ny, Nz) + 5 * cube_size] = intBY(0, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
                intvar[IDX3D(0, j, Nz-1, Nx, Ny, Nz) + 6 * cube_size] = intBZ(0, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
                intvar[IDX3D(0, j, Nz-1, Nx, Ny, Nz) + 7 * cube_size] = intE(0, j, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }
        
        // (i, 0, Nz-1) - "BackLeft"
        for (int i = tidx + 1; i < Nx; i += xthreads){
            intvar[IDX3D(i, 0, Nz-1, Nx, Ny, Nz)] = intRho(i, 0, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(i, 0, Nz-1, Nx, Ny, Nz) + cube_size] = intRhoVX(i, 0, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(i, 0, Nz-1, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(i, 0, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(i, 0, Nz-1, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(i, 0, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(i, 0, Nz-1, Nx, Ny, Nz) + 4 * cube_size] = intBX(i, 0, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(i, 0, Nz-1, Nx, Ny, Nz) + 5 * cube_size] = intBY(i, 0, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(i, 0, Nz-1, Nx, Ny, Nz) + 6 * cube_size] = intBZ(i, 0, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(i, 0, Nz-1, Nx, Ny, Nz) + 7 * cube_size] = intE(i, 0, Nz-1, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // (Nx-1, 0, k) - "BottomLeft"
        for (int k = tidz + 1; k < Nz - 1; k += zthreads){
            intvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz)] = intRhoBottomLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + cube_size] = intRhoVXBottomLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYBottomLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZBottomLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 4 * cube_size] = intBXBottomLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 5 * cube_size] = intBYBottomLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 6 * cube_size] = intBZBottomLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 7 * cube_size] = intEBottomLeft(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        /* 
        [1/3/25]
        I DON'T BELIEVE THIS NEEDS TO BE CALCULATED
        So why do something you don't believe in?
        Because, I believe that it is good practice to do as complete a job as possible
        More importantly, I don't believe that this will cause a serious performance hit 
        But, I could be wrong about that 
        */
        // Pick up straggler point: {(0, 0, 0)}
        if (tidx == 0 && tidy == 0 && tidz == 0){
            intvar[0] = fluidvar[0] 
                - (dt / dx) * (XFluxRho(0, 0, 0, fluidvar, Nx, Ny, Nz))
                - (dt / dy) * (YFluxRho(0, 0, 0, fluidvar, Nx, Ny, Nz))
                - (dt / dz) * (ZFluxRho(0, 0, 0, fluidvar, Nx, Ny, Nz) - ZFluxRho(0, 0, Nz-2, fluidvar, Nx, Ny, Nz));
            
            intvar[Nx * Ny * Nz] = fluidvar[Nx * Ny * Nz]
                - (dt / dx) * (XFluxRhoVX(0, 0, 0, fluidvar, Nx, Ny, Nz))
                - (dt / dy) * (YFluxRhoVX(0, 0, 0, fluidvar, Nx, Ny, Nz))
                - (dt / dz) * (ZFluxRhoVX(0, 0, 0, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(0, 0, Nz-2, fluidvar, Nx, Ny, Nz));  

            intvar[2 * Nx * Ny * Nz] = fluidvar[2 * Nx * Ny * Nz]
                - (dt / dx) * (XFluxRhoVY(0, 0, 0, fluidvar, Nx, Ny, Nz))
                - (dt / dy) * (YFluxRhoVY(0, 0, 0, fluidvar, Nx, Ny, Nz))
                - (dt / dz) * (ZFluxRhoVY(0, 0, 0, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(0, 0, Nz-2, fluidvar, Nx, Ny, Nz)); 
    
            intvar[3 * Nx * Ny * Nz] = fluidvar[3 * Nx * Ny * Nz]
                - (dt / dx) * (XFluxRhoVY(0, 0, 0, fluidvar, Nx, Ny, Nz))
                - (dt / dy) * (YFluxRhoVY(0, 0, 0, fluidvar, Nx, Ny, Nz))
                - (dt / dz) * (ZFluxRhoVY(0, 0, 0, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(0, 0, Nz-2, fluidvar, Nx, Ny, Nz)); 

                
            intvar[4 * Nx * Ny * Nz] = fluidvar[4 * Nx * Ny * Nz]
                - (dt / dx) * (XFluxBX(0, 0, 0, fluidvar, Nx, Ny, Nz))
                - (dt / dy) * (YFluxBX(0, 0, 0, fluidvar, Nx, Ny, Nz))
                - (dt / dz) * (ZFluxBX(0, 0, 0, fluidvar, Nx, Ny, Nz) - ZFluxBX(0, 0, Nz-2, fluidvar, Nx, Ny, Nz)); 

                
            intvar[5 * Nx * Ny * Nz] = fluidvar[5 * Nx * Ny * Nz]
                - (dt / dx) * (XFluxBY(0, 0, 0, fluidvar, Nx, Ny, Nz))
                - (dt / dy) * (YFluxBY(0, 0, 0, fluidvar, Nx, Ny, Nz))
                - (dt / dz) * (ZFluxBY(0, 0, 0, fluidvar, Nx, Ny, Nz) - ZFluxBY(0, 0, Nz-2, fluidvar, Nx, Ny, Nz)); 

                
            intvar[6 * Nx * Ny * Nz] = fluidvar[6 * Nx * Ny * Nz]
                - (dt / dx) * (XFluxBZ(0, 0, 0, fluidvar, Nx, Ny, Nz))
                - (dt / dy) * (YFluxBZ(0, 0, 0, fluidvar, Nx, Ny, Nz))
                - (dt / dz) * (ZFluxBZ(0, 0, 0, fluidvar, Nx, Ny, Nz) - ZFluxBZ(0, 0, Nz-2, fluidvar, Nx, Ny, Nz)); 

                
            intvar[7 * Nx * Ny * Nz] = fluidvar[7 * Nx * Ny * Nz]
                - (dt / dx) * (XFluxE(0, 0, 0, fluidvar, Nx, Ny, Nz))
                - (dt / dy) * (YFluxE(0, 0, 0, fluidvar, Nx, Ny, Nz))
                - (dt / dz) * (ZFluxE(0, 0, 0, fluidvar, Nx, Ny, Nz) - ZFluxE(0, 0, Nz-2, fluidvar, Nx, Ny, Nz));
        }

        /*
        [1/3/25] 
        Does this slow things down - YES 
        Does it matter - PROBABLY NOT: O(max(Nx*Ny, Nx*Nz, Ny*Nz)) < O(Nx*Ny*Nz)), [BCs] vs. [Interior] 
        Is it necessary to avoid a race condition - YES
        Does \bar{Q}_{0, 0, 0} need to be calculated - Honestly, no
        */
        __syncthreads(); 

        // PBCs
        for (int i = tidx ; i < Nx; i += xthreads){ // Do not ignore the edges b/c intermediate variables got calculated there
            for (int j = tidy ; j < Ny ; j += ythreads){
                for (int ivf = 0; ivf < 8; ivf++){ // PBCs - they the SAME point 
                    intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + ivf * cube_size] += intvar[IDX3D(i, j, Nz - 1, Nx, Ny, Nz) + ivf * cube_size];
                    intvar[IDX3D(i, j, Nz - 1, Nx, Ny, Nz) + ivf * cube_size] = intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + ivf * cube_size];
                }
            }
        }
        return;
    }

// Kernels that deal with the interior of the k = 0 plane: (i, j, 0)
/* 
THESE ARE WRONG 
THEY IMPLEMENT CORRECTOR STEP INSTEAD
*/
__device__ float intRhoRight(const int i, const int k, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz)]
            - (dt / dx) * (XFluxRho(i+1, Ny-1, k, fluidvar, Nx, Ny, Nz) - XFluxRho(i, Ny-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (-YFluxRho(i, Ny-1, k, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRho(i, Ny-1, k+1, fluidvar, Nx, Ny, Nz) - ZFluxRho(i, Ny-1, k, fluidvar, Nx, Ny, Nz));        
    }

__device__ float intRhoVXRight(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + Nx*Ny*Nz]
            - (dt / dx) * (XFluxRhoVX(i+1, Ny-1, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVX(i, Ny-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (-YFluxRhoVX(i, Ny-1, k, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRhoVX(i, Ny-1, k+1, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(i, Ny-1, k, fluidvar, Nx, Ny, Nz)); 
    }

__device__ float intRhoVYRight(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 2*Nx*Ny*Nz]
            - (dt / dx) * (XFluxRhoVY(i+1, Ny-1, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVY(i, Ny-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (-YFluxRhoVY(i, Ny-1, k, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRhoVY(i, Ny-1, k+1, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(i, Ny-1, k, fluidvar, Nx, Ny, Nz)); 
    }

__device__ float intRhoVZRight(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 3*Nx*Ny*Nz]
            - (dt / dx) * (XFluxRhoVZ(i+1, Ny-1, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVZ(i, Ny-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (-YFluxRhoVZ(i, Ny-1, k, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRhoVZ(i, Ny-1, k+1, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, Ny-1, k, fluidvar, Nx, Ny, Nz)); 
    }

__device__ float intBXRight(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 4*Nx*Ny*Nz]
            - (dt / dx) * (XFluxBX(i+1, Ny-1, k, fluidvar, Nx, Ny, Nz) - XFluxBX(i, Ny-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (-YFluxBX(i, Ny-1, k, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxBX(i, Ny-1, k+1, fluidvar, Nx, Ny, Nz) - ZFluxBX(i, Ny-1, k, fluidvar, Nx, Ny, Nz)); 
    }

__device__ float intBYRight(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 5*Nx*Ny*Nz]
            - (dt / dx) * (XFluxBY(i+1, Ny-1, k, fluidvar, Nx, Ny, Nz) - XFluxBY(i, Ny-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (-YFluxBY(i, Ny-1, k, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxBY(i, Ny-1, k+1, fluidvar, Nx, Ny, Nz) - ZFluxBY(i, Ny-1, k, fluidvar, Nx, Ny, Nz)); 
    }

__device__ float intBZRight(const int i, const int k, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 6*Nx*Ny*Nz]
            - (dt / dx) * (XFluxBZ(i+1, Ny-1, k, fluidvar, Nx, Ny, Nz) - XFluxBZ(i, Ny-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (-YFluxBZ(i, Ny-1, k, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxBZ(i, Ny-1, k+1, fluidvar, Nx, Ny, Nz) - ZFluxBZ(i, Ny-1, k, fluidvar, Nx, Ny, Nz)); 
    }

__device__ float intERight(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 7*Nx*Ny*Nz]
            - (dt / dx) * (XFluxE(i+1, Ny-1, k, fluidvar, Nx, Ny, Nz) - XFluxE(i, Ny-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (-YFluxE(i, Ny-1, k, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxE(i, Ny-1, k+1, fluidvar, Nx, Ny, Nz) - ZFluxE(i, Ny-1, k, fluidvar, Nx, Ny, Nz)); 
    }

/* NEED TO FIX */
// Kernels that deal with Bottom Face
// i = Nx-1
// j \in [1,Ny-1]
// k \in [1,Nz-2] - leave the front / back faces alone
__device__ float intRhoTop(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz)]
            - (dt / dx) * (XFluxRho(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRho(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxRho(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxRho(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRho(0, j, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVXTop(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVX(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVX(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVX(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxRhoVX(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(0, j, k-1, fluidvar, Nx, Ny, Nz));
    }
    
__device__ float intRhoVYTop(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVY(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVY(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVY(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxRhoVY(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(0, j, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVZTop(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 3 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVZ(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVZ(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVZ(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxRhoVZ(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(0, j, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBXTop(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBX(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBX(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxBX(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxBX(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxBX(0, j, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBYTop(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBY(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBY(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxBY(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxBY(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxBY(0, j, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBZTop(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBZ(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBZ(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxBZ(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxBZ(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxBZ(0, j, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intETop(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, k, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxE(0, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxE(0, j, k, fluidvar, Nx, Ny, Nz) - YFluxE(0, j-1, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxE(0, j, k, fluidvar, Nx, Ny, Nz) - ZFluxE(0, j, k-1, fluidvar, Nx, Ny, Nz));
    }

/* NEED TO FIX */
// Kernels that deal with the Back Face 
// k = Nz-1
// i \in [1, Nx-2]
// j \in [1, Nz-2]
__device__ float intRhoLeft(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz)]
            - (dt / dx) * (XFluxRho(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRho(i-1, 0, k, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxRho(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRho(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRho(i, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVXLeft(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVX(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVX(i-1, 0, k, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxRhoVX(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVX(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(i, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVYLeft(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVY(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVY(i-1, 0, k, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxRhoVY(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVY(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(i, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVZLeft(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVZ(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVZ(i-1, 0, k, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxRhoVZ(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVZ(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBXLeft(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBX(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxBX(i-1, 0, k, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxBX(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBX(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBX(i, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBYLeft(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBY(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxBY(i-1, 0, k, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxBY(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBY(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBY(i, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBZLeft(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBZ(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxBZ(i-1, 0, k, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxBZ(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBZ(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBZ(i, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intELeft(const int i, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxE(i, 0, k, fluidvar, Nx, Ny, Nz) - XFluxE(i-1, 0, k, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxE(i, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxE(i, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxE(i, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

/* NEED TO FIX */
// Kernels that deal with the (0, j, 0) line
__device__ float intRhoFrontTop(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz)]
            - (dt / dx) * (XFluxRho(0, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRho(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxRho(0, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRho(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxRho(0, j, Nz-2, fluidvar, Nx, Ny, Nz));        
    }

__device__ float intRhoVXFrontTop(const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVX(0, j, 0, fluidvar, Nx, Ny, Nz)) // i = -1 flux is zero => rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVX(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxRhoVX(0, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRhoVX(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(0, j, Nx-2, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVYFrontTop(const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVY(0, j, 0, fluidvar, Nx, Ny, Nz)) // i = -1 flux is zero => rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVY(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxRhoVY(0, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRhoVY(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(0, j, Nz-2, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVZFrontTop(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 3 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVZ(0, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVZ(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxRhoVZ(0, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRhoVZ(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(0, j, Nz-2, fluidvar, Nx, Ny, Nz));  
    }

__device__ float intBXFrontTop(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBX(0, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBX(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxBX(0, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxBX(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxBX(0, j, Nz-2, fluidvar, Nx, Ny, Nz));  
    }

__device__ float intBYFrontTop(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBY(0, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBY(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxBY(0, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxBY(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxBY(0, j, Nz-2, fluidvar, Nx, Ny, Nz));  
    }

__device__ float intBZFrontTop(const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBZ(0, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBZ(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxBZ(0, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxBZ(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxBZ(0, j, Nz-2, fluidvar, Nx, Ny, Nz));  
    }

__device__ float intEFrontTop(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, j, 0, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxE(0, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxE(0, j, 0, fluidvar, Nx, Ny, Nz) - YFluxE(0, j-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxE(0, j, 0, fluidvar, Nx, Ny, Nz) - ZFluxE(0, j, Nz-2, fluidvar, Nx, Ny, Nz));  
    }

/* NEED TO FIX */
// Kernels that deal with Front Left Line
// i \in [1, Nx-1]
// j = 0
// k = 0
__device__ float intRhoFrontLeft(const int i, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz)]
            - (dt / dx) * (XFluxRho(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxRho(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxRho(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRho(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall       
    }

__device__ float intRhoVXFrontLeft(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVX(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxRhoVX(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxRhoVX(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVX(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
    }

__device__ float intRhoVYFrontLeft(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVY(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxRhoVY(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxRhoVY(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVY(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
    }

__device__ float intRhoVZFrontLeft(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 3 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVZ(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxRhoVZ(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxRhoVZ(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVZ(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
    }

__device__ float intBXFrontLeft(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBX(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxBX(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxBX(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBX(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
    }

__device__ float intBYFrontLeft(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBY(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxBY(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxBY(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBY(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
    }

__device__ float intBZFrontLeft(const int i, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBZ(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxBZ(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxBZ(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBZ(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall  
    }

__device__ float intEFrontLeft(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, 0, 0, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxE(i, 0, 0, fluidvar, Nx, Ny, Nz) - XFluxE(i-1, 0, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxE(i, 0, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxE(i, 0, 0, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
    }

/* NEED TO FIX */
// Kernels that deal with the Top Left Line
// i = 0
// j = 0
// k \in [1, Nz-1]
__device__ float intRhoTopLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz)]
            - (dt / dx) * (XFluxRho(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRho(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRho(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRho(0, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVXTopLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVX(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVX(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVX(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(0, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVYTopLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVY(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVY(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVY(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(0, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVZTopLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVZ(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVZ(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVZ(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(0, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBXTopLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBX(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBX(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBX(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBX(0, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBYTopLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBY(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBY(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBY(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBY(0, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBZTopLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBZ(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBZ(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBZ(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBZ(0, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intETopLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(0, 0, k, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxE(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxE(0, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxE(0, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxE(0, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

/* NEED TO FIX */
// Kernels that deal with the Bottom Left Line
// i = Nx-1
// j = 0
// k \in [1, Nz-2]
__device__ float intRhoBottomLeft(const int k, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz)]
            - (dt / dx) * (XFluxRho(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRho(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxRho(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRho(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRho(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));        
    }

__device__ float intRhoVXBottomLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVX(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVX(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxRhoVX(Nx-1, 0, k, fluidvar, Nx, Ny, Nz))  // No flux on opposite site of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVX(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVYBottomLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVY(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVY(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxRhoVY(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVY(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVZBottomLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 3 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVZ(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxRhoVZ(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxRhoVZ(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVZ(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBXBottomLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBX(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxBX(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxBX(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBX(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBX(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBYBottomLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBY(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxBY(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxBY(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBY(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBY(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBZBottomLeft(const int k, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBZ(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxBZ(Nx-2, 0, k, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxBZ(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBZ(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxBZ(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));  
    }

__device__ float intEBottomLeft(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, 0, k, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxE(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - XFluxE(Nx-2, 0, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (YFluxE(Nx-1, 0, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite site of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxE(Nx-1, 0, k, fluidvar, Nx, Ny, Nz) - ZFluxE(Nx-1, 0, k-1, fluidvar, Nx, Ny, Nz));
    }
    
// I think that's everything! 