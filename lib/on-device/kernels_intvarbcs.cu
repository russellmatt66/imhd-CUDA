#include "kernels_od_fluxes.cuh"
#include "kernels_od_intvar.cuh"
#include "kernels_intvarbcs.cuh"
#include "diffusion.cuh"

/* THIS NEEDS TO BE DEFINED A SINGLE TIME IN A SINGLE PLACE */
#define IDX3D(i, j, k, Nx, Ny, Nz) ((k) * (Nx) * (Ny) + (i) * (Ny) + j)

// Megakernels that suffer from thread divergence
__global__ void ComputeIntermediateVariablesBoundaryNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz,
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
        // i \in [1,Nx-2]
        // j \in [1,Ny-2]
        if (i < Nx-1 && j < Ny-1){
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz)] = intRho(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + cube_size] = intRhoVX(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 4 * cube_size] = intBX(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 5 * cube_size] = intBY(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 6 * cube_size] = intBZ(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 7 * cube_size] = intE(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
        }

        // Left and Right Faces
        // j = {0, Ny - 1}
        // i \in [0,Nx-2] 
        // k \in [1,Nz-2]
        if (i < Nx-1 && k > 0 && k < Nz-1){

            // Left Face
            // j = 0
            // i \in [0,Nx-2] 
            // k \in [1,Nz-2]
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz)] = intRhoRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + cube_size] = intRhoVXRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 4 * cube_size] = intBXRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 5 * cube_size] = intBYRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 6 * cube_size] = intBZRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 7 * cube_size] = intERight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e

            // Right Face
            // j = Ny-1
            // i \in [0,Nx-2] 
            // k \in [1,Nz-2]
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz)] = intRho(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + cube_size] = intRhoVX(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 4 * cube_size] = intBX(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 5 * cube_size] = intBY(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 6 * cube_size] = intBZ(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 7 * cube_size] = intE(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // Top and Bottom Faces
        // i = {0, Nx-1}
        // j \in [0,Ny-2]
        // k \in [1,Nz-2]
        if (j < Ny-1 && k > 0 && k < Nz-1){
            
            // Top Face
            // i = 0
            // j \in [0,Ny-2]
            // k \in [1,Nz-2]
            intvar[IDX3D(0, j, k, Nx, Ny, Nz)] = intRho(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + cube_size] = intRhoVX(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 4 * cube_size] = intBX(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 5 * cube_size] = intBY(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 6 * cube_size] = intBZ(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 7 * cube_size] = intE(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
            
            // Bottom Face
            // i = Nx - 1
            // j \in [0,Ny-2]
            // k \in [1,Nz-2]
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz)] = intRhoBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + cube_size] = intRhoVXBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 4 * cube_size] = intBXBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 5 * cube_size] = intBYBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 6 * cube_size] = intBZBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 7 * cube_size] = intEBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // After the above is done, there are still FIVE lines where data has not been specified
        // {(i, Ny-1, 0), (Nx-1, j, 0), (Nx-1, Ny-1, k), (Nx-1, j, Nz-1), (i, Ny-1, Nz-1)} 
        // {"FrontRight", "FrontBottom", "BottomRight", "BackBottom", "BackRight"}    
        // The values on the Back Face, k = Nz-1, will be handled by PBCs

        // "FrontRight"
        if (i < Nx-1){
            // (i, Ny-1, 0) - "FrontRight"
            intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz)] = intRhoFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + cube_size] = intRhoVXFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 4 * cube_size] = intBXFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 5 * cube_size] = intBYFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 6 * cube_size] = intBZFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 7 * cube_size] = intEFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // "FrontBottom"
        if (j < Ny-1){
            // (Nx-1, j, 0) - "FrontBottom"
            intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz)] = intRhoFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + cube_size] = intRhoVXFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 4 * cube_size] = intBXFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 5 * cube_size] = intBYFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 6 * cube_size] = intBZFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 7 * cube_size] = intEFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // (Nx-1, Ny-1, k) - "BottomRight"
        if (k < Nz-1){
            intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz)] = intRhoBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + cube_size] = intRhoVXBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 4 * cube_size] = intBXBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 5 * cube_size] = intBYBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 6 * cube_size] = intBZBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 7 * cube_size] = intEBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // Avoid race condition when setting PBCs
        __syncthreads(); 

        // PBCs @ k = Nz-1
        if (i < Nx && j < Ny){ // Do not ignore the edges b/c intermediate variables got calculated there
            for (int ivf = 0; ivf < 8; ivf++){ // PBCs - they are the SAME point 
                intvar[IDX3D(i, j, Nz - 1, Nx, Ny, Nz) + ivf * cube_size] = intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + ivf * cube_size];
            }
        }

        return;
    }

/* ADD DIFFUSION */
__global__ void ComputeIntermediateVariablesBoundary(const float* fluidvar, float* intvar,
    const float D, const float dt, const float dx, const float dy, const float dz,
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
        // i \in [1,Nx-2]
        // j \in [1,Ny-2]
        if (i < Nx-1 && j < Ny-1){
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz)] = intRho(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + cube_size] = intRhoVX(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 4 * cube_size] = intBX(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 5 * cube_size] = intBY(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 6 * cube_size] = intBZ(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 7 * cube_size] = intE(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
        }

        /* SURELY THESE DON'T NEED DIFFUSION */
        // Left and Right Faces
        // j = {0, Ny - 1}
        // i \in [0,Nx-2] 
        // k \in [1,Nz-2]
        if (i < Nx-1 && k > 0 && k < Nz-1){

            // Left Face
            // j = 0
            // i \in [0,Nx-2] 
            // k \in [1,Nz-2]
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz)] = intRhoRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + cube_size] = intRhoVXRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 4 * cube_size] = intBXRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 5 * cube_size] = intBYRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 6 * cube_size] = intBZRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 7 * cube_size] = intERight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e

            // Right Face
            // j = Ny-1
            // i \in [0,Nx-2] 
            // k \in [1,Nz-2]
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz)] = intRho(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + cube_size] = intRhoVX(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 4 * cube_size] = intBX(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 5 * cube_size] = intBY(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 6 * cube_size] = intBZ(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 7 * cube_size] = intE(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        /* SURELY THESE DON'T NEED DIFFUSION */
        // Top and Bottom Faces
        // i = {0, Nx-1}
        // j \in [0,Ny-2]
        // k \in [1,Nz-2]
        if (j < Ny-1 && k > 0 && k < Nz-1){
            
            // Top Face
            // i = 0
            // j \in [0,Ny-2]
            // k \in [1,Nz-2]
            intvar[IDX3D(0, j, k, Nx, Ny, Nz)] = intRho(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + cube_size] = intRhoVX(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 4 * cube_size] = intBX(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 5 * cube_size] = intBY(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 6 * cube_size] = intBZ(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 7 * cube_size] = intE(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
            
            // Bottom Face
            // i = Nx - 1
            // j \in [0,Ny-2]
            // k \in [1,Nz-2]
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz)] = intRhoBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + cube_size] = intRhoVXBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 4 * cube_size] = intBXBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 5 * cube_size] = intBYBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 6 * cube_size] = intBZBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 7 * cube_size] = intEBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        /* SURELY THESE DON'T NEED DIFFUSION */
        // After the above is done, there are still FIVE lines where data has not been specified
        // {(i, Ny-1, 0), (Nx-1, j, 0), (Nx-1, Ny-1, k), (Nx-1, j, Nz-1), (i, Ny-1, Nz-1)} 
        // {"FrontRight", "FrontBottom", "BottomRight", "BackBottom", "BackRight"}
        
        // "FrontRight" and "BackRight"
        if (i < Nx-1){
            // (i, Ny-1, 0) - "FrontRight"
            intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz)] = intRhoFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + cube_size] = intRhoVXFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 4 * cube_size] = intBXFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 5 * cube_size] = intBYFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 6 * cube_size] = intBZFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 7 * cube_size] = intEFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
            
            /* 
            I THINK THE FOLLOWING IS UNNECESSARY
            COMPUTING ANYTHING TO DO WITH THE Back Face IS POINTLESS 
            */
            // (i, Ny-1, Nz-1) - "BackRight"
            // intvar[IDX3D(i, Ny-1, Nz-1, Nx, Ny, Nz)] = intRhoBackRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            // intvar[IDX3D(i, Ny-1, Nz-1, Nx, Ny, Nz) + cube_size] = intRhoVXBackRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            // intvar[IDX3D(i, Ny-1, Nz-1, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYBackRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            // intvar[IDX3D(i, Ny-1, Nz-1, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZBackRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            // intvar[IDX3D(i, Ny-1, Nz-1, Nx, Ny, Nz) + 4 * cube_size] = intBXBackRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            // intvar[IDX3D(i, Ny-1, Nz-1, Nx, Ny, Nz) + 5 * cube_size] = intBYBackRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            // intvar[IDX3D(i, Ny-1, Nz-1, Nx, Ny, Nz) + 6 * cube_size] = intBZBackRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            // intvar[IDX3D(i, Ny-1, Nz-1, Nx, Ny, Nz) + 7 * cube_size] = intEBackRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // "FrontBottom" and "BackBottom"
        if (j < Ny-1){
            // (Nx-1, j, 0) - "FrontBottom"
            intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz)] = intRhoFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + cube_size] = intRhoVXFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 4 * cube_size] = intBXFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 5 * cube_size] = intBYFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 6 * cube_size] = intBZFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 7 * cube_size] = intEFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
            
            /* 
            I THINK THE FOLLOWING IS UNNECESSARY
            COMPUTING ANYTHING TO DO WITH THE Back Face IS POINTLESS 
            */
            // (Nx-1, j, Nz-1) - "BackBottom"
            // intvar[IDX3D(Nx-1, j, Nz-1, Nx, Ny, Nz)] = intRhoBackBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            // intvar[IDX3D(Nx-1, j, Nz-1, Nx, Ny, Nz) + cube_size] = intRhoVXBackBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            // intvar[IDX3D(Nx-1, j, Nz-1, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYBackBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            // intvar[IDX3D(Nx-1, j, Nz-1, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZBackBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            // intvar[IDX3D(Nx-1, j, Nz-1, Nx, Ny, Nz) + 4 * cube_size] = intBXBackBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            // intvar[IDX3D(Nx-1, j, Nz-1, Nx, Ny, Nz) + 5 * cube_size] = intBYBackBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            // intvar[IDX3D(Nx-1, j, Nz-1, Nx, Ny, Nz) + 6 * cube_size] = intBZBackBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            // intvar[IDX3D(Nx-1, j, Nz-1, Nx, Ny, Nz) + 7 * cube_size] = intEBackBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // (Nx-1, Ny-1, k) - "BottomRight"
        if (k < Nz-1){
            intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz)] = intRhoBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
            intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + cube_size] = intRhoVXBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
            intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
            intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
            intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 4 * cube_size] = intBXBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
            intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 5 * cube_size] = intBYBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
            intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 6 * cube_size] = intBZBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
            intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 7 * cube_size] = intEBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        }

        // Avoid race condition when setting PBCs
        __syncthreads(); 

        // PBCs @ Nz-1
        if (i < Nx && j < Ny){ // Do not ignore the edges b/c intermediate variables got calculated there
            for (int ivf = 0; ivf < 8; ivf++){ // PBCs - they are the SAME point 
                intvar[IDX3D(i, j, Nz - 1, Nx, Ny, Nz) + ivf * cube_size] = intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + ivf * cube_size];
            }
        }

        return;
    }

// Microkernels to eliminate the problem of thread divergence in the megakernels that make them a bottleneck
__global__ void QintBdryFrontNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;

        int cube_size = Nx * Ny * Nz;

        if (i < Nx-1 && j < Ny-1){
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz)] = intRho(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + cube_size] = intRhoVX(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 4 * cube_size] = intBX(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 5 * cube_size] = intBY(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 6 * cube_size] = intBZ(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
            intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + 7 * cube_size] = intE(i, j, 0, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz);
        }
        return;
    }

__global__ void QintBdryPBCs(const float* fluidvar, float* intvar,
    const int Nx, const int Ny, const int Nz)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;

        int cube_size = Nx * Ny * Nz;

        // PBCs @ k = Nz-1
        if (i < Nx && j < Ny){ // Do not ignore the edges b/c intermediate variables got calculated there
            for (int ivf = 0; ivf < 8; ivf++){ // PBCs - they are the SAME point 
                intvar[IDX3D(i, j, Nz - 1, Nx, Ny, Nz) + ivf * cube_size] = intvar[IDX3D(i, j, 0, Nx, Ny, Nz) + ivf * cube_size];
            }
        }
        return;
    }

// Can deal with Left and Right faces at the same time
__global__ void QintBdryLeftRightNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    { 
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int k = threadIdx.z + blockDim.z * blockIdx.z;

    int cube_size = Nx * Ny * Nz;

    if (i < Nx-1 && k > 0 && k < Nz-1){
        // Left Face
        // j = 0
        // i \in [0,Nx-2] 
        // k \in [1,Nz-2]
        intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz)] = intRhoRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
        intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + cube_size] = intRhoVXRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
        intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
        intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
        intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 4 * cube_size] = intBXRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
        intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 5 * cube_size] = intBYRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
        intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 6 * cube_size] = intBZRight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
        intvar[IDX3D(i, Ny-1, k, Nx, Ny, Nz) + 7 * cube_size] = intERight(i, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e

        // Right Face
        // j = Ny-1
        // i \in [0,Nx-2] 
        // k \in [1,Nz-2]
        intvar[IDX3D(i, 0, k, Nx, Ny, Nz)] = intRho(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
        intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + cube_size] = intRhoVX(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
        intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
        intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
        intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 4 * cube_size] = intBX(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
        intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 5 * cube_size] = intBY(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
        intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 6 * cube_size] = intBZ(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
        intvar[IDX3D(i, 0, k, Nx, Ny, Nz) + 7 * cube_size] = intE(i, 0, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
    }

    return;
}

// Can deal with Top and Bottom Faces at the same time
__global__ void QintBdryTopBottomNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int k = threadIdx.z + blockDim.z * blockIdx.z;

    int cube_size = Nx * Ny * Nz;

    if (j < Ny-1 && k > 0 && k < Nz-1){
        // Top Face
        // i = 0
        // j \in [0,Ny-2]
        // k \in [1,Nz-2]
        intvar[IDX3D(0, j, k, Nx, Ny, Nz)] = intRho(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
        intvar[IDX3D(0, j, k, Nx, Ny, Nz) + cube_size] = intRhoVX(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
        intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVY(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
        intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZ(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
        intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 4 * cube_size] = intBX(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
        intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 5 * cube_size] = intBY(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
        intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 6 * cube_size] = intBZ(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
        intvar[IDX3D(0, j, k, Nx, Ny, Nz) + 7 * cube_size] = intE(0, j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
        
        // Bottom Face
        // i = Nx - 1
        // j \in [0,Ny-2]
        // k \in [1,Nz-2]
        intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz)] = intRhoBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
        intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + cube_size] = intRhoVXBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
        intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
        intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
        intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 4 * cube_size] = intBXBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
        intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 5 * cube_size] = intBYBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
        intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 6 * cube_size] = intBZBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
        intvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 7 * cube_size] = intEBottom(j, k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
    }

    return;
}

__global__ void QintBdryFrontRightNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    int cube_size = Nx * Ny * Nz;

    // (i, Ny-1, 0) - "FrontRight"
    if (i < Nx-1){
        intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz)] = intRhoFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
        intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + cube_size] = intRhoVXFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
        intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
        intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
        intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 4 * cube_size] = intBXFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
        intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 5 * cube_size] = intBYFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
        intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 6 * cube_size] = intBZFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
        intvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 7 * cube_size] = intEFrontRight(i, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e   
    }

    return;
} 

__global__ void QintBdryFrontBottomNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz) 
    {
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    int cube_size = Nx * Ny * Nz;

    // (Nx-1, j, 0) - "FrontBottom"
    if (j < Ny-1){
        intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz)] = intRhoFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
        intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + cube_size] = intRhoVXFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
        intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
        intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
        intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 4 * cube_size] = intBXFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
        intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 5 * cube_size] = intBYFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
        intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 6 * cube_size] = intBZFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
        intvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 7 * cube_size] = intEFrontBottom(j, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
    }

    return;
}

__global__ void QintBdryBottomRightNoDiff(const float* fluidvar, float* intvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
    int k = threadIdx.z + blockDim.z * blockIdx.z;

    int cube_size = Nx * Ny * Nz;

    // (Nx-1, Ny-1, k) - "BottomRight"
    if (k < Nz-1){
        intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz)] = intRhoBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rho
        intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + cube_size] = intRhoVXBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_x
        intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 2 * cube_size] = intRhoVYBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_y
        intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 3 * cube_size] = intRhoVZBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // rhov_z
        intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 4 * cube_size] = intBXBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bx
        intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 5 * cube_size] = intBYBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // By
        intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 6 * cube_size] = intBZBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // Bz
        intvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 7 * cube_size] = intEBottomRight(k, fluidvar, dt, dx, dy, dz, Nx, Ny, Nz); // e
    }

    return;
}

// Device kernels that deal with the Right Face
// j = Ny-1
// i \in [0, Nx-1]
// k \in [1, Nz-2]
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

// Device kernels that deal with Bottom Face
// i = Nx-1
// j \in [1,Ny-1]
// k \in [1,Nz-2] - leave the front / back faces alone
__device__ float intRhoBottom(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz)]
            - (dt / dx) * (-XFluxRho(Nx-1, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRho(Nx-1, j+1, k, fluidvar, Nx, Ny, Nz) - YFluxRho(Nx-1, j, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxRho(Nx-1, j, k+1, fluidvar, Nx, Ny, Nz) - ZFluxRho(Nx-1, j, k, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVXBottom(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (-XFluxRhoVX(Nx-1, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVX(Nx-1, j+1, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVX(Nx-1, j, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxRhoVX(Nx-1, j, k+1, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(Nx-1, j, k, fluidvar, Nx, Ny, Nz));
    }
    
__device__ float intRhoVYBottom(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
            - (dt / dx) * (-XFluxRhoVY(Nx-1, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVY(Nx-1, j+1, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVY(Nx-1, j, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxRhoVY(Nx-1, j, k+1, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(Nx-1, j, k, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVZBottom(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 3 * Nx * Ny * Nz]
            - (dt / dx) * (-XFluxRhoVZ(Nx-1, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVZ(Nx-1, j+1, k, fluidvar, Nx, Ny, Nz) - YFluxRhoVZ(Nx-1, j, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxRhoVZ(Nx-1, j, k+1, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(Nx-1, j, k, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBXBottom(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
            - (dt / dx) * (-XFluxBX(Nx-1, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBX(Nx-1, j+1, k, fluidvar, Nx, Ny, Nz) - YFluxBX(Nx-1, j, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxBX(Nx-1, j, k+1, fluidvar, Nx, Ny, Nz) - ZFluxBX(Nx-1, j, k, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBYBottom(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
            - (dt / dx) * (-XFluxBY(Nx-1, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBY(Nx-1, j+1, k, fluidvar, Nx, Ny, Nz) - YFluxBY(Nx-1, j, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxBY(Nx-1, j, k+1, fluidvar, Nx, Ny, Nz) - ZFluxBY(Nx-1, j, k, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBZBottom(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
            - (dt / dx) * (-XFluxBZ(Nx-1, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBZ(Nx-1, j+1, k, fluidvar, Nx, Ny, Nz) - YFluxBZ(Nx-1, j, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxBZ(Nx-1, j, k+1, fluidvar, Nx, Ny, Nz) - ZFluxBZ(Nx-1, j, k, fluidvar, Nx, Ny, Nz));
    }

__device__ float intEBottom(const int j, const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, k, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
            - (dt / dx) * (-XFluxE(Nx-1, j, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxE(Nx-1, j+1, k, fluidvar, Nx, Ny, Nz) - YFluxE(Nx-1, j, k, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxE(Nx-1, j, k+1, fluidvar, Nx, Ny, Nz) - ZFluxE(Nx-1, j, k, fluidvar, Nx, Ny, Nz));
    }

/* 
I THINK THE FOLLOWING IS UNNECESSARY
COMPUTING ANYTHING TO DO WITH THE Back Face IS POINTLESS 
*/
// Device kernels that deal with the Back Face 
// k = Nz-1
// i \in [1, Nx-2]
// j \in [1, Nz-2]
__device__ float intRhoBack(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, j, Nz-1, Nx, Ny, Nz)]
            - (dt / dx) * (XFluxRho(i+1, j, Nz-1, fluidvar, Nx, Ny, Nz) - XFluxRho(i, j, Nz-1, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxRho(i, j+1, Nz-1, fluidvar, Nx, Ny, Nz) - YFluxRho(i, j, Nz-1, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxRho(i, j, 1, fluidvar, Nx, Ny, Nz) - ZFluxRho(i, j, Nz-1, fluidvar, Nx, Ny, Nz)); // PBCs
    }

__device__ float intRhoVXBack(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVX(i+1, j, Nz-1, fluidvar, Nx, Ny, Nz) - XFluxRhoVX(i, j, Nz-1, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxRhoVX(i, j+1, Nz-1, fluidvar, Nx, Ny, Nz) - YFluxRhoVY(i, j, Nz-1, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxRhoVX(i, j, 1, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(i, j, Nz-1, fluidvar, Nx, Ny, Nz)); // PBCs
    }

__device__ float intRhoVYBack(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVY(i+1, j, Nz-1, fluidvar, Nx, Ny, Nz) - XFluxRhoVY(i, j, Nz-1, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxRhoVY(i, j+1, Nz-1, fluidvar, Nx, Ny, Nz) - YFluxRhoVY(i, j, Nz-1, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxRhoVY(i, j, 1, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(i, j, Nz-1, fluidvar, Nx, Ny, Nz)); // PBCs
    }

__device__ float intRhoVZBack(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 3 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVZ(i+1, j, Nz-1, fluidvar, Nx, Ny, Nz) - XFluxRhoVZ(i, j, Nz-1, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxRhoVZ(i, j+1, Nz-1, fluidvar, Nx, Ny, Nz) - YFluxRhoVZ(i, j, Nz-1, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRhoVZ(i, j, 1, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, j, Nz-1, fluidvar, Nx, Ny, Nz)); // PBCs
    }

__device__ float intBXBack(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBX(i+1, j, Nz-1, fluidvar, Nx, Ny, Nz) - XFluxBX(i, j, Nz-1, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxBX(i, j+1, Nz-1, fluidvar, Nx, Ny, Nz) - YFluxBX(i, j, Nz-1, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxBX(i, j, 1, fluidvar, Nx, Ny, Nz) - ZFluxBX(i, j, Nz-1, fluidvar, Nx, Ny, Nz)); // PBCs
    }

__device__ float intBYBack(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBY(i+1, j, Nz-1, fluidvar, Nx, Ny, Nz) - XFluxBY(i, j, Nz-1, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxBY(i, j+1, Nz-1, fluidvar, Nx, Ny, Nz) - YFluxBY(i, j, Nz-1, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxBY(i, j, 1, fluidvar, Nx, Ny, Nz) - ZFluxBY(i, j, Nz-1, fluidvar, Nx, Ny, Nz)); // PBCs
    }

__device__ float intBZBack(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBZ(i+1, j, Nz-1, fluidvar, Nx, Ny, Nz) - XFluxBZ(i, j, Nz-1, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxBZ(i, j+1, Nz-1, fluidvar, Nx, Ny, Nz) - YFluxBZ(i, j, Nz-1, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxBZ(i, j, 1, fluidvar, Nx, Ny, Nz) - ZFluxBZ(i, j, Nz-1, fluidvar, Nx, Ny, Nz)); // PBCs
    }

__device__ float intEBack(const int i, const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, j, Nz-1, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxE(i+1, j, Nz-1, fluidvar, Nx, Ny, Nz) - XFluxE(i, j, Nz-1, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (YFluxE(i, j+1, Nz-1, fluidvar, Nx, Ny, Nz) - YFluxE(i, j, Nz-1, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxE(i, j, 1, fluidvar, Nx, Ny, Nz) - ZFluxE(i, j, Nz-1, fluidvar, Nx, Ny, Nz)); // PBCs
    }

// Device kernels that deal with the FrontRight line
// i \in [0,Nx-2]
// j = Ny-1
// k = 0
__device__ float intRhoFrontRight(const int i, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz)]
            - (dt / dx) * (XFluxRho(i+1, Ny-1, 0, fluidvar, Nx, Ny, Nz) - XFluxRho(i, Ny-1, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (-YFluxRho(i, Ny-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRho(i, Ny-1, 1, fluidvar, Nx, Ny, Nz) - ZFluxRho(i, Ny-1, 0, fluidvar, Nx, Ny, Nz));        
    }

__device__ float intRhoVXFrontRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVX(i+1, Ny-1, 0, fluidvar, Nx, Ny, Nz) - XFluxRhoVX(i, Ny-1, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (-YFluxRhoVX(i, Ny-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRhoVX(i, Ny-1, 1, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(i, Ny-1, 0, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVYFrontRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVY(i+1, Ny-1, 0, fluidvar, Nx, Ny, Nz) - XFluxRhoVY(i, Ny-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (-YFluxRhoVY(i, Ny-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRhoVY(i, Ny-1, 1, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(i, Ny-1, 0, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVZFrontRight(const int i, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 3 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxRhoVZ(i+1, Ny-1, 0, fluidvar, Nx, Ny, Nz) - XFluxRhoVZ(i, Ny-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (-YFluxRhoVZ(i, Ny-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRhoVZ(i, Ny-1, 1, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(i, Ny-1, 0, fluidvar, Nx, Ny, Nz));  
    }

__device__ float intBXFrontRight(const int i, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBX(i+1, Ny-1, 0, fluidvar, Nx, Ny, Nz) - XFluxBX(i, Ny-1, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (-YFluxBX(i, Ny-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxBX(i, Ny-1, 1, fluidvar, Nx, Ny, Nz) - ZFluxBX(i, Ny-1, 0, fluidvar, Nx, Ny, Nz));  
    }

__device__ float intBYFrontRight(const int i, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBY(i+1, Ny-1, 0, fluidvar, Nx, Ny, Nz) - XFluxBY(i, Ny-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (-YFluxBY(i, Ny-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxBY(i, Ny-1, 1, fluidvar, Nx, Ny, Nz) - ZFluxBY(i, Ny-1, 0, fluidvar, Nx, Ny, Nz));  
    }

__device__ float intBZFrontRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxBZ(i+1, Ny-1, 0, fluidvar, Nx, Ny, Nz) - XFluxBZ(i, Ny-1, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (-YFluxBZ(i, Ny-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxBZ(i, Ny-1, 1, fluidvar, Nx, Ny, Nz) - ZFluxBZ(i, Ny-1, 0, fluidvar, Nx, Ny, Nz));  
    }

__device__ float intEFrontRight(const int i, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, 0, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
            - (dt / dx) * (XFluxE(i, Ny-1, 0, fluidvar, Nx, Ny, Nz) - XFluxE(i, Ny-1, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dy) * (-YFluxE(i, Ny-1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxE(i, Ny-1, 1, fluidvar, Nx, Ny, Nz) - ZFluxE(i, Ny-1, 0, fluidvar, Nx, Ny, Nz));  
    }

// Kernels that deal with FrontBottom Line
// i = Nx-1
// j \in [0, Ny-2]
// k = 0
__device__ float intRhoFrontBottom(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz)]
            - (dt / dx) * (-XFluxRho(Nx-1, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall 
            - (dt / dy) * (YFluxRho(Nx-1, j+1, 0, fluidvar, Nx, Ny, Nz) - YFluxRho(Nx-1, j, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxRho(Nx-1, j, 1, fluidvar, Nx, Ny, Nz) - ZFluxRho(Nx-1, j, 0, fluidvar, Nx, Ny, Nz));       
    }

__device__ float intRhoVXFrontBottom(const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (-XFluxRhoVX(Nx-1, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVX(Nx-1, j+1, 0, fluidvar, Nx, Ny, Nz) - YFluxRhoVX(Nx-1, j, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxRhoVX(Nx-1, j, 1, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(Nx-1, j, 0, fluidvar, Nx, Ny, Nz)); 
    }

__device__ float intRhoVYFrontBottom(const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
            - (dt / dx) * (-XFluxRhoVY(Nx-1, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVY(Nx-1, j+1, 0, fluidvar, Nx, Ny, Nz) - YFluxRhoVY(Nx-1, j, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxRhoVY(Nx-1, j, 1, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(Nx-1, j, 0, fluidvar, Nx, Ny, Nz)); 
    }

__device__ float intRhoVZFrontBottom(const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 3 * Nx * Ny * Nz]
            - (dt / dx) * (-XFluxRhoVZ(Nx-1, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVZ(Nx-1, j+1, 0, fluidvar, Nx, Ny, Nz) - YFluxRhoVZ(Nx-1, j, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxRhoVZ(Nx-1, j, 1, fluidvar, Nx, Ny, Nz) - YFluxRhoVZ(Nx-1, j, 0, fluidvar, Nx, Ny, Nz)); 
    }

__device__ float intBXFrontBottom(const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
            - (dt / dx) * (-XFluxBX(Nx-1, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBX(Nx-1, j+1, 0, fluidvar, Nx, Ny, Nz) - YFluxBX(Nx-1, j, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxBX(Nx-1, j, 1, fluidvar, Nx, Ny, Nz) - ZFluxBX(Nx-1, j, 0, fluidvar, Nx, Ny, Nz)); 
    }

__device__ float intBYFrontBottom(const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
            - (dt / dx) * (-XFluxBY(Nx-1, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBY(Nx-1, j+1, 0, fluidvar, Nx, Ny, Nz) - YFluxBY(Nx-1, j, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxBY(Nx-1, j, 1, fluidvar, Nx, Ny, Nz) - ZFluxBY(Nx-1, j, 0, fluidvar, Nx, Ny, Nz)); 
    }

__device__ float intBZFrontBottom(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
            - (dt / dx) * (-XFluxBZ(Nx-1, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall 
            - (dt / dy) * (YFluxBZ(Nx-1, j+1, 0, fluidvar, Nx, Ny, Nz) - YFluxBZ(Nx-1, j+1, 0, fluidvar, Nx, Ny, Nz))
            - (dt / dz) * (ZFluxBZ(Nx-1, j, 1, fluidvar, Nx, Ny, Nz) - ZFluxBZ(Nx-1, j, 0, fluidvar, Nx, Ny, Nz));   
    }

__device__ float intEFrontBottom(const int j,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, 0, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
            - (dt / dx) * (-XFluxE(Nx-1, j, 0, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall 
            - (dt / dy) * (YFluxE(Nx-1, j+1, 0, fluidvar, Nx, Ny, Nz) - YFluxE(Nx-1, j, 0, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (ZFluxE(Nx-1, j, 1, fluidvar, Nx, Ny, Nz) - YFluxE(Nx-1, j, 0, fluidvar, Nx, Ny, Nz));
    }

// Device kernels that deal with the BottomRight Line
// i = Nx-1
// j = Ny-1
// k \in [0, Nz-2]
__device__ float intRhoBottomRight(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz)]
            - (dt / dx) * (-XFluxRho(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (-YFluxRho(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRho(Nx-1, Ny-1, k+1, fluidvar, Nx, Ny, Nz) - ZFluxRho(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVXBottomRight(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (-XFluxRhoVX(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (-YFluxRhoVX(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVX(Nx-1, Ny-1, k+1, fluidvar, Nx, Ny, Nz) - ZFluxRhoVX(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVYBottomRight(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 2 * Nx * Ny * Nz]
            - (dt / dx) * (-XFluxRhoVY(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (-YFluxRhoVY(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVY(Nx-1, Ny-1, k+1, fluidvar, Nx, Ny, Nz) - ZFluxRhoVY(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz));
    }

__device__ float intRhoVZBottomRight(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + Nx * Ny * Nz]
            - (dt / dx) * (-XFluxRhoVZ(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (-YFluxRhoVZ(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxRhoVZ(Nx-1, Ny-1, k+1, fluidvar, Nx, Ny, Nz) - ZFluxRhoVZ(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBXBottomRight(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 4 * Nx * Ny * Nz]
            - (dt / dx) * (-XFluxBX(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (-YFluxBX(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBX(Nx-1, Ny-1, k+1, fluidvar, Nx, Ny, Nz) - ZFluxBX(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBYBottomRight(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 5 * Nx * Ny * Nz]
            - (dt / dx) * (-XFluxBY(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (-YFluxBY(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBY(Nx-1, Ny-1, k+1, fluidvar, Nx, Ny, Nz) - ZFluxBY(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz));
    }

__device__ float intBZBottomRight(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 6 * Nx * Ny * Nz]
            - (dt / dx) * (-XFluxBZ(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (-YFluxBZ(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxBZ(Nx-1, Ny-1, k+1, fluidvar, Nx, Ny, Nz) - ZFluxBZ(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz));
    }

__device__ float intEBottomRight(const int k,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, Ny-1, k, Nx, Ny, Nz) + 7 * Nx * Ny * Nz]
            - (dt / dx) * (-XFluxE(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (-YFluxE(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (ZFluxE(Nx-1, Ny-1, k+1, fluidvar, Nx, Ny, Nz) - ZFluxE(Nx-1, Ny-1, k, fluidvar, Nx, Ny, Nz));
    }

/* 
I THINK THE FOLLOWING IS UNNECESSARY
COMPUTING ANYTHING TO DO WITH THE Back Face IS POINTLESS 
*/
// Device kernels that deal with the BackBottom Line
// i = Nx-1
// j \in [0, Ny-2]
// k = Nz-1
__device__ float intRhoBackBottom(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, Nz-1, Nx, Ny, Nz)]
            - (dt / dx) * (-XFluxRho(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRho(Nx-1, j+1, Nz-1, fluidvar, Nx, Ny, Nz) - YFluxRho(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (-ZFluxRho(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall        
    }

__device__ float intRhoVXBackBottom(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, Nz-1, Nx, Ny, Nz) + Nx*Ny*Nz]
            - (dt / dx) * (-XFluxRhoVX(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVX(Nx-1, j+1, Nz-1, fluidvar, Nx, Ny, Nz) - YFluxRhoVX(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (-ZFluxRhoVX(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall        
    }

__device__ float intRhoVYBackBottom(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, Nz-1, Nx, Ny, Nz) + 2*Nx*Ny*Nz]
            - (dt / dx) * (-XFluxRhoVY(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVY(Nx-1, j+1, Nz-1, fluidvar, Nx, Ny, Nz) - YFluxRhoVY(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (-ZFluxRhoVY(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall        
    }

__device__ float intRhoVZBackBottom(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, Nz-1, Nx, Ny, Nz) + 3*Nx*Ny*Nz]
            - (dt / dx) * (-XFluxRhoVZ(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxRhoVZ(Nx-1, j+1, Nz-1, fluidvar, Nx, Ny, Nz) - YFluxRhoVZ(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (-ZFluxRhoVZ(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall        
    }

__device__ float intBXBackBottom(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, Nz-1, Nx, Ny, Nz) + 4*Nx*Ny*Nz]
            - (dt / dx) * (-XFluxBX(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBX(Nx-1, j+1, Nz-1, fluidvar, Nx, Ny, Nz) - YFluxBX(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (-ZFluxBX(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall        
    }

__device__ float intBYBackBottom(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, Nz-1, Nx, Ny, Nz) + 5*Nx*Ny*Nz]
            - (dt / dx) * (-XFluxBY(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBY(Nx-1, j+1, Nz-1, fluidvar, Nx, Ny, Nz) - YFluxBY(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (-ZFluxBY(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall        
    }

__device__ float intBZBackBottom(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, Nz-1, Nx, Ny, Nz) + 3*Nx*Ny*Nz]
            - (dt / dx) * (-XFluxBZ(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxBZ(Nx-1, j+1, Nz-1, fluidvar, Nx, Ny, Nz) - YFluxBZ(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (-ZFluxBZ(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall        
    }

__device__ float intEBackBottom(const int j, 
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(Nx-1, j, Nz-1, Nx, Ny, Nz) + 3*Nx*Ny*Nz]
            - (dt / dx) * (-XFluxE(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dy) * (YFluxE(Nx-1, j+1, Nz-1, fluidvar, Nx, Ny, Nz) - YFluxE(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)) 
            - (dt / dz) * (-ZFluxE(Nx-1, j, Nz-1, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall        
    }

/* 
I THINK THE FOLLOWING IS UNNECESSARY
COMPUTING ANYTHING TO DO WITH THE Back Face IS POINTLESS 
*/
// Device kernels that deal with BackRight
// i \in [0, Nx-2]
// j = Ny-1
// k = Nz-1 
__device__ float intRhoBackRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, Nz-1, Nx, Ny, Nz)]
            - (dt / dx) * (XFluxRho(i+1, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz) - XFluxRho(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (-YFluxRho(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (-ZFluxRho(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
    }

__device__ float intRhoVXBackRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, Nz-1, Nx, Ny, Nz) + Nx*Ny*Nz]
            - (dt / dx) * (XFluxRhoVX(i+1, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz) - XFluxRhoVX(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (-YFluxRhoVX(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (-ZFluxRhoVX(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
    }

__device__ float intRhoVYBackRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, Nz-1, Nx, Ny, Nz) + 2*Nx*Ny*Nz]
            - (dt / dx) * (XFluxRhoVY(i+1, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz) - XFluxRhoVY(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (-YFluxRhoVY(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (-ZFluxRhoVY(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
    }

__device__ float intRhoVZBackRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, Nz-1, Nx, Ny, Nz) + 3*Nx*Ny*Nz]
            - (dt / dx) * (XFluxRhoVZ(i+1, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz) - XFluxRhoVZ(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (-YFluxRhoVZ(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (-ZFluxRhoVZ(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
    }

__device__ float intBXBackRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, Nz-1, Nx, Ny, Nz) + 4*Nx*Ny*Nz]
            - (dt / dx) * (XFluxBX(i+1, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz) - XFluxBX(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (-YFluxBX(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (-ZFluxBX(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
    }

__device__ float intBYBackRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, Nz-1, Nx, Ny, Nz) + 5*Nx*Ny*Nz]
            - (dt / dx) * (XFluxBY(i+1, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz) - XFluxBY(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (-YFluxBY(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (-ZFluxBY(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
    }

__device__ float intBZBackRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, Nz-1, Nx, Ny, Nz) + 6*Nx*Ny*Nz]
            - (dt / dx) * (XFluxBZ(i+1, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz) - XFluxBZ(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (-YFluxBZ(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (-ZFluxBZ(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
    }

__device__ float intEBackRight(const int i,
    const float* fluidvar,
    const float dt, const float dx, const float dy, const float dz,
    const int Nx, const int Ny, const int Nz)
    {
        return fluidvar[IDX3D(i, Ny-1, Nz-1, Nx, Ny, Nz) + 7*Nx*Ny*Nz]
            - (dt / dx) * (XFluxE(i+1, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz) - XFluxE(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz))
            - (dt / dy) * (-YFluxE(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz)) // No flux on opposite side of rigid, perfectly-conducting wall
            - (dt / dz) * (-ZFluxE(i, Ny-1, Nz-1, fluidvar, Nx, Ny, Nz)); // No flux on opposite side of rigid, perfectly-conducting wall
    }
// I think that's everything! 