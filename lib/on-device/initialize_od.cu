#include <stdio.h>

#include "initialize_od.cuh"
#include "utils.cuh"

#include <string>
#include <map>
#include <functional>
#include <stdexcept>
#include <iostream>

#define IDX3D(i, j, k, Nx, Ny, Nz) ((k) * (Nx) * (Ny) + (i) * (Ny) + j) // parentheses are necessary to avoid calculating `i - 1 * Ny` or `k - 1 * (Nx * Ny)`

// I don't want to have a separate file for each equilibrium
class SimulationInitializer {
    private:
        using KernelLauncher = std::function<void(float*, const InitConfig&, const dim3, const dim3)>;
        std::map<std::string, KernelLauncher> initFunctions;
        InitConfig config;
        dim3 egd_init, tbd_init;
    
    public: 
        SimulationInitializer(const InitConfig& config, const dim3 egd_init, const dim3 tbd_init) : config(config), egd_init(egd_init), tbd_init(tbd_init) {
            initFunctions["screwpinch"] = [this](float* data, const InitConfig& cfg, const dim3 egd_init, const dim3 tbd_init) {
                // ScrewPinch<<<cfg.gridDim, cfg.blockDim>>>(data);
                LaunchScrewPinch(data, cfg, egd_init, tbd_init); // Do not want to pass cfg to GPU or make this code less readable by passing long list of cfg parameters
            };
            initFunctions["screwpinch-stride"]=[this](float* data, const InitConfig& cfg, const dim3 egd_init, const dim3 tbd_init) {
                LaunchScrewPinchStride(data, cfg, egd_init, tbd_init);
            };
            /* ADD OTHER INITIALIZERS */
        }

        void initialize(const std::string& simType, float* data){
            auto it = initFunctions.find(simType);
            if (it == initFunctions.end()) {
                throw std::runtime_error("Unknown simulation type: " + simType);
            }
            it->second(data, config, egd_init, tbd_init);
        }
};

// 16 registers / thread
/* NOTE: Better way to do this is split into microkernels */ 
__global__ void InitializeGrid(const float x_min, const float x_max, const float y_min, const float y_max, const float z_min, const float z_max, 
    const float dx, const float dy, const float dz, float* grid_x, float* grid_y, float* grid_z, 
    const int Nx, const int Ny, const int Nz)
    {
        int tidx = threadIdx.x + blockDim.x * blockIdx.x;
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;

        if (tidx < Nx && tidy < Ny && tidz < Nz){ // better way to do this is split into microkernels
            grid_z[tidz] = z_min + tidz * dz;
            grid_x[tidx] = x_min + tidx * dx;
            grid_y[tidy] = y_min + tidy * dy;
        }

        return;
    }

__global__ void InitializeX(float* grid_x, const float x_min, const float dx, const int Nx)
{
    u_int32_t i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < Nx){
        grid_x[i] = x_min + i * dx;
    }

    return;
}

__global__ void InitializeY(float* grid_y, const float y_min, const float dy, const int Ny)
{
    u_int32_t j = threadIdx.y + blockDim.y * blockIdx.y;

    if (j < Ny){
        grid_y[j] = y_min + j * dy;
    }
    
    return;
}

__global__ void InitializeZ(float* grid_z, const float z_min, const float dz, const int Nz)
{
    u_int32_t k = threadIdx.z + blockDim.z * blockIdx.z;

    if (k < Nz){
        grid_z[k] = z_min + k * dz;
    }
    
    return;
}

/* NOTE: Does this even need to be here? */
__global__ void ZeroVars(float* vars, const int Nx, const int Ny, const int Nz)
{
    u_int32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    u_int32_t j = threadIdx.y + blockDim.y * blockIdx.y;
    u_int32_t k = threadIdx.z + blockDim.z * blockIdx.z;

    u_int32_t cube_size = Nx * Ny * Nz;

    if (i < Nx && j < Ny && k < Nz){
        vars[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.0;
        vars[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0;
        vars[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
        vars[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
        vars[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0;
        vars[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0;
        vars[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
        vars[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = 0.0;
    }
    return;
}

__global__ void ZeroVarsStride(float* vars, const int Nx, const int Ny, const int Nz)
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
                    vars[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.0;
                    vars[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0;
                    vars[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
                    vars[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
                    vars[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0;
                    vars[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0;
                    vars[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
                    vars[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = 0.0; 
            }
        }
    }
    return;
}

__global__ void ScrewPinch(float* fluidvar, 
    const float J0, const float r_max_coeff, 
    const float* grid_x, const float* grid_y, const float* grid_z,
    const int Nx, const int Ny, const int Nz)
    {
        u_int32_t i = threadIdx.x + blockDim.x * blockIdx.x;
        u_int32_t j = threadIdx.y + blockDim.y * blockIdx.y;
        u_int32_t k = threadIdx.z + blockDim.z * blockIdx.z;

        u_int32_t cube_size = Nx * Ny * Nz;

        float r = 0.0;
        float r_pinch = r_max_coeff * sqrtf(pow(grid_x[Nx-1],2) + pow(grid_y[Ny-1],2)); // r_pinch is a fraction of r_max 

        float p = 0.0;

        float Jr = 0.0; // Makes conversion from cylindrical coordinates more readable 
        float Jphi = 0.0;

        float Br = 0.0;
        float Btheta = 0.0;
        float B0 = 1.0;

        float x = 0.0; // More readable
        float y = 0.0;

        if (i < Nx && j < Ny && k < Nz){
            x = grid_x[i];
            y = grid_y[j];
            r = sqrtf(pow(x, 2) + pow(y, 2));
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.01;
        }

        __syncthreads();

        if (r < r_pinch && i < Nx && j < Ny && k < Nz){ // thread divergence is not a big problem in a one-use kernel
            Btheta = 0.5 * J0 * r * (1.0 - 0.5 * pow(r, 2) / pow(r_pinch, 2)); 

            // Calculated from equilibrium force balance
            p = -0.25 * (pow(J0, 2) / pow(r_pinch, 4)) * (pow(r, 6) / 6.0 - 0.75 * pow(r_pinch, 2) * pow(r, 4) + pow(r_pinch, 4) * pow(r, 2));

            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = Jr * x - Jphi * y / r;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = Jr * y + Jphi * x / r;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = J0 * (1 - pow(r, 2) / pow(r_pinch, 2));
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = Br * x - Btheta * y / r;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = Br * y + Btheta * x / r;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = B0;
            fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = (p / (gamma - 1.0)) 
                                                                    + (pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size], 2) 
                                                                        + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size], 2)
                                                                        + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size], 2)
                                                                        ) / (2.0 * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)])
                                                                    + 0.5 * (pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size], 2) 
                                                                    + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size], 2) 
                                                                    + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size], 2));
        }

        return;
    }

void LaunchScrewPinch(float *fluidvar, const InitConfig& cfg, const dim3 gridDim, const dim3 blockDim){
    ScrewPinch<<<gridDim, blockDim>>>(fluidvar, cfg.J0, cfg.r_max_coeff, cfg.x_grid, cfg.y_grid, cfg.z_grid, cfg.Nx, cfg.Ny, cfg.Nz);
    return;
}

// 56 registers / thread
__global__ void ScrewPinchStride(float* fluidvar, const float J0, const float* grid_x, const float* grid_y, const float* grid_z, 
    const int Nx, const int Ny, const int Nz)
    {
        int tidx = threadIdx.x + blockDim.x * blockIdx.x;
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;

        int xthreads = blockDim.x * gridDim.x;
        int ythreads = blockDim.y * gridDim.y;
        int zthreads = blockDim.z * gridDim.z;

        float r = 0.0;
        float r_pinch = 0.25 * sqrtf(pow(grid_x[Nx-1],2) + pow(grid_y[Ny-1],2)); // r_pinch = 0.25 * r_max 

        int cube_size = Nx * Ny * Nz;

        float p = 0.0;

        float Jr = 0.0; // Makes conversion from cylindrical coordinates more readable 
        float Jphi = 0.0;

        float Br = 0.0;
        float Btheta = 0.0;
        float B0 = 1.0;

        float x = 0.0; // More readable
        float y = 0.0;
        for (int k = tidz; k < Nz; k += zthreads){ // THIS LOOP ORDER IMPLEMENTS CONTIGUOUS MEMORY ACCESSES
            for (int i = tidx; i < Nx; i += xthreads){
                for (int j = tidy; j < Ny; j += ythreads){
                    x = grid_x[i];
                    y = grid_y[j];
                    r = sqrtf(pow(x, 2) + pow(y, 2));

                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 0.01;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = 0.0;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = 0.0;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = 0.0;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = 0.0;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = 0.0;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = 0.0;
                    fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = 0.0;
                    
                    if (r < r_pinch){
                        Btheta = 0.5 * J0 * r * (1.0 - 0.5 * pow(r, 2) / pow(r_pinch, 2)); 

                        // Calculated from equilibrium force balance
                        p = -0.25 * (pow(J0, 2) / pow(r_pinch, 4)) * (pow(r, 6) / 6.0 - 0.75 * pow(r_pinch, 2) * pow(r, 4) + pow(r_pinch, 4) * pow(r, 2));

                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)] = 1.0;
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] = Jr * x - Jphi * y / r;
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] = Jr * y + Jphi * x / r;
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] = J0 * (1 - pow(r, 2) / pow(r_pinch, 2));
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] = Br * x - Btheta * y / r;
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] = Br * y + Btheta * x / r;
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size] = B0;
                        fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] = (p / (gamma - 1.0)) 
                                                                                + (pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size], 2) 
                                                                                    + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size], 2)
                                                                                    + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size], 2)
                                                                                    ) / (2.0 * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)])
                                                                                + 0.5 * (pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size], 2) 
                                                                                + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size], 2) 
                                                                                + pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size], 2));    
                        
                        // printf("At offset %d, point (%f, %f, %f), rho is %f, rhovx is %f, rhovy is %f, rhovz is %f, Bx is %f, By is %f, Bz is %f, and energy density is %f\n", 
                        //     IDX3D(i, j, k, Nx, Ny, Nz), x, y, grid_z[k], 
                        //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)], fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size], 
                        //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size], fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size],
                        //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size], fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size],
                        //     fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size], fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size]);
                    }
                }
            }
        }
        return;
    }

void LaunchScrewPinchStride(float *fluidvar, const InitConfig& cfg, const dim3 gridDim, const dim3 blockDim){
    std::cout << "Inside LSPS" << std::endl;
    ScrewPinchStride<<<gridDim, blockDim>>>(fluidvar, cfg.J0, cfg.x_grid, cfg.y_grid, cfg.z_grid, cfg.Nx, cfg.Ny, cfg.Nz);
    // checkCuda(cudaDeviceSynchronize());
    std::cout << "After launching" << std::endl;
    return;
}

__global__ void ZPinch(float* fluidvar, const float Btheta_a, const float* grid_x, const float* grid_y, const float* grid_z, 
    const int Nx, const int Ny, const int Nz)
    {
        int tidx = threadIdx.x + blockDim.x * blockIdx.x;
        int tidy = threadIdx.y + blockDim.y * blockIdx.y;
        int tidz = threadIdx.z + blockDim.z * blockIdx.z;

        int xthreads = blockDim.x * gridDim.x;
        int ythreads = blockDim.y * gridDim.y;
        int zthreads = blockDim.z * gridDim.z;
    
        /* 
        SPECIFY A Z-PINCH EQUILIBRIUM
        */
        return;
    }

