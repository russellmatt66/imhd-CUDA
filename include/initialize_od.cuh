#ifndef INIT_OD_CUH
#define INIT_OD_CUH

// physical constants
#define gamma 5.0 / 3.0
#define q_e 1.6 * pow(10, -19) // [C]
#define m 1.67 * pow(10, -27) // [kg]

__global__ void InitializeGrid(const float x_min, const float x_max, const float y_min, const float y_max, const float z_min, const float z_max, 
    const float dx, const float dy, const float dz, float* grid_x, float* grid_y, float* grid_z, 
    const int Nx, const int Ny, const int Nz);

__global__ void InitialConditions(float* rho, float* rhov_x, float* rhov_y, float* rhov_z, float* Bx, float* By, float* Bz, float* e,
    const float J0, const float* grid_x, const float* grid_y, const float* grid_z,
    const int Nx, const int Ny, const int Nz);

__global__ void InitializeIntAndSwap(float* rho_np1, float* rhovx_np1, float* rhovy_np1, float* rhovz_np1, 
     float* Bx_np1, float* By_np1, float* Bz_np1, float* e_np1,
     float* rho_int, float* rhovx_int, float* rhovy_int, float* rhovz_int, float* Bx_int, float* By_int, float* Bz_int, float* e_int,
     const int Nx, const int Ny, const int Nz);

#endif