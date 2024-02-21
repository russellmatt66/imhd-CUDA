#ifndef INIT_OD_CUH
#define INIT_OD_CUH

// physical constants
#define gamma = 5.0 / 3.0
#define q_e = 1.6 * pow(10,-19) // [C]
#define m = 1.67 * pow(10, -27) // [kg]

__global__ void InitializeGrid(float* grid, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max, 
    const int Nx, const int Ny, const int Nz);

__global__ void InitialConditions(float* rho, float* rhov_x, float* rhov_y, float* rhov_z, float* B_x, float* B_y, float* B_z, float* e,
    const int Nx, const int Ny, const int Nz, const int ICs_flag);

#endif