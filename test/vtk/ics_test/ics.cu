/* 
WIP 
Visualize the initial conditions
(1) Compute ICs
(2) Output them for visualization
*/

#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <cstdio>

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char* argv[]){
    int Nx = atoi(argv[1]);
    int Ny = atoi(argv[2]);
    int Nz = atoi(argv[3]);

    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    int num_blocks = 2 * numberOfSMs;
    int num_threads_per_block = 1024;

    dim3 grid_dimensions(num_blocks, num_blocks, num_blocks);
    dim3 block_dimensions(num_threads_per_block, num_threads_per_block, num_threads_per_block);

    float *rho, *rhov_x, *rhov_y, *rhov_z, *Bx, *By, *Bz, *e;
    float *x_grid, *y_grid, *z_grid;

    checkCuda(cudaMalloc(&rho, sizeof(float)*Nx*Ny*Nz));
    checkCuda(cudaMalloc(&rhov_x, sizeof(float)*Nx*Ny*Nz));
    checkCuda(cudaMalloc(&rhov_y, sizeof(float)*Nx*Ny*Nz));
    checkCuda(cudaMalloc(&rhov_z, sizeof(float)*Nx*Ny*Nz));
    checkCuda(cudaMalloc(&Bx, sizeof(float)*Nx*Ny*Nz));
    checkCuda(cudaMalloc(&By, sizeof(float)*Nx*Ny*Nz));
    checkCuda(cudaMalloc(&Bz, sizeof(float)*Nx*Ny*Nz));
    checkCuda(cudaMalloc(&e, sizeof(float)*Nx*Ny*Nz));

    checkCuda(cudaMalloc(&x_grid, sizeof(float)*Nx*Ny*Nz));
    checkCuda(cudaMalloc(&y_grid, sizeof(float)*Nx*Ny*Nz));
    checkCuda(cudaMalloc(&z_grid, sizeof(float)*Nx*Ny*Nz));

    InitializeGrid<<<grid_dimensions, block_dimensions>>>(x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz, x_grid, y_grid, z_grid, Nx, Ny, Nz);
    checkCuda(cudaDeviceSynchronize());
    InitialConditions<<<grid_dimensions, block_dimensions>>>(rho, rhov_x, rhov_y, rhov_z, Bx, By, Bz, e, 1.0, x_grid, y_grid, z_grid, Nx, Ny, Nz);
    checkCuda(cudaDeviceSynchronize());

    return 0;
}