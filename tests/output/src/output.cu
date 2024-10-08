#include <string>

#include "../../../include/initialize_od.cuh"
#include "../../../include/utils.hpp"

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

    float x_min = -3.14159;
    float x_max = 3.14159;
    float y_min = -3.14159;
    float y_max = 3.14159;
    float z_min = -3.14159;
    float z_max = 3.14159;

    float dx = (x_max - x_min) / (Nx - 1);
	float dy = (y_max - y_min) / (Ny - 1);
	float dz = (z_max - z_min) / (Nz - 1);

    float J0 = 1.0;

    int deviceId;
	int numberOfSMs;

	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    float *fluidvar;
	float *grid_x, *grid_y, *grid_z;

	int cube_size = Nx * Ny * Nz;
	int fluid_data_size = sizeof(float) * Nx * Ny * Nz;

	checkCuda(cudaMalloc(&fluidvar, 8 * fluid_data_size));
	checkCuda(cudaMalloc(&grid_x, sizeof(float) * Nx));
	checkCuda(cudaMalloc(&grid_y, sizeof(float) * Ny));
	checkCuda(cudaMalloc(&grid_z, sizeof(float) * Nz));

    int SM_mult_x = 1;
    int SM_mult_y = 1;
    int SM_mult_z = 1;

	dim3 grid_dimensions(SM_mult_x * numberOfSMs, SM_mult_y * numberOfSMs, SM_mult_z * numberOfSMs);
	dim3 block_dims_grid(4, 4, 4);  
	dim3 block_dims_init(4, 4, 4);

	InitializeGrid<<<grid_dimensions, block_dims_grid>>>(x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz,
															grid_x, grid_y, grid_z, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());

	ScrewPinch<<<grid_dimensions, block_dims_init>>>(fluidvar, J0, grid_x, grid_y, grid_z, Nx, Ny, Nz); // Screw-pinch
	checkCuda(cudaDeviceSynchronize());

	float *h_rho, *h_rhovx, *h_rhovy, *h_rhovz, *h_Bx, *h_By, *h_Bz, *h_e;

	h_rho = (float*)malloc(fluid_data_size);
	h_rhovx = (float*)malloc(fluid_data_size);
	h_rhovy = (float*)malloc(fluid_data_size);
	h_rhovz = (float*)malloc(fluid_data_size);
	h_Bx = (float*)malloc(fluid_data_size);
	h_By = (float*)malloc(fluid_data_size);
	h_Bz = (float*)malloc(fluid_data_size);
	h_e = (float*)malloc(fluid_data_size);

    /* Transfer device data to host, and write to .h5 file */

    checkCuda(cudaDeviceSynchronize());
    
	checkCuda(cudaFree(fluidvar));
    checkCuda(cudaFree(grid_x));
	checkCuda(cudaFree(grid_y));
	checkCuda(cudaFree(grid_z));

    free(h_rho);
    free(h_rhovx);
    free(h_rhovy);
    free(h_rhovz);
    free(h_Bx);
    free(h_By);
    free(h_Bz);
    free(h_e);
    return 0;
}