#include <assert.h>
#include <cstdlib>
#include <stdio.h>

#include "../include/kernels_od.cuh"
#include "../include/initialize_od.cuh"

// #include "../include/kernel_od.cu"

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
	// Parse inputs - provided by imhdLauncher.py because Python parsing is easiest, and Python launching is easy
	int Nt = atoi(argv[1]);
	int Nx = atoi(argv[2]);
	int Ny = atoi(argv[3]);
	int Nz = atoi(argv[4]);
	int SM_mult_x = atoi(argv[5]);
	int SM_mult_y = atoi(argv[6]);
	int SM_mult_z = atoi(argv[7]);
	int num_threads_per_block_x = atoi(argv[8]);
	int num_threads_per_block_y = atoi(argv[9]);
	int num_threads_per_block_z = atoi(argv[10]);
	// float gamma = atof(argv[11]);
	int ics_flag = atoi(argv[11]);
	float x_min = atof(argv[12]);
	float x_max = atof(argv[13]);
	float y_min = atof(argv[14]);
	float y_max = atof(argv[15]);
	float z_min = atof(argv[16]);
	float z_max = atof(argv[17]);

	/* Initialize device data */
	int deviceId;
	int numberOfSMs;

	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	/* Simulation loop */
	for (int it = 0; it < Nt; it++){
		/* Compute interior */

		/* Compute boundaries */

		checkCuda(cudaDeviceSynchronize());
		/* Write data out - use GPUDirect Storage (GDS) */
	}

	/* Free */ 
	return 0;
}