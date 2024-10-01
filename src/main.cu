#include <assert.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include "../include/kernels_od.cuh"
#include "../include/kernels_od_intvar.cuh"
#include "../include/initialize_od.cuh"
#include "../include/gds.cuh"
#include "../include/utils.hpp"

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
/* This REALLY needs to be a library somewhere */
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* REFACTOR THIS ACCORDING TO REFACTORED LIBRARIES */
int main(int argc, char* argv[]){
	/* 
   PARSE ARGUMENTS 
   */
   int Nt = atoi(argv[1]);
   int Nx = atoi(argv[2]);
   int Ny = atoi(argv[3]);
   int Nz = atoi(argv[4]);
   float J0 = atof(argv[5]); // amplitude of the current
	float D = atof(argv[6]); // coefficient of numerical diffusion
	float x_min = atof(argv[7]);
	float x_max = atof(argv[8]);
	float y_min = atof(argv[9]);
	float y_max = atof(argv[10]);
	float z_min = atof(argv[11]);
	float z_max = atof(argv[12]);
	float dt = atof(argv[13]);
   std::string path_to_data = argv[14];
   std::string phdf5_bin_name = argv[15];
   std::string attr_bin_name = argv[16];
   std::string eigen_bin_name = argv[17];

   /* 
   CUDA BOILERPLATE 
   */
   int deviceId;
   int numberOfSMs;

   cudaGetDevice(&deviceId);
   cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

   /* SPECIFY EXECUTION CONFIGURATIONS */
   dim3 exec_grid_dims(numberOfSMs, numberOfSMs, numberOfSMs);
   dim3 mesh_block_dims(8,8,8);
   dim3 init_block_dims(8,8,8);
   dim3 intvar_block_dims(8,8,8);
   dim3 fluid_block_dims(8,8,8);

   /* 
   INITIALIZE SIMULATION DATA
   */
   size_t cube_size = Nx * Ny * Nz;
   size_t fluidvar_size = sizeof(float) * cube_size;
   size_t fluid_data_size = 8 * fluidvar_size;

   float *fluidvars, *fluidvars_np1, *fluidvars_int;

   checkCuda(cudaMalloc(&fluidvars, fluid_data_size));
   checkCuda(cudaMalloc(&fluidvars_np1, fluid_data_size));
   checkCuda(cudaMalloc(&fluidvars_int, fluid_data_size));

   float *x_grid, *y_grid, *z_grid;

   checkCuda(cudaMalloc(&x_grid, sizeof(float) * Nx));
   checkCuda(cudaMalloc(&y_grid, sizeof(float) * Ny));
   checkCuda(cudaMalloc(&z_grid, sizeof(float) * Nz));

   float dx = (x_max - x_min) / (Nx - 1);
	float dy = (y_max - y_min) / (Ny - 1);
	float dz = (z_max - z_min) / (Nz - 1);

   /* CALL INITIALIZATION KERNELS */
   InitializeGrid<<<exec_grid_dims, mesh_block_dims>>>(x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz, x_grid, y_grid, z_grid, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());

   ScrewPinch<<<exec_grid_dims, init_block_dims>>>(fluidvars, J0, x_grid, y_grid, z_grid, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());

   InitializeIntAndSwap<<<exec_grid_dims, intvar_block_dims>>>(fluidvars_np1, fluidvars_int, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());
    
   /* WRITE INITIAL DATA OUT */
   std::string shm_name_fluidvar = "/shared_h_fluidvar";
   int shm_fd = shm_open(shm_name_fluidvar.data(), O_CREAT | O_RDWR, 0666);
      if (shm_fd == -1) {
      std::cerr << "Failed to create shared memory!" << std::endl;
      return EXIT_FAILURE;
   }
   ftruncate(shm_fd, fluid_data_size);
   float* shm_h_fluidvar = (float*)mmap(0, fluid_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
   if (shm_h_fluidvar == MAP_FAILED) {
      std::cerr << "mmap failed!" << std::endl;
      return EXIT_FAILURE;
   }

   std::cout << "Transferring device data to host" << std::endl;
   cudaMemcpy(shm_h_fluidvar, fluidvars, fluid_data_size, cudaMemcpyDeviceToHost);
   checkCuda(cudaDeviceSynchronize());

   // 4 is the magic number b/c 8 fluid variables need to be written out in parallel, and not every machine has 8 cores
   std::string filename_fluidvar = path_to_data + "fluidvar0.h5";

   std::cout << "Writing Screw-Pinch ICs out with PHDF5" << std::endl;
   int ret = callBinary_PHDF5Write(filename_fluidvar, Nx, Ny, Nz, shm_name_fluidvar, fluid_data_size, std::to_string(4), phdf5_bin_name); 
   if (ret != 0) {
        std::cerr << "Error executing PHDF5 command" << std::endl;
   }

   std::cout << "Writing attributes to the dataset with HDF5" << std::endl;
   ret = callBinary_Attributes(filename_fluidvar, Nx, Ny, Nz, attr_bin_name); // inadvisable to write attributes in a PHDF5 context
   if (ret != 0) {
        std::cerr << "Error executing attribute command" << std::endl;
   }

   /* COMPUTE STABILITY CRITERION */
   // First, transfer grid data
   std::string shm_name_gridx = "/shared_h_gridx";
   shm_fd = shm_open(shm_name_gridx.data(), O_CREAT | O_RDWR, 0666);
   ftruncate(shm_fd, sizeof(float) * Nx);
   float* shm_h_gridx = (float*)mmap(0, sizeof(float) * Nx, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
   if (shm_h_gridx == MAP_FAILED) {
      std::cerr << "mmap failed for grid_x!" << std::endl;
      return EXIT_FAILURE;
   }
   cudaMemcpy(shm_h_gridx, x_grid, sizeof(float) * Nx, cudaMemcpyDeviceToHost);

   std::string shm_name_gridy = "/shared_h_gridy";
   shm_fd = shm_open(shm_name_gridy.data(), O_CREAT | O_RDWR, 0666);
   ftruncate(shm_fd, sizeof(float) * Ny);
   float* shm_h_gridy = (float*)mmap(0, sizeof(float) * Ny, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
   if (shm_h_gridy == MAP_FAILED) {
      std::cerr << "mmap failed for grid_y!" << std::endl;
      return EXIT_FAILURE;
   }
   cudaMemcpy(shm_h_gridy, y_grid, sizeof(float) * Ny, cudaMemcpyDeviceToHost);

   std::string shm_name_gridz = "/shared_h_gridz";
   shm_fd = shm_open(shm_name_gridz.data(), O_CREAT | O_RDWR, 0666);
   ftruncate(shm_fd, sizeof(float) * Nz);
   float* shm_h_gridz = (float*)mmap(0, sizeof(float) * Nz, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
   if (shm_h_gridz == MAP_FAILED) {
      std::cerr << "mmap failed for grid_z!" << std::endl;
      return EXIT_FAILURE;
   }
   cudaMemcpy(shm_h_gridz, z_grid, sizeof(float) * Nz, cudaMemcpyDeviceToHost);
   checkCuda(cudaDeviceSynchronize());

   std::cout << "Forking to process for computing stability" << std::endl;
   ret = callBinary_EigenSC(shm_name_fluidvar, Nx, Ny, Nz, eigen_bin_name, dt, dx, dy, dz, shm_name_gridx, shm_name_gridy, shm_name_gridz);
   if (ret != 0) {
        std::cerr << "Error executing Eigen command" << std::endl;
   }


   /* SIMULATION LOOP */
   for (int it = 0; it < Nt; it++){
      std::cout << "Starting timestep " << it << std::endl;
      /* UPDATE fluidvars_np1 USING fluidvars */
      /* COPY fluidvars TO HOST */
      checkCuda(cudaDeviceSynchronize());

      /* COMPUTE fluidvars_int */
      /* WRITE fluidvars_np1 INTO fluidvars */
      /* WRITE .h5 file TO STORAGE USING fluidvars_np1 */   
      checkCuda(cudaDeviceSynchronize());
      
      std::cout << "Timestep " << it << " complete" << std::endl;
   }

   /* FREE EVERYTHING */
   // Device
   checkCuda(cudaFree(fluidvars));
   checkCuda(cudaFree(fluidvars_np1));
   checkCuda(cudaFree(fluidvars_int));
   checkCuda(cudaFree(x_grid));
   checkCuda(cudaFree(y_grid));
   checkCuda(cudaFree(z_grid));
   
   // Host
   munmap(shm_h_fluidvar, 8 * fluid_data_size);
   munmap(shm_h_gridx, sizeof(float) * Nx);
   munmap(shm_h_gridy, sizeof(float) * Ny);
   munmap(shm_h_gridz, sizeof(float) * Nz);
   shm_unlink(shm_name_fluidvar.data());
   shm_unlink(shm_name_gridx.data());
   shm_unlink(shm_name_gridy.data());
   shm_unlink(shm_name_gridz.data());
   return 0;
}

/* */