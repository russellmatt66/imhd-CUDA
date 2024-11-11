#include <iostream>
#include <cstdlib>
#include <cstring>
#include <assert.h>
#include <stdio.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include "initialize_od.cuh"
#include "kernels_od.cuh"
#include "kernels_od_intvar.cuh"

#include "utils.cuh"
#include "utils.hpp"

int main(int argc, char* argv[]){
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
   std::string write_grid_bin_name = argv[17];
   std::string eigen_bin_name = argv[18];
   std::string num_proc = argv[19];

   int meshblockdims_xthreads = atoi(argv[20]);
   int meshblockdims_ythreads = atoi(argv[21]);
   int meshblockdims_zthreads = atoi(argv[22]);

   int initblockdims_xthreads = atoi(argv[23]);
   int initblockdims_ythreads = atoi(argv[24]);
   int initblockdims_zthreads = atoi(argv[25]);
   
   int intvarblockdims_xthreads = atoi(argv[26]);
   int intvarblockdims_ythreads = atoi(argv[27]);
   int intvarblockdims_zthreads = atoi(argv[28]);

   int fluidblockdims_xthreads = atoi(argv[29]);
   int fluidblockdims_ythreads = atoi(argv[30]);
   int fluidblockdims_zthreads = atoi(argv[31]);

   // CUDA BOILERPLATE 
   int deviceId;
   int numberOfSMs;

   cudaGetDevice(&deviceId); // number of blocks should be a multiple of the number of device SMs
   cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

   // SPECIFY EXECUTION CONFIGURATIONS
   dim3 exec_grid_dims(numberOfSMs, numberOfSMs, numberOfSMs);
   dim3 mesh_block_dims(meshblockdims_xthreads, meshblockdims_ythreads, meshblockdims_zthreads);
   dim3 init_block_dims(initblockdims_xthreads, initblockdims_ythreads, initblockdims_zthreads);
   dim3 intvar_block_dims(intvarblockdims_xthreads, intvarblockdims_ythreads, intvarblockdims_zthreads);
   dim3 fluid_block_dims(fluidblockdims_xthreads, fluidblockdims_ythreads, fluidblockdims_zthreads);

   // INITIALIZE SIMULATION DATA
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

   std::string filename_fluidvar = path_to_data + "fluidvars_0.h5";

   std::cout << "Writing Screw-Pinch ICs out with PHDF5" << std::endl;
   int ret = callBinary_PHDF5Write(filename_fluidvar, Nx, Ny, Nz, shm_name_fluidvar, fluid_data_size, num_proc, phdf5_bin_name); 
   if (ret != 0) {
        std::cerr << "Error executing PHDF5 command" << std::endl;
   }

   std::cout << "Writing attributes to the dataset with HDF5" << std::endl;
   ret = callBinary_AttrWrite(filename_fluidvar, Nx, Ny, Nz, attr_bin_name); // inadvisable to write attributes in a PHDF5 context
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

   std::cout << "Forking to process for writing grid to storage" << std::endl;
   ret = callBinary_WriteGrid(write_grid_bin_name, path_to_data, shm_name_gridx, shm_name_gridy, shm_name_gridz, Nx, Ny, Nz);
   if (ret != 0) {
         std::cerr << "Error executing writegrid binary: " << eigen_bin_name << std::endl;
   }

   if (!(eigen_bin_name == "none")){ // Don't always want to check stability - expensive raster scan
      std::cout << "Forking to process for checking stability" << std::endl;
      ret = callBinary_EigenSC(shm_name_fluidvar, Nx, Ny, Nz, eigen_bin_name, dt, dx, dy, dz, shm_name_gridx, shm_name_gridy, shm_name_gridz);
      if (ret != 0) {
         std::cerr << "Error executing Eigen binary: " << eigen_bin_name << std::endl;
      }
   }

   /* SIMULATION LOOP */
   for (int it = 1; it < Nt; it++){
      std::cout << "Starting timestep " << it << std::endl;

      std::cout << "Launching kernels for computing fluid variables on interior and boundary" << std::endl;
      FluidAdvance<<<exec_grid_dims, fluid_block_dims>>>(fluidvars_np1, fluidvars, fluidvars_int, D, dt, dx, dy, dz, Nx, Ny, Nz);
      BoundaryConditions<<<exec_grid_dims, fluid_block_dims>>>(fluidvars_np1, fluidvars, fluidvars_int, D, dt, dx, dy, dz, Nx, Ny, Nz);
      checkCuda(cudaDeviceSynchronize());
      
      std::cout << "Kernels for computing fluid variables completed" << std::endl;
      std::cout << "Launching kernels for computing intermediate variables on interior and boundary" << std::endl; 
      ComputeIntermediateVariables<<<exec_grid_dims, intvar_block_dims>>>(fluidvars, fluidvars_int, dt, dx, dy, dz, Nx, Ny, Nz); // Compute fluidvars_int
      ComputeIntermediateVariablesBoundary<<<exec_grid_dims, intvar_block_dims>>>(fluidvars, fluidvars_int, dt, dx, dy, dz, Nx, Ny, Nz);

      std::cout << "Launching kernel for writing updated fluid data into current timestep fluid data" << std::endl;
      SwapSimData<<<exec_grid_dims, intvar_block_dims>>>(fluidvars, fluidvars_np1, Nx, Ny, Nz); // Write fluidvars_np1 into fluidvars

      std::cout << "Transferring updated fluid data to host" << std::endl;
      cudaMemcpy(shm_h_fluidvar, fluidvars_np1, fluid_data_size, cudaMemcpyDeviceToHost);
      checkCuda(cudaDeviceSynchronize());

      /* WRITE .h5 file TO STORAGE USING fluidvars_np1 */        
      std::cout << "Kernels for intermediate variables, buffer write, and D2H transfer complete" << std::endl; 
      std::cout << "Writing updated fluid data out" << std::endl;
      filename_fluidvar = path_to_data + "fluidvars_" + std::to_string(it) + ".h5";
      ret = callBinary_PHDF5Write(filename_fluidvar, Nx, Ny, Nz, shm_name_fluidvar, fluid_data_size, num_proc, phdf5_bin_name);
      if (ret != 0) {
         std::cerr << "Error forking PHDF5Write binary on timestep " << std::to_string(it) << std::endl;
      }  

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