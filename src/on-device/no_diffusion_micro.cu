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
#include "kernels_fluidbcs.cuh"
#include "kernels_od_intvar.cuh"
#include "kernels_intvarbcs.cuh"

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

   int SM_mult_x_grid = atoi(argv[32]);
   int SM_mult_y_grid = atoi(argv[33]);
   int SM_mult_z_grid = atoi(argv[34]);

	int SM_mult_x_intvar = atoi(argv[35]);
	int SM_mult_y_intvar = atoi(argv[36]);
	int SM_mult_z_intvar = atoi(argv[37]);

   // CUDA BOILERPLATE 
   int deviceId;
   int numberOfSMs;

   cudaGetDevice(&deviceId); // number of blocks should be a multiple of the number of device SMs
   cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

   // Execution grid configurations
   dim3 exec_grid_dims(numberOfSMs, numberOfSMs, numberOfSMs);
   dim3 exec_grid_dims_grid(SM_mult_x_grid * numberOfSMs, SM_mult_y_grid * numberOfSMs, SM_mult_z_grid * numberOfSMs);
   dim3 exec_grid_dims_intvar(SM_mult_x_intvar * numberOfSMs, SM_mult_y_intvar * numberOfSMs, SM_mult_z_intvar * numberOfSMs);

   // Execution-grid configurations for the FluidAdvance microkernels
   dim3 exec_grid_dims_fluidadvance(numberOfSMs, numberOfSMs, numberOfSMs);

   // Execution-grid configurations for the Boundary Condition Microkernels
   dim3 exec_grid_dims_bdry_frontback(numberOfSMs, numberOfSMs, 1); // can also be used for PBCs and 
   dim3 exec_grid_dims_bdry_leftright(numberOfSMs, 1, numberOfSMs);
   dim3 exec_grid_dims_bdry_topbottom(1, numberOfSMs, numberOfSMs);

   // Qint BCs require specifying values along certain lines
   dim3 exec_grid_dims_qintbdry_frontright(numberOfSMs, 1, 1);
   dim3 exec_grid_dims_qintbdry_frontbottom(1, numberOfSMs, 1);
   dim3 exec_grid_dims_qintbdry_bottomright(1, 1, numberOfSMs);

   // Threadblock execution configurations for the megakernels
   dim3 mesh_block_dims(meshblockdims_xthreads, meshblockdims_ythreads, meshblockdims_zthreads);
   dim3 init_block_dims(initblockdims_xthreads, initblockdims_ythreads, initblockdims_zthreads);
   dim3 intvar_block_dims(intvarblockdims_xthreads, intvarblockdims_ythreads, intvarblockdims_zthreads);
   dim3 fluid_block_dims(fluidblockdims_xthreads, fluidblockdims_ythreads, fluidblockdims_zthreads);

   // Threadblock execution configurations for the Fluid Advance Microkernels
   dim3 fluidadvance_blockdims(8, 8, 8); // power of two

   // Threadblock execution configurations for the BoundaryCondition Microkernels
   /* 
   Refactor these so they are input
   */
   dim3 bdry_frontback_blockdims(8, 8, 1); // can also be used for PBCs
   dim3 bdry_leftright_blockdims(8, 1, 8);
   dim3 bdry_topbottom_blockdims(1, 8, 8);
   dim3 qintbdry_frontright_blockdims(1024, 1, 1);
   dim3 qintbdry_frontbottom_blockdims(1, 1024, 1);
   dim3 qintbdry_bottomright_blockdims(1, 1, 1024);

   size_t cube_size = Nx * Ny * Nz;
   size_t fluidvar_size = sizeof(float) * cube_size;
   size_t fluid_data_size = 8 * fluidvar_size;

   float *fluidvars, *intvars;

   checkCuda(cudaMalloc(&fluidvars, fluid_data_size));
   checkCuda(cudaMalloc(&intvars, fluid_data_size));
   
   float *x_grid, *y_grid, *z_grid;

   checkCuda(cudaMalloc(&x_grid, sizeof(float) * Nx));
   checkCuda(cudaMalloc(&y_grid, sizeof(float) * Ny));
   checkCuda(cudaMalloc(&z_grid, sizeof(float) * Nz));

   float dx = (x_max - x_min) / (Nx - 1);
	float dy = (y_max - y_min) / (Ny - 1);
	float dz = (z_max - z_min) / (Nz - 1);

   InitializeGrid<<<exec_grid_dims_grid, mesh_block_dims>>>(x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz, x_grid, y_grid, z_grid, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());

   ScrewPinch<<<exec_grid_dims, init_block_dims>>>(fluidvars, J0, x_grid, y_grid, z_grid, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());

   rigidConductingWallBCsLeftRight<<<exec_grid_dims_bdry_leftright, bdry_leftright_blockdims>>>(fluidvars, Nx, Ny, Nz);
   rigidConductingWallBCsTopBottom<<<exec_grid_dims_bdry_topbottom, bdry_topbottom_blockdims>>>(fluidvars, Nx, Ny, Nz);
   PBCs<<<exec_grid_dims_bdry_frontback, bdry_frontback_blockdims>>>(fluidvars, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());

   ComputeIntermediateVariablesNoDiff<<<exec_grid_dims_intvar, intvar_block_dims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());    
   
   // Maybe this should be in a wrapper
   QintBdryFrontNoDiff<<<exec_grid_dims_bdry_frontback, bdry_frontback_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   QintBdryLeftRightNoDiff<<<exec_grid_dims_bdry_leftright, bdry_leftright_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   QintBdryTopBottomNoDiff<<<exec_grid_dims_bdry_topbottom, bdry_topbottom_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   QintBdryFrontBottomNoDiff<<<exec_grid_dims_qintbdry_frontbottom, qintbdry_frontbottom_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   QintBdryFrontRightNoDiff<<<exec_grid_dims_qintbdry_frontright, qintbdry_frontright_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   QintBdryBottomRightNoDiff<<<exec_grid_dims_qintbdry_bottomright, qintbdry_bottomright_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());    

   QintBdryPBCs<<<exec_grid_dims_bdry_frontback, bdry_frontback_blockdims>>>(fluidvars, intvars, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());    

   // Use IPC to write data out in order to avoid redundant work 
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

   // COMPUTE STABILITY CRITERION
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
      std::cout << "Forking to process for computing CFL number (checking stability)" << std::endl;
      ret = callBinary_EigenSC(shm_name_fluidvar, Nx, Ny, Nz, eigen_bin_name, dt, dx, dy, dz, shm_name_gridx, shm_name_gridy, shm_name_gridz);
      if (ret != 0) {
         std::cerr << "Error executing Eigen binary: " << eigen_bin_name << std::endl;
         std::cerr << "Error code: " << ret << std::endl;
      }
   }

   // SIMULATION LOOP
   for (int it = 1; it < Nt; it++){
      std::cout << "Starting timestep " << it << std::endl;

      std::cout << "Launching Microkernels for computing fluid variables" << std::endl;
      // FluidAdvanceLocalNoDiff<<<exec_grid_dims, fluid_block_dims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      FluidAdvanceMicroRhoLocalNoDiff<<<exec_grid_dims_fluidadvance, fluidadvance_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      FluidAdvanceMicroRhoVXLocalNoDiff<<<exec_grid_dims_fluidadvance, fluidadvance_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      FluidAdvanceMicroRhoVYLocalNoDiff<<<exec_grid_dims_fluidadvance, fluidadvance_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      FluidAdvanceMicroRhoVZLocalNoDiff<<<exec_grid_dims_fluidadvance, fluidadvance_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      FluidAdvanceMicroBXLocalNoDiff<<<exec_grid_dims_fluidadvance, fluidadvance_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      FluidAdvanceMicroBYLocalNoDiff<<<exec_grid_dims_fluidadvance, fluidadvance_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      FluidAdvanceMicroBZLocalNoDiff<<<exec_grid_dims_fluidadvance, fluidadvance_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      FluidAdvanceMicroELocalNoDiff<<<exec_grid_dims_fluidadvance, fluidadvance_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      checkCuda(cudaDeviceSynchronize());
      
      std::cout << "Launching microkernels for computing the fluid boundaries" << std::endl; 
      // BoundaryConditions<<<exec_grid_dims, fluid_block_dims>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
      PBCs<<<exec_grid_dims_bdry_frontback, bdry_frontback_blockdims>>>(fluidvars, Nx, Ny, Nz);
      checkCuda(cudaDeviceSynchronize());
      std::cout << "Kernels for computing fluid variables completed" << std::endl;
      
      std::cout << "Launching kernel for computing intermediate variables" << std::endl; 
      ComputeIntermediateVariablesNoDiff<<<exec_grid_dims_intvar, intvar_block_dims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      checkCuda(cudaDeviceSynchronize());

      std::cout << "Launching microkernels for computing Qint boundaries" << std::endl; 
      QintBdryFrontNoDiff<<<exec_grid_dims_bdry_frontback, bdry_frontback_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      QintBdryLeftRightNoDiff<<<exec_grid_dims_bdry_leftright, bdry_leftright_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      QintBdryTopBottomNoDiff<<<exec_grid_dims_bdry_topbottom, bdry_topbottom_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      QintBdryFrontBottomNoDiff<<<exec_grid_dims_qintbdry_frontbottom, qintbdry_frontbottom_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      QintBdryFrontRightNoDiff<<<exec_grid_dims_qintbdry_frontright, qintbdry_frontright_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      QintBdryBottomRightNoDiff<<<exec_grid_dims_qintbdry_bottomright, qintbdry_bottomright_blockdims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      checkCuda(cudaDeviceSynchronize());    

      std::cout << "Launching kernel for computing Qint PBCs" << std::endl; 
      QintBdryPBCs<<<exec_grid_dims_bdry_frontback, bdry_frontback_blockdims>>>(fluidvars, intvars, Nx, Ny, Nz);
      checkCuda(cudaDeviceSynchronize());    

      std::cout << "Transferring updated fluid data to host" << std::endl;
      cudaMemcpy(shm_h_fluidvar, fluidvars, fluid_data_size, cudaMemcpyDeviceToHost);
      checkCuda(cudaDeviceSynchronize());
      std::cout << "Kernels for computing intermediate variables completed" << std::endl;
      std::cout << "Fluid D2H data migration completed" << std::endl;

      std::cout << "Writing updated fluid data out" << std::endl;
      filename_fluidvar = path_to_data + "fluidvars_" + std::to_string(it) + ".h5";
      ret = callBinary_PHDF5Write(filename_fluidvar, Nx, Ny, Nz, shm_name_fluidvar, fluid_data_size, num_proc, phdf5_bin_name);
      if (ret != 0) {
         std::cerr << "Error forking PHDF5Write binary on timestep " << std::to_string(it) << std::endl;
      }  

      std::cout << "Timestep " << it << " complete" << std::endl;

      if (!(eigen_bin_name == "none")){ // Don't always want to check stability - expensive raster scan
         std::cout << "Forking to process for computing CFL number (checking stability)" << std::endl;
         ret = callBinary_EigenSC(shm_name_fluidvar, Nx, Ny, Nz, eigen_bin_name, dt, dx, dy, dz, shm_name_gridx, shm_name_gridy, shm_name_gridz);
         if (ret != 0) {
            std::cerr << "Error executing Eigen binary: " << eigen_bin_name << std::endl;
            std::cerr << "Error code: " << ret << std::endl;
         }
      }  
   } 

   // FREE EVERYTHING
   // Device
   checkCuda(cudaFree(fluidvars));
   checkCuda(cudaFree(intvars));
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