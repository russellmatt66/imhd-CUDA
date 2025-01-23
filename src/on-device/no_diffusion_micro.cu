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
	float r_max_coeff = atof(argv[6]); // r_pinch = r_max_coeff * r_max
	
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

   int xgrid_threads = atoi(argv[20]);
   int ygrid_threads = atoi(argv[21]);
   int zgrid_threads = atoi(argv[22]);

   int init_xthreads = atoi(argv[23]);
   int init_ythreads = atoi(argv[24]);
   int init_zthreads = atoi(argv[25]);
   
   int FA_xthreads = atoi(argv[26]);
   int FA_ythreads = atoi(argv[27]);
   int FA_zthreads = atoi(argv[28]);

   int SM_mult_grid_x = atoi(argv[29]);
   int SM_mult_grid_y = atoi(argv[30]);
   int SM_mult_grid_z = atoi(argv[31]);

	int SM_mult_init_x = atoi(argv[32]);
	int SM_mult_init_y = atoi(argv[33]);
	int SM_mult_init_z = atoi(argv[34]);

   int SM_mult_FA_x = atoi(argv[35]);
   int SM_mult_FA_y = atoi(argv[36]);
   int SM_mult_FA_z = atoi(argv[37]);

   // CUDA BOILERPLATE 
   int deviceId;
   int numberOfSMs;

   cudaGetDevice(&deviceId); // number of blocks should be a multiple of the number of device SMs
   cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

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

   // Execution grid and threadblock configurations for the Grid Initialization microkernels
   dim3 egd_xgrid(SM_mult_grid_x * numberOfSMs, 1, 1); // "egd" = "execution_grid_dimensions"
   dim3 egd_ygrid(1, SM_mult_grid_y * numberOfSMs, 1);
   dim3 egd_zgrid(1, 1, SM_mult_grid_z * numberOfSMs);

   dim3 tbd_xgrid(xgrid_threads, 1, 1); // "tbd = thread_block_dimensions"
   dim3 tbd_ygrid(1, ygrid_threads, 1);
   dim3 tbd_zgrid(1, 1, zgrid_threads);
   
   // Execution grid and threadblock configurations for the 
   dim3 egd_init(SM_mult_init_x * numberOfSMs, SM_mult_init_y * numberOfSMs, SM_mult_init_z * numberOfSMs);
   dim3 tbd_init(init_xthreads, init_ythreads, init_zthreads); 

   // TODO: Microkernels should be in a wrapper   
   // InitializeGrid<<<exec_grid_dims_grid, mesh_block_dims>>>(x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz, x_grid, y_grid, z_grid, Nx, Ny, Nz);
   InitializeX<<<egd_xgrid, tbd_xgrid>>>(x_grid, x_min, dx, Nx);
   InitializeY<<<egd_ygrid, tbd_ygrid>>>(y_grid, y_min, dy, Ny);
   InitializeZ<<<egd_zgrid, tbd_zgrid>>>(z_grid, z_min, dz, Nz);
   ZeroVars<<<egd_init, tbd_init>>>(fluidvars, Nx, Ny, Nz); 
   checkCuda(cudaDeviceSynchronize());

   ScrewPinch<<<egd_init, tbd_init>>>(fluidvars, J0, r_max_coeff, x_grid, y_grid, z_grid, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());

   // Execution grid and threadblock configurations for the Boundary Condition microkernels
   dim3 egd_bdry_leftright(numberOfSMs, 1, numberOfSMs);
   dim3 egd_bdry_topbottom(1, numberOfSMs, numberOfSMs);
   dim3 egd_bdry_frontback(numberOfSMs, numberOfSMs, 1);

   dim3 tbd_bdry_leftright(8, 1, 8);
   dim3 tbd_bdry_topbottom(1, 8, 8);
   dim3 tbd_bdry_frontback(8, 8, 1); // can also be used for PBCs
   dim3 tbd_qintbdry_frontright(1024, 1, 1);
   dim3 tbd_qintbdry_frontbottom(1, 1024, 1);
   dim3 tbd_qintbdry_bottomright(1, 1, 1024);

   rigidConductingWallBCsLeftRight<<<egd_bdry_leftright, tbd_bdry_leftright>>>(fluidvars, Nx, Ny, Nz);
   rigidConductingWallBCsTopBottom<<<egd_bdry_topbottom, tbd_bdry_topbottom>>>(fluidvars, Nx, Ny, Nz);
   PBCs<<<egd_bdry_frontback, tbd_bdry_frontback>>>(fluidvars, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());

   // Execution grid and threadblock configurations for the Predictor and Corrector microkernels
   dim3 egd_fluidadvance(SM_mult_FA_x * numberOfSMs, SM_mult_FA_y * numberOfSMs, SM_mult_FA_z * numberOfSMs);
   
   dim3 tbd_fluidadvance(FA_xthreads, FA_ythreads, FA_zthreads); 

   // ComputeIntermediateVariablesNoDiff<<<exec_grid_dims_intvar, intvar_block_dims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   ComputeIntRhoMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   ComputeIntRhoVXMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   ComputeIntRhoVYMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   ComputeIntRhoVZMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   ComputeIntBXMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   ComputeIntBYMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   ComputeIntBZMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   ComputeIntEMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());    
   
   // Execution grid specification for the Qint BCs - they require specifying values along certain lines
   dim3 egd_qintbdry_frontright(numberOfSMs, 1, 1); 
   dim3 egd_qintbdry_frontbottom(1, numberOfSMs, 1);
   dim3 egd_qintbdry_bottomright(1, 1, numberOfSMs);

   QintBdryFrontNoDiff<<<egd_bdry_frontback, tbd_bdry_frontback>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   QintBdryLeftRightNoDiff<<<egd_bdry_leftright, tbd_bdry_leftright>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   QintBdryTopBottomNoDiff<<<egd_bdry_topbottom, tbd_bdry_topbottom>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   QintBdryFrontBottomNoDiff<<<egd_qintbdry_frontbottom, tbd_qintbdry_frontbottom>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   QintBdryFrontRightNoDiff<<<egd_qintbdry_frontright, tbd_qintbdry_frontright>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   QintBdryBottomRightNoDiff<<<egd_qintbdry_bottomright, tbd_qintbdry_bottomright>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());    

   QintBdryPBCs<<<egd_bdry_frontback, tbd_bdry_frontback>>>(fluidvars, intvars, Nx, Ny, Nz);
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

      std::cout << "Launching microkernels for computing fluid variables" << std::endl;
      // FluidAdvanceLocalNoDiff<<<exec_grid_dims, fluid_block_dims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      FluidAdvanceMicroRhoLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      FluidAdvanceMicroRhoVXLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      FluidAdvanceMicroRhoVYLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      FluidAdvanceMicroRhoVZLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      FluidAdvanceMicroBXLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      FluidAdvanceMicroBYLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      FluidAdvanceMicroBZLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      FluidAdvanceMicroELocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      checkCuda(cudaDeviceSynchronize());
      
      std::cout << "Launching microkernels for computing the fluid boundaries" << std::endl; 
      // BoundaryConditions<<<exec_grid_dims, fluid_block_dims>>>(fluidvars, intvars, D, dt, dx, dy, dz, Nx, Ny, Nz);
      PBCs<<<egd_bdry_frontback, tbd_bdry_frontback>>>(fluidvars, Nx, Ny, Nz);
      checkCuda(cudaDeviceSynchronize());
      std::cout << "Kernels for computing fluid variables completed" << std::endl;
      
      std::cout << "Launching microkernels for computing intermediate variables" << std::endl; 
      // ComputeIntermediateVariablesNoDiff<<<exec_grid_dims_intvar, intvar_block_dims>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      ComputeIntRhoMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      ComputeIntRhoVXMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      ComputeIntRhoVYMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      ComputeIntRhoVZMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      ComputeIntBXMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      ComputeIntBYMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      ComputeIntBZMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      ComputeIntEMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      checkCuda(cudaDeviceSynchronize());

      std::cout << "Launching microkernels for computing Qint boundaries" << std::endl; 
      QintBdryFrontNoDiff<<<egd_bdry_frontback, tbd_bdry_frontback>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      QintBdryLeftRightNoDiff<<<egd_bdry_leftright, tbd_bdry_leftright>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      QintBdryTopBottomNoDiff<<<egd_bdry_topbottom, tbd_bdry_topbottom>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      QintBdryFrontBottomNoDiff<<<egd_qintbdry_frontbottom, tbd_qintbdry_frontbottom>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      QintBdryFrontRightNoDiff<<<egd_qintbdry_frontright, tbd_qintbdry_frontright>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      QintBdryBottomRightNoDiff<<<egd_qintbdry_bottomright, tbd_qintbdry_bottomright>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      checkCuda(cudaDeviceSynchronize());    

      std::cout << "Launching kernel for computing Qint PBCs" << std::endl; 
      QintBdryPBCs<<<egd_bdry_frontback, tbd_bdry_frontback>>>(fluidvars, intvars, Nx, Ny, Nz);
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