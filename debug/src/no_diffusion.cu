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
   std::vector<float> inputs (41, 0.0);
   parseInputFileDebug(inputs, "./debug.inp"); 
	
	for (int i = 0; i < inputs.size(); i++){
		std::cout << "inputs[" << i << "] = " << inputs[i] << std::endl; 
	}

   int Nt = int(inputs[0]);
   int Nx = int(inputs[1]);
   int Ny = int(inputs[2]);
   int Nz = int(inputs[3]);

   float J0 = inputs[4]; // amplitude of the current
   float D = inputs[5]; // diffusion
   float r_max_coeff = inputs[6]; // r_pinch = r_max_coeff * r_max
   
   float x_min = inputs[7];
   float x_max = inputs[8];
   float y_min = inputs[9];
   float y_max = inputs[10];
   float z_min = inputs[11];
   float z_max = inputs[12];
   float dt = inputs[13];

   int xgrid_threads = int(inputs[14]);
   int ygrid_threads = int(inputs[15]);
   int zgrid_threads = int(inputs[16]);

   int init_xthreads = int(inputs[17]);
   int init_ythreads = int(inputs[18]);
   int init_zthreads = int(inputs[19]);
   
   int FA_xthreads = int(inputs[20]);
   int FA_ythreads = int(inputs[21]);
   int FA_zthreads = int(inputs[22]);

   int BCLeftRight_xthreads = int(inputs[23]);
   int BCLeftRight_zthreads = int(inputs[24]);

   int BCTopBottom_ythreads = int(inputs[25]);
   int BCTopBottom_zthreads = int(inputs[26]);

   int PBC_xthreads = int(inputs[27]);
   int PBC_ythreads = int(inputs[28]);

   int QintBC_FrontRight_xthreads = int(inputs[29]);
   int QintBC_FrontBottom_ythreads = int(inputs[30]);
   int QintBC_BottomRight_zthreads = int(inputs[31]);

   int SM_mult_grid_x = int(inputs[32]);
   int SM_mult_grid_y = int(inputs[33]);
   int SM_mult_grid_z = int(inputs[34]);

   int SM_mult_init_x = int(inputs[35]);
   int SM_mult_init_y = int(inputs[36]);
   int SM_mult_init_z = int(inputs[37]);

   int SM_mult_FA_x = int(inputs[38]);
   int SM_mult_FA_y = int(inputs[39]);
   int SM_mult_FA_z = int(inputs[40]);

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
   
   // Execution grid and threadblock configurations for the Grid Initialization kernels
   // dim3 egd_grid(SM_mult_grid_x * numberOfSMs, SM_mult_grid_y * numberOfSMs, SM_mult_grid_z * numberOfSMs);
   // std::cout << "egd_grid is: (" << egd_grid.x << "," << egd_grid.y << "," << egd_grid.z << ")" << std::endl;

   // dim3 tbd_grid(xgrid_threads, ygrid_threads, zgrid_threads);
   // std::cout << "tbd_grid is: (" << tbd_grid.x << "," << tbd_grid.y << "," << tbd_grid.z << ")" << std::endl;

   // Execution grid and threadblock configurations for the initialization kernels
   dim3 egd_init(SM_mult_init_x * numberOfSMs, SM_mult_init_y * numberOfSMs, SM_mult_init_z * numberOfSMs);
   std::cout << "egd_init is: (" << egd_init.x << "," << egd_init.y << "," << egd_init.z << ")" << std::endl;

   dim3 tbd_init(init_xthreads, init_ythreads, init_zthreads); 
   std::cout << "tbd_init is: (" << tbd_init.x << "," << tbd_init.y << "," << tbd_init.z << ")" << std::endl;

   // Execution grid and threadblock configurations for the Boundary Condition microkernels
   dim3 egd_bdry_leftright(numberOfSMs, 1, numberOfSMs);
   dim3 egd_bdry_topbottom(1, numberOfSMs, numberOfSMs);
   dim3 egd_bdry_frontback(numberOfSMs, numberOfSMs, 1);

   // Execution grid specification for the Qint BCs - they require specifying values along certain lines
   dim3 egd_qintbdry_frontright(numberOfSMs, 1, 1); 
   dim3 egd_qintbdry_frontbottom(1, numberOfSMs, 1);
   dim3 egd_qintbdry_bottomright(1, 1, numberOfSMs);
   
   dim3 tbd_bdry_leftright(BCLeftRight_xthreads, 1, BCLeftRight_zthreads);
   dim3 tbd_bdry_topbottom(1, BCTopBottom_ythreads, BCTopBottom_zthreads);
   dim3 tbd_bdry_frontback(PBC_xthreads, PBC_ythreads, 1); // can also be used for PBCs
   dim3 tbd_qintbdry_frontright(QintBC_FrontRight_xthreads, 1, 1);
   dim3 tbd_qintbdry_frontbottom(1, QintBC_FrontBottom_ythreads, 1);
   dim3 tbd_qintbdry_bottomright(1, 1, QintBC_BottomRight_zthreads);

   // Execution grid and threadblock configurations for the Predictor and Corrector microkernels
   dim3 egd_fluidadvance(SM_mult_FA_x * numberOfSMs, SM_mult_FA_y * numberOfSMs, SM_mult_FA_z * numberOfSMs);
   std::cout << "egd_fluidadvance is: (" << egd_fluidadvance.x << "," << egd_fluidadvance.y << "," << egd_fluidadvance.z << ")" << std::endl;

   dim3 tbd_fluidadvance(FA_xthreads, FA_ythreads, FA_zthreads); 
   std::cout << "tbd_fluidadvance is: (" << tbd_fluidadvance.x << "," << tbd_fluidadvance.y << "," << tbd_fluidadvance.z << ")" << std::endl;

   // dim3 tbd_fluidadvance(6, 6, 6); 

   // TODO: Microkernels should be in a wrapper   
   // InitializeGrid<<<egd_grid, tbd_grid>>>(x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz, x_grid, y_grid, z_grid, Nx, Ny, Nz);
   InitializeX<<<egd_xgrid, tbd_xgrid>>>(x_grid, x_min, dx, Nx);
   InitializeY<<<egd_ygrid, tbd_ygrid>>>(y_grid, y_min, dy, Ny);
   InitializeZ<<<egd_zgrid, tbd_zgrid>>>(z_grid, z_min, dz, Nz);
   // ZeroVars<<<egd_init, tbd_init>>>(fluidvars, Nx, Ny, Nz); 
   // ZeroVars<<<egd_init, tbd_init>>>(intvars, Nx, Ny, Nz); 
   checkCuda(cudaDeviceSynchronize());

   /*
   TODO: "ScrewPinch" doesn't work
   */ 
   // ScrewPinch<<<egd_init, tbd_init>>>(fluidvars, J0, r_max_coeff, x_grid, y_grid, z_grid, Nx, Ny, Nz);
   ScrewPinchStride<<<egd_init, tbd_init>>>(fluidvars, J0, x_grid, y_grid, z_grid, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());

   rigidConductingWallBCsLeftRight<<<egd_bdry_leftright, tbd_bdry_leftright>>>(fluidvars, Nx, Ny, Nz);
   rigidConductingWallBCsTopBottom<<<egd_bdry_topbottom, tbd_bdry_topbottom>>>(fluidvars, Nx, Ny, Nz);
   PBCs<<<egd_bdry_frontback, tbd_bdry_frontback>>>(fluidvars, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());

   /*
   NOTE: 
   If you want to use microkernels here, you have to come up with an execution configuration set, and addtl. synchronization 
   */
   ComputeIntermediateVariablesNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());    

   /*
   NOTE:
   You DEFINITELY want to use microkernels here
   */
   // ComputeIntermediateVariablesBoundaryNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz); 
   QintBdryFrontNoDiff<<<egd_bdry_frontback, tbd_bdry_frontback>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   QintBdryLeftRightNoDiff<<<egd_bdry_leftright, tbd_bdry_leftright>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   QintBdryTopBottomNoDiff<<<egd_bdry_topbottom, tbd_bdry_topbottom>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   QintBdryFrontBottomNoDiff<<<egd_qintbdry_frontbottom, tbd_qintbdry_frontbottom>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   QintBdryFrontRightNoDiff<<<egd_qintbdry_frontright, tbd_qintbdry_frontright>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   QintBdryBottomRightNoDiff<<<egd_qintbdry_bottomright, tbd_qintbdry_bottomright>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());    

   QintBdryPBCs<<<egd_bdry_frontback, tbd_bdry_frontback>>>(fluidvars, intvars, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());    

   // // Use IPC to write data out in order to avoid redundant work 
   // std::string shm_name_fluidvar = "/shared_h_fluidvar";
   // int shm_fd = shm_open(shm_name_fluidvar.data(), O_CREAT | O_RDWR, 0666);
   // if (shm_fd == -1) {
   //    std::cerr << "Failed to create shared memory!" << std::endl;
   //    return EXIT_FAILURE;
   // }
   
   // ftruncate(shm_fd, fluid_data_size);
   // float* shm_h_fluidvar = (float*)mmap(0, fluid_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
   // if (shm_h_fluidvar == MAP_FAILED) {
   //    std::cerr << "mmap failed!" << std::endl;
   //    return EXIT_FAILURE;
   // }

   // std::cout << "Transferring device data to host" << std::endl;
   // cudaMemcpy(shm_h_fluidvar, fluidvars, fluid_data_size, cudaMemcpyDeviceToHost);
   // checkCuda(cudaDeviceSynchronize());

   // std::string filename_fluidvar = path_to_data + "fluidvars_0.h5";

   // std::cout << "Writing Screw-Pinch ICs out with PHDF5" << std::endl;
   // int ret = callBinary_PHDF5Write(filename_fluidvar, Nx, Ny, Nz, shm_name_fluidvar, fluid_data_size, num_proc, phdf5_bin_name); 
   // if (ret != 0) {
   //      std::cerr << "Error executing PHDF5 command" << std::endl;
   // }

   // std::cout << "Writing attributes to the dataset with HDF5" << std::endl;
   // ret = callBinary_AttrWrite(filename_fluidvar, Nx, Ny, Nz, attr_bin_name); // inadvisable to write attributes in a PHDF5 context
   // if (ret != 0) {
   //      std::cerr << "Error executing attribute command" << std::endl;
   // }

   // // COMPUTE STABILITY CRITERION
   // // First, transfer grid data
   // std::string shm_name_gridx = "/shared_h_gridx";
   // shm_fd = shm_open(shm_name_gridx.data(), O_CREAT | O_RDWR, 0666);
   // ftruncate(shm_fd, sizeof(float) * Nx);
   // float* shm_h_gridx = (float*)mmap(0, sizeof(float) * Nx, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
   // if (shm_h_gridx == MAP_FAILED) {
   //    std::cerr << "mmap failed for grid_x!" << std::endl;
   //    return EXIT_FAILURE;
   // }
   // cudaMemcpy(shm_h_gridx, x_grid, sizeof(float) * Nx, cudaMemcpyDeviceToHost);

   // std::string shm_name_gridy = "/shared_h_gridy";
   // shm_fd = shm_open(shm_name_gridy.data(), O_CREAT | O_RDWR, 0666);
   // ftruncate(shm_fd, sizeof(float) * Ny);
   // float* shm_h_gridy = (float*)mmap(0, sizeof(float) * Ny, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
   // if (shm_h_gridy == MAP_FAILED) {
   //    std::cerr << "mmap failed for grid_y!" << std::endl;
   //    return EXIT_FAILURE;
   // }
   // cudaMemcpy(shm_h_gridy, y_grid, sizeof(float) * Ny, cudaMemcpyDeviceToHost);

   // std::string shm_name_gridz = "/shared_h_gridz";
   // shm_fd = shm_open(shm_name_gridz.data(), O_CREAT | O_RDWR, 0666);
   // ftruncate(shm_fd, sizeof(float) * Nz);
   // float* shm_h_gridz = (float*)mmap(0, sizeof(float) * Nz, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
   // if (shm_h_gridz == MAP_FAILED) {
   //    std::cerr << "mmap failed for grid_z!" << std::endl;
   //    return EXIT_FAILURE;
   // }
   // cudaMemcpy(shm_h_gridz, z_grid, sizeof(float) * Nz, cudaMemcpyDeviceToHost);
   // checkCuda(cudaDeviceSynchronize());

   // std::cout << "Forking to process for writing grid to storage" << std::endl;
   // ret = callBinary_WriteGrid(write_grid_bin_name, path_to_data, shm_name_gridx, shm_name_gridy, shm_name_gridz, Nx, Ny, Nz);
   // if (ret != 0) {
   //       std::cerr << "Error executing writegrid binary: " << eigen_bin_name << std::endl;
   // }

   // if (!(eigen_bin_name == "none")){ // Don't always want to check stability - expensive raster scan
   //    std::cout << "Forking to process for computing CFL number (checking stability)" << std::endl;
   //    ret = callBinary_EigenSC(shm_name_fluidvar, Nx, Ny, Nz, eigen_bin_name, dt, dx, dy, dz, shm_name_gridx, shm_name_gridy, shm_name_gridz);
   //    if (ret != 0) {
   //       std::cerr << "Error executing Eigen binary: " << eigen_bin_name << std::endl;
   //       std::cerr << "Error code: " << ret << std::endl;
   //    }
   // }

   // SIMULATION LOOP
   for (int it = 1; it < Nt; it++){
      std::cout << "Starting timestep " << it << std::endl;

      std::cout << "Launching megakernel for computing fluid variables" << std::endl;
      FluidAdvanceLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      checkCuda(cudaDeviceSynchronize());
      
      std::cout << "Launching microkernel for PBCs" << std::endl; 
      PBCs<<<egd_bdry_frontback, tbd_bdry_frontback>>>(fluidvars, Nx, Ny, Nz);
      checkCuda(cudaDeviceSynchronize());
      std::cout << "Kernels for computing fluid variables completed" << std::endl;
      
      std::cout << "Launching megakernel for computing intermediate variables" << std::endl; 
      /* NOTE: Thrashes the cache */
      ComputeIntermediateVariablesNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
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

      // std::cout << "Transferring updated fluid data to host" << std::endl;
      // cudaMemcpy(shm_h_fluidvar, fluidvars, fluid_data_size, cudaMemcpyDeviceToHost);
      // checkCuda(cudaDeviceSynchronize());
      // std::cout << "Kernels for computing intermediate variables completed" << std::endl;
      // std::cout << "Fluid D2H data migration completed" << std::endl;

      // std::cout << "Writing updated fluid data out" << std::endl;
      // filename_fluidvar = path_to_data + "fluidvars_" + std::to_string(it) + ".h5";
      // ret = callBinary_PHDF5Write(filename_fluidvar, Nx, Ny, Nz, shm_name_fluidvar, fluid_data_size, num_proc, phdf5_bin_name);
      // if (ret != 0) {
      //    std::cerr << "Error forking PHDF5Write binary on timestep " << std::to_string(it) << std::endl;
      // }  

      // std::cout << "Timestep " << it << " complete" << std::endl;

      // if (!(eigen_bin_name == "none")){ // Don't always want to check stability - expensive raster scan
      //    std::cout << "Forking to process for computing CFL number (checking stability)" << std::endl;
      //    ret = callBinary_EigenSC(shm_name_fluidvar, Nx, Ny, Nz, eigen_bin_name, dt, dx, dy, dz, shm_name_gridx, shm_name_gridy, shm_name_gridz);
      //    if (ret != 0) {
      //       std::cerr << "Error executing Eigen binary: " << eigen_bin_name << std::endl;
      //       std::cerr << "Error code: " << ret << std::endl;
      //    }
      // }  
   } 

   // FREE EVERYTHING
   // Device
   checkCuda(cudaFree(fluidvars));
   checkCuda(cudaFree(intvars));
   checkCuda(cudaFree(x_grid));
   checkCuda(cudaFree(y_grid));
   checkCuda(cudaFree(z_grid));
   
   // Host
   // munmap(shm_h_fluidvar, 8 * fluid_data_size);
   // munmap(shm_h_gridx, sizeof(float) * Nx);
   // munmap(shm_h_gridy, sizeof(float) * Ny);
   // munmap(shm_h_gridz, sizeof(float) * Nz);
   // shm_unlink(shm_name_fluidvar.data());
   // shm_unlink(shm_name_gridx.data());
   // shm_unlink(shm_name_gridy.data());
   // shm_unlink(shm_name_gridz.data());
   return 0;
}