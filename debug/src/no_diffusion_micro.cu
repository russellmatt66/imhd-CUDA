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
   std::vector<float> inputs (32, 0.0);
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
   float r_max_coeff = inputs[7]; // r_pinch = r_max_coeff * r_max
   
   float x_min = inputs[8];
   float x_max = inputs[9];
   float y_min = inputs[10];
   float y_max = inputs[11];
   float z_min = inputs[12];
   float z_max = inputs[13];
   float dt = inputs[14];
   
   // std::string path_to_data = argv[15];
   // std::string phdf5_bin_name = argv[16];
   // std::string attr_bin_name = argv[17];
   // std::string write_grid_bin_name = argv[18];
   // std::string eigen_bin_name = argv[19];
   // std::string num_proc = argv[20];

   int xgrid_threads = int(inputs[15]);
   int ygrid_threads = int(inputs[16]);
   int zgrid_threads = int(inputs[17]);

   int init_xthreads = int(inputs[18]);
   int init_ythreads = int(inputs[19]);
   int init_zthreads = int(inputs[20]);
   
   int FA_xthreads = int(inputs[21]);
   int FA_ythreads = int(inputs[22]);
   int FA_zthreads = int(inputs[23]);

   int SM_mult_grid_x = int(inputs[24]);
   int SM_mult_grid_y = int(inputs[25]);
   int SM_mult_grid_z = int(inputs[26]);

   int SM_mult_init_x = int(inputs[27]);
   int SM_mult_init_y = int(inputs[28]);
   int SM_mult_init_z = int(inputs[29]);

   int SM_mult_FA_x = int(inputs[30]);
   int SM_mult_FA_y = int(inputs[31]);
   int SM_mult_FA_z = int(inputs[32]);

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
   // dim3 egd_xgrid(SM_mult_grid_x * numberOfSMs, 1, 1); // "egd" = "execution_grid_dimensions"
   // dim3 egd_ygrid(1, SM_mult_grid_y * numberOfSMs, 1);
   // dim3 egd_zgrid(1, 1, SM_mult_grid_z * numberOfSMs);

   // dim3 tbd_xgrid(xgrid_threads, 1, 1); // "tbd = thread_block_dimensions"
   // dim3 tbd_ygrid(1, ygrid_threads, 1);
   // dim3 tbd_zgrid(1, 1, zgrid_threads);
   
   // Execution grid and threadblock configurations for the Grid Initialization kernels
   dim3 egd_grid(SM_mult_grid_x * numberOfSMs, SM_mult_grid_y * numberOfSMs, SM_mult_grid_z * numberOfSMs);
   std::cout << "egd_grid is: (" << egd_grid.x << "," << egd_grid.y << "," << egd_grid.z << ")" << std::endl;

   dim3 tbd_grid(xgrid_threads, ygrid_threads, zgrid_threads);
   std::cout << "tbd_grid is: (" << tbd_grid.x << "," << tbd_grid.y << "," << tbd_grid.z << ")" << std::endl;

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
   
   dim3 tbd_bdry_leftright(8, 1, 8);
   dim3 tbd_bdry_topbottom(1, 8, 8);
   dim3 tbd_bdry_frontback(8, 8, 1); // can also be used for PBCs
   dim3 tbd_qintbdry_frontright(16, 1, 1);
   dim3 tbd_qintbdry_frontbottom(1, 16, 1);
   dim3 tbd_qintbdry_bottomright(1, 1, 16);

   // Execution grid and threadblock configurations for the Predictor and Corrector microkernels
   dim3 egd_fluidadvance(SM_mult_FA_x * numberOfSMs, SM_mult_FA_y * numberOfSMs, SM_mult_FA_z * numberOfSMs);
   std::cout << "egd_fluidadvance is: (" << egd_fluidadvance.x << "," << egd_fluidadvance.y << "," << egd_fluidadvance.z << ")" << std::endl;

   dim3 tbd_fluidadvance(FA_xthreads, FA_ythreads, FA_zthreads); 
   std::cout << "tbd_fluidadvance is: (" << tbd_fluidadvance.x << "," << tbd_fluidadvance.y << "," << tbd_fluidadvance.z << ")" << std::endl;

   // dim3 tbd_fluidadvance(6, 6, 6); 

   // TODO: Microkernels should be in a wrapper   
   InitializeGrid<<<egd_grid, tbd_grid>>>(x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz, x_grid, y_grid, z_grid, Nx, Ny, Nz);
   // InitializeX<<<egd_xgrid, tbd_xgrid>>>(x_grid, x_min, dx, Nx);
   // InitializeY<<<egd_ygrid, tbd_ygrid>>>(y_grid, y_min, dy, Ny);
   // InitializeZ<<<egd_zgrid, tbd_zgrid>>>(z_grid, z_min, dz, Nz);
   ZeroVars<<<egd_init, tbd_init>>>(fluidvars, Nx, Ny, Nz); 
   ZeroVars<<<egd_init, tbd_init>>>(intvars, Nx, Ny, Nz); 
   checkCuda(cudaDeviceSynchronize());

   /*
   TODO: "ScrewPinch" doesn't work
   */ 
   // ScrewPinch<<<egd_init, tbd_init>>>(fluidvars, J0, r_max_coeff, x_grid, y_grid, z_grid, Nx, Ny, Nz);
   ScrewPinchLoop<<<egd_init, tbd_init>>>(fluidvars, J0, x_grid, y_grid, z_grid, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());

   rigidConductingWallBCsLeftRight<<<egd_bdry_leftright, tbd_bdry_leftright>>>(fluidvars, Nx, Ny, Nz);
   rigidConductingWallBCsTopBottom<<<egd_bdry_topbottom, tbd_bdry_topbottom>>>(fluidvars, Nx, Ny, Nz);
   PBCs<<<egd_bdry_frontback, tbd_bdry_frontback>>>(fluidvars, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());

   /*
   NOTE: For some reason the microkernels don't work here, have to use megakernel
   */ 
   ComputeIntermediateVariablesNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   // ComputeIntRhoMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   // ComputeIntRhoVXMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   // ComputeIntRhoVYMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   // ComputeIntRhoVZMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   // ComputeIntBXMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   // ComputeIntBYMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   // ComputeIntBZMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   // ComputeIntEMicroLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
   checkCuda(cudaDeviceSynchronize());    

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

   // SIMULATION LOOP
   for (int it = 1; it < Nt; it++){
      std::cout << "Starting timestep " << it << std::endl;

      std::cout << "Launching microkernels for computing fluid variables" << std::endl;
      // FluidAdvanceLocalNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
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
      // BoundaryConditionsNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
      PBCs<<<egd_bdry_frontback, tbd_bdry_frontback>>>(fluidvars, Nx, Ny, Nz);
      checkCuda(cudaDeviceSynchronize());
      std::cout << "Kernels for computing fluid variables completed" << std::endl;
      
      std::cout << "Launching microkernels for computing intermediate variables" << std::endl; 
      // ComputeIntermediateVariablesNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz);
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
      // ComputeIntermediateVariablesBoundaryNoDiff<<<egd_fluidadvance, tbd_fluidadvance>>>(fluidvars, intvars, dt, dx, dy, dz, Nx, Ny, Nz); // Hella slow b/c megakernel
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

   } 

   // FREE EVERYTHING
   // Device
   checkCuda(cudaFree(fluidvars));
   checkCuda(cudaFree(intvars));
   checkCuda(cudaFree(x_grid));
   checkCuda(cudaFree(y_grid));
   checkCuda(cudaFree(z_grid));
   return 0;
}