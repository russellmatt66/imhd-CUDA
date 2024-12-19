#ifndef VIS_HELPER_HPP
#define VIS_HELPER_HPP

#include <iostream>
#include <string>

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// Header files for the basic functionality utilized by the visualization code 
int getGridAttributes_SHM(float* shm_gridattr, const std::string shm_gridattr_name, const std::string filename_grid);

int setupFluidVarData_SHM(float* shm_fluidvar, const std::string shm_fluidvar_name, const int Nx, const int Ny, const int Nz);
int setupFluidVarPlane_SHM(float* shm_fluidvar, const std::string shm_fluidvar_name, const int N_1, const int N_2);

#endif