// Definitions for the basic functionality utilized by the visualization code
// Code reuse = good
// DRY = good
 
#include "vis_helper.hpp"

// Obtains the dimensions, and spacing, associated with the grid 
// Shared memory, and forks, because VTK does not play nice w/HDF5 unless you do the specific leg work to ensure that it does - I didn't 
int getGridAttributes_SHM(float* shm_gridattr, const std::string shm_gridattr_name, const std::string filename_grid){
    int shm_fd_gridattr = shm_open(shm_gridattr_name.data(), O_CREAT | O_RDWR, 0666);
    if (shm_fd_gridattr == -1){
        std::cerr << "Failed to create shared memory for grid attributes" << std::endl;
        return 1;
    }

    size_t gridattr_data_size = sizeof(float) * 6;
    ftruncate(shm_fd_gridattr, gridattr_data_size); // Nx, Ny, Nz, ...

    shm_gridattr = (float*)mmap(0, gridattr_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_gridattr, 0);
    if (shm_gridattr == MAP_FAILED) {
        std::cerr << "Failed to map shared memory for grid attributes" << std::endl;
        return 1;
    }

    // Fork process to obtain necessary grid attributes
    std::string gridattr_command = "./read_grid_data " + shm_gridattr_name + " " + filename_grid + " " + std::to_string(gridattr_data_size);
    std::cout << "Forking to process for obtaining grid attributes" << std::endl;
    int ret = std::system(gridattr_command.data());
    if (ret != 0) {
        std::cerr << "Error executing command: " << gridattr_command << std::endl;
        return ret;
    }
    return 0;
}

// These just prepare shared memory to store fluid data in
// Does not fork to process for populating because the population needs to be done repetitively 
int setupFluidVarData_SHM(float* shm_fluidvar, const std::string shm_fluidvar_name, const int Nx, const int Ny, const int Nz){
    int shm_fd_fluidvar = shm_open(shm_fluidvar_name.data(), O_CREAT | O_RDWR, 0666);
    if (shm_fd_fluidvar == -1){
        std::cerr << "Failed to create shared memory for fluidvar data" << std::endl;
        return 1;
    }

    size_t fluid_data_size = sizeof(float) * Nx * Ny * Nz;
    ftruncate(shm_fd_fluidvar, fluid_data_size);

    shm_fluidvar = (float*)mmap(0, fluid_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_fluidvar, 0);
    if (shm_fluidvar == MAP_FAILED) {
        std::cerr << "Failed to map shared memory for fluidvar data" << std::endl;
        return 1;
    }
    return 0;
}

// Same as above, but with 2D case, i.e., a plane
int setupFluidVarPlane_SHM(float* shm_fluidvar, const std::string shm_fluidvar_name, const int N_1, const int N_2){
    int shm_fd_fluidvar = shm_open(shm_fluidvar_name.data(), O_CREAT | O_RDWR, 0666);
    if (shm_fd_fluidvar == -1){
        std::cerr << "Failed to create shared memory for fluidvar data" << std::endl;
        return 1;
    }

    size_t fluid_plane_size = sizeof(float) * N_1 * N_2;
    ftruncate(shm_fd_fluidvar, fluid_plane_size);

    shm_fluidvar = (float*)mmap(0, fluid_plane_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_fluidvar, 0);
    if (shm_fluidvar == MAP_FAILED) {
        std::cerr << "Failed to map shared memory for fluidvar data" << std::endl;
        return 1;
    }
    return 0;
}