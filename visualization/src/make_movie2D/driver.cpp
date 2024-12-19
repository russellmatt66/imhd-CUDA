#include <iostream>
#include <string>

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char* argv[]){
    std::cout << "Inside make_movie driver" << std::endl;

    std::string path_to_data = argv[1];
    std::string dset_name = argv[2]; // specific fluid variable to visualize
    std::string filename_grid = argv[3];
    std::string plane_name = argv[4]; // 'xy', 'xz', or 'yz'
    size_t plane_val = atoi(argv[5]); // {[0, Nx), [0, Ny), [0, Nz)}
    size_t Nt = atoi(argv[6]);

    // Get mesh dimensions
    // Allocate shared memory that will store the necessary attributes
    // = Nx, Ny, Nz, dx, dy, dz
    // Shared memory = workaround b/c VTK and .h5 files don't mix
    std::string shm_gridattr_name = "/shm_grid_attributes";
    int shm_fd_gridattr = shm_open(shm_gridattr_name.data(), O_CREAT | O_RDWR, 0666);
    if (shm_fd_gridattr == -1){
        std::cerr << "Failed to create shared memory for grid attributes" << std::endl;
        return 1;
    }

    size_t gridattr_data_size = sizeof(float) * 6;
    ftruncate(shm_fd_gridattr, gridattr_data_size); // Nx, Ny, Nz, ...

    float* shm_gridattr = (float*)mmap(0, gridattr_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_gridattr, 0);
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
        return 1;
    }

    int Nx = (int)shm_gridattr[0], Ny = (int)shm_gridattr[1], Nz = (int)shm_gridattr[2];
    float dx = shm_gridattr[3], dy = shm_gridattr[4], dz = shm_gridattr[5];

    std::cout << "Returned successfully from process. Attribute values are: " << std::endl;
    std::cout << "(Nx, Ny, Nz) = " << "(" << Nx << "," << Ny << "," << Nz << ")" << std::endl;
    std::cout << "(dx, dy, dz) = " << "(" << dx << "," << dy << "," << dz << ")" << std::endl; 

    // Allocate shared memory that will contain the fluidvar data 
    std::cout << "Allocating shared memory for fluidvar data" << std::endl;
    std::string shm_fluidvar_name = "/shm_fluidvar_data";
    int shm_fd_fluidvar = shm_open(shm_fluidvar_name.data(), O_CREAT | O_RDWR, 0666);
    if (shm_fd_fluidvar == -1){
        std::cerr << "Failed to create shared memory for fluidvar data" << std::endl;
        return 1;
    }

    size_t N_1 = 0, N_2 = 0; // N_1 is major dimension, N_2 is minor dimension of plane - don't want to allocate shared memory inside of a conditional statement

    if (plane_name == "xy" || plane_name == "yx"){ // visualize xy plane
        N_1 = Nx;
        N_2 = Ny;
    }
    else if (plane_name == "xz" || plane_name == "zx"){ // visualize xz plane
        N_1 = Nx;
        N_2 = Nz;
    }
    else if (plane_name == "yz" || plane_name == "zy"){ // visualize yz plane
        N_1 = Ny;
        N_2 = Nz;
    }
    else { // complain about the input data
        /* Print a helpful message */
    }

    size_t fluid_plane_size = sizeof(float) * N_1 * N_2;
    ftruncate(shm_fd_fluidvar, fluid_plane_size);

    float* shm_fluidvar = (float*)mmap(0, fluid_plane_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_fluidvar, 0);
    if (shm_fluidvar == MAP_FAILED) {
        std::cerr << "Failed to map shared memory for fluidvar data" << std::endl;
        return 1;
    }
    std::cout << "Shared memory for fluidvar data successfully allocated" << std::endl;

    /* Fork to process that reads the given plane in, for the specified fluid variable */
    std::string read_fluid_plane_command = "";

    for (size_t it = 0; it < Nt; it++){
        /* Create a movie of the state of the given plane with VTK */
    }

    // Manage memory
    std::cout << "Freeing shared memory" << std::endl;
    shm_unlink(shm_gridattr_name.data());
    shm_unlink(shm_fluidvar_name.data());
    munmap(shm_gridattr, gridattr_data_size);
    munmap(shm_fluidvar, fluid_plane_size);
    close(shm_fd_gridattr);
    close(shm_fd_fluidvar);
    return 0;
}