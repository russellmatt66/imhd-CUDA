#include <iostream>
#include <string>

#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

#include "hdf5.h" 

// Load necessary attributes of mesh for rendering pipeline
int main(int argc, char* argv[]){
    std::cout << "Inside read_grid_data" << std::endl;

    std::string shm_gridattr_name = argv[1];
    std::string filename_grid = argv[2];
    size_t gridattr_data_size = atoi(argv[3]);

    /* Open shared memory for holding grid attributes */
    int shm_grid_fd = shm_open(shm_gridattr_name.data(), O_RDWR, 0666);
    if (shm_grid_fd == -1){
        std::cerr << "Inside read_grid_data" << std::endl;
        std::cerr << "Failed to open shared memory for grid attribute data" << std::endl;
        return EXIT_FAILURE;
    }

    float* shm_gridattr = (float*)mmap(0, gridattr_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_grid_fd, 0);

    int Nx = 0, Ny = 0, Nz = 0;
    float dx = 0.0, dy = 0.0, dz = 0.0;

    /* Read attributes of grid into shared memory */
    hid_t grid_file_id, grid_dset_id, grid_attr_id;
    
    herr_t status;

    std::cout << "Opening grid file: " << filename_grid << std::endl;
    grid_file_id = H5Fopen(filename_grid.data(), H5F_ACC_RDONLY, H5P_DEFAULT);
    std::cout << grid_file_id << std::endl;
    if (grid_file_id < 0) {
        std::cout << "Failed to open the data file." << std::endl;
        return 1;
    }
    std::cout << "File: " << filename_grid.data() << " opened successfully" << std::endl;

    // Get necessary attributes from `x_grid` dataset
    std::string grid_dset_name = "x_grid";
    std::cout << "Opening dataset: " << grid_dset_name.data() << std::endl;
    grid_dset_id = H5Dopen(grid_file_id, grid_dset_name.data(), H5P_DEFAULT);
    if (grid_dset_id < 0) {
        std::cout << "Failed to open the dataset: " << grid_dset_name.data() << std::endl;
        std::cout << "Closing grid_file, and returning" << std::endl;
        H5Fclose(grid_file_id);
        return 1;
    }
    std::cout << "Dataset: " << grid_dset_name.data() << " opened successfully" << std::endl;

    std::cout << "Reading attributes" << std::endl;
    grid_attr_id = H5Aopen(grid_dset_id, "spacing", H5P_DEFAULT);
    status = H5Aread(grid_attr_id, H5T_NATIVE_FLOAT, &dx);
    grid_attr_id = H5Aopen(grid_dset_id, "dimension", H5P_DEFAULT);
    status = H5Aread(grid_attr_id, H5T_NATIVE_INT, &Nx);

    std::cout << "dx = " << dx << ", Nx = " << Nx << std::endl;

    // Get necessary attributes from `y_grid` dataset
    grid_dset_name = "y_grid";
    std::cout << "Opening dataset: " << grid_dset_name.data() << std::endl;
    grid_dset_id = H5Dopen(grid_file_id, grid_dset_name.data(), H5P_DEFAULT);
    if (grid_dset_id < 0) {
        std::cout << "Failed to open the dataset: " << grid_dset_name.data() << std::endl;
        std::cout << "Closing grid_file, and returning" << std::endl;
        H5Fclose(grid_file_id);
        return 1;
    }
    std::cout << "Dataset: " << grid_dset_name.data() << " opened successfully" << std::endl;

    std::cout << "Reading attributes" << std::endl;
    grid_attr_id = H5Aopen(grid_dset_id, "spacing", H5P_DEFAULT);
    status = H5Aread(grid_attr_id, H5T_NATIVE_FLOAT, &dy);
    grid_attr_id = H5Aopen(grid_dset_id, "dimension", H5P_DEFAULT);
    status = H5Aread(grid_attr_id, H5T_NATIVE_INT, &Ny);

    std::cout << "dy = " << dy << ", Ny = " << Ny << std::endl;

    // Get necessary attributes from `z_grid` dataset
    grid_dset_name = "z_grid";
    std::cout << "Opening dataset: " << grid_dset_name.data() << std::endl;
    grid_dset_id = H5Dopen(grid_file_id, grid_dset_name.data(), H5P_DEFAULT);
    if (grid_dset_id < 0) {
        std::cout << "Failed to open the dataset: " << grid_dset_name.data() << std::endl;
        std::cout << "Closing grid_file, and returning" << std::endl;
        H5Fclose(grid_file_id);
        return 1;
    }
    std::cout << "Dataset: " << grid_dset_name.data() << " opened successfully" << std::endl;

    std::cout << "Reading attributes" << std::endl;
    grid_attr_id = H5Aopen(grid_dset_id, "spacing", H5P_DEFAULT);
    status = H5Aread(grid_attr_id, H5T_NATIVE_FLOAT, &dz);
    grid_attr_id = H5Aopen(grid_dset_id, "dimension", H5P_DEFAULT);
    status = H5Aread(grid_attr_id, H5T_NATIVE_INT, &Nz);

    std::cout << "dz = " << dz << ", Nz = " << Nz << std::endl;
    
    // VTK and .h5 don't mix so using IPC to get this data
    shm_gridattr[0] = (float)Nx;
    shm_gridattr[1] = (float)Ny;
    shm_gridattr[2] = (float)Nz;
    shm_gridattr[3] = dx;
    shm_gridattr[4] = dy;
    shm_gridattr[5] = dz;

    /* Close everything */
    H5Fclose(grid_file_id);
    H5Dclose(grid_dset_id); 
    H5Aclose(grid_attr_id);

    // Manage memory
    munmap(shm_gridattr, gridattr_data_size);
    close(shm_grid_fd);
    return 0;
}