#include <iostream>
#include <string>

#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

#include "hdf5.h" 

// What: 
//  Loads grid data of mesh into shared memory for VTK rendering pipeline 

// Why: 
//  FDM methods are notoriously unstable, this one is no different
//  Therefore, a visualization that studies the appearance of blowup is required
//  This visualization requires knowledge of more than just the grid attributes
//  Consequently, to keep concerns separate, this reads in the grid data 
int main(int argc, char* argv[]){
    std::cout << "Inside read_grid_data" << std::endl;

    // std::string shm_gridattr_name = argv[1];
    std::string filename_grid = argv[1];
    // size_t gridattr_data_size = atoi(argv[3]);
    std::string shm_xgrid_name = argv[2];
    std::string shm_ygrid_name = argv[3];
    std::string shm_zgrid_name = argv[4];

    size_t Nx = atoi(argv[5]);
    size_t Ny = atoi(argv[6]);
    size_t Nz = atoi(argv[7]);

    /* Open shared memory for holding grid attributes */
    // int shm_grid_fd = shm_open(shm_gridattr_name.data(), O_RDWR, 0666);
    // if (shm_grid_fd == -1){
    //     std::cerr << "Inside read_grid_data" << std::endl;
    //     std::cerr << "Failed to open shared memory for grid attribute data" << std::endl;
    //     return EXIT_FAILURE;
    // }

    // float* shm_gridattr = (float*)mmap(0, gridattr_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_grid_fd, 0);

    // int Nx = 0, Ny = 0, Nz = 0;
    // float dx = 0.0, dy = 0.0, dz = 0.0;

    /* Read attributes of grid into shared memory */
    // hid_t grid_file_id, grid_dset_id, grid_attr_id;
    hid_t grid_file_id, grid_dset_id;
    
    herr_t status;

    std::cout << "Opening grid file: " << filename_grid << std::endl;
    grid_file_id = H5Fopen(filename_grid.data(), H5F_ACC_RDONLY, H5P_DEFAULT);
    std::cout << grid_file_id << std::endl;
    if (grid_file_id < 0) {
        std::cout << "Failed to open the data file." << std::endl;
        return 1;
    }
    std::cout << "File: " << filename_grid.data() << " opened successfully" << std::endl;

    // Get necessary attributes and data from `x_grid` dataset
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

    // std::cout << "Reading attributes" << std::endl;
    // grid_attr_id = H5Aopen(grid_dset_id, "spacing", H5P_DEFAULT);
    // status = H5Aread(grid_attr_id, H5T_NATIVE_FLOAT, &dx);
    // grid_attr_id = H5Aopen(grid_dset_id, "dimension", H5P_DEFAULT);
    // status = H5Aread(grid_attr_id, H5T_NATIVE_INT, &Nx);

    // std::cout << "dx = " << dx << ", Nx = " << Nx << std::endl;

    int shm_xgrid_fd = shm_open(shm_xgrid_name.data(), O_RDWR, 0666);
    if (shm_xgrid_fd == -1){
        std::cerr << "Inside read_grid_data" << std::endl;
        std::cerr << "Failed to open shared memory for grid attribute data" << std::endl;
        return EXIT_FAILURE;
    }

    size_t xgrid_data_size = Nx * sizeof(float);

    float* shm_xgrid = (float*)mmap(0, xgrid_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_xgrid_fd, 0);

    status = H5Dread(grid_dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, shm_xgrid);

    // Get necessary attributes and data from `y_grid` dataset
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

    // std::cout << "Reading attributes" << std::endl;
    // grid_attr_id = H5Aopen(grid_dset_id, "spacing", H5P_DEFAULT);
    // status = H5Aread(grid_attr_id, H5T_NATIVE_FLOAT, &dy);
    // grid_attr_id = H5Aopen(grid_dset_id, "dimension", H5P_DEFAULT);
    // status = H5Aread(grid_attr_id, H5T_NATIVE_INT, &Ny);

    // std::cout << "dy = " << dy << ", Ny = " << Ny << std::endl;

    int shm_ygrid_fd = shm_open(shm_ygrid_name.data(), O_RDWR, 0666);
    if (shm_ygrid_fd == -1){
        std::cerr << "Inside read_grid_data" << std::endl;
        std::cerr << "Failed to open shared memory for grid attribute data" << std::endl;
        return EXIT_FAILURE;
    }

    size_t ygrid_data_size = Ny * sizeof(float);

    float* shm_ygrid = (float*)mmap(0, ygrid_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_ygrid_fd, 0);

    status = H5Dread(grid_dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, shm_ygrid);

    // Get necessary attributes and data from `z_grid` dataset
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

    // std::cout << "Reading attributes" << std::endl;
    // grid_attr_id = H5Aopen(grid_dset_id, "spacing", H5P_DEFAULT);
    // status = H5Aread(grid_attr_id, H5T_NATIVE_FLOAT, &dz);
    // grid_attr_id = H5Aopen(grid_dset_id, "dimension", H5P_DEFAULT);
    // status = H5Aread(grid_attr_id, H5T_NATIVE_INT, &Nz);

    // std::cout << "dz = " << dz << ", Nz = " << Nz << std::endl;
    
    int shm_zgrid_fd = shm_open(shm_zgrid_name.data(), O_RDWR, 0666);
    if (shm_zgrid_fd == -1){
        std::cerr << "Inside read_grid_data" << std::endl;
        std::cerr << "Failed to open shared memory for grid attribute data" << std::endl;
        return EXIT_FAILURE;
    }

    size_t zgrid_data_size = Nz * sizeof(float);

    float* shm_zgrid = (float*)mmap(0, zgrid_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_zgrid_fd, 0);

    status = H5Dread(grid_dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, shm_zgrid);

    // VTK and .h5 don't mix so using IPC to get this data
    // shm_gridattr[0] = (float)Nx;
    // shm_gridattr[1] = (float)Ny;
    // shm_gridattr[2] = (float)Nz;
    // shm_gridattr[3] = dx;
    // shm_gridattr[4] = dy;
    // shm_gridattr[5] = dz;

    /* Close everything */
    H5Fclose(grid_file_id);
    H5Dclose(grid_dset_id); 
    // H5Aclose(grid_attr_id);

    // Manage memory
    // munmap(shm_gridattr, gridattr_data_size);
    munmap(shm_xgrid, Nx * sizeof(float));
    munmap(shm_ygrid, Ny * sizeof(float));
    munmap(shm_zgrid, Nz * sizeof(float));
    // close(shm_grid_fd);
    close(shm_xgrid_fd);
    close(shm_ygrid_fd);
    close(shm_zgrid_fd);
    return 0;
}