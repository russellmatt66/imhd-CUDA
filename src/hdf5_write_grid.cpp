/*
Writes the mesh data for the simulation to a .h5 file 
*/

#include <iostream>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

#include "hdf5.h"

void writeGrid(const std::string file_name, const float *shm_x_grid, const float *shm_y_grid, const float *shm_z_grid, const int Nx, const int Ny, const int Nz);

int main(int argc, char* argv[]){
    int Nx = atoi(argv[1]);
    int Ny = atoi(argv[2]);
    int Nz = atoi(argv[3]);
    std::string shm_name_gridx = argv[4];
    std::string shm_name_gridy = argv[5];
    std::string shm_name_gridz = argv[6];
    std::string path_to_data = argv[7];

    std::string file_name = path_to_data + "grid.h5";

    /* Access shared memory */
    int shm_fd = shm_open(shm_name_gridx.data(), O_RDWR, 0666);
    if (shm_fd == -1){
        std::cerr << "Inside write_grid" << std::endl;
        std::cerr << "Failed to open shared memory for x-axis data" << std::endl;
        return EXIT_FAILURE;
    }

    float *shm_x_grid = (float*)mmap(0, sizeof(float) * Nx, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_x_grid == MAP_FAILED){
        std::cerr << "Inside write_grid" << std::endl;
        std::cerr << "Failed to connect pointer to shared memory for x-axis data" << std::endl;
        return EXIT_FAILURE;
    }

    shm_fd = shm_open(shm_name_gridy.data(), O_RDWR, 0666);
    if (shm_fd == -1){
        std::cerr << "Inside write_grid" << std::endl;
        std::cerr << "Failed to open shared memory for y-axis data" << std::endl;
        return EXIT_FAILURE;
    }

    float *shm_y_grid = (float*)mmap(0, sizeof(float) * Nx, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_y_grid == MAP_FAILED){
        std::cerr << "Inside write_grid" << std::endl;
        std::cerr << "Failed to connect pointer to shared memory for y-axis data" << std::endl;
        return EXIT_FAILURE;
    }

    shm_fd = shm_open(shm_name_gridz.data(), O_RDWR, 0666);
    if (shm_fd == -1){
        std::cerr << "Inside write_grid" << std::endl;
        std::cerr << "Failed to open shared memory for z-axis data" << std::endl;
        return EXIT_FAILURE;
    }

    float *shm_z_grid = (float*)mmap(0, sizeof(float) * Nx, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_z_grid == MAP_FAILED){
        std::cerr << "Inside write_grid" << std::endl;
        std::cerr << "Failed to connect pointer to shared memory for z-axis data" << std::endl;
        return EXIT_FAILURE;
    }

    writeGrid(file_name, shm_x_grid, shm_y_grid, shm_z_grid, Nx, Ny, Nz);
    return 0;
}

/* Write the mesh data to a .h5 file */
void writeGrid(const std::string file_name, const float *shm_x_grid, const float *shm_y_grid, const float *shm_z_grid, const int Nx, const int Ny, const int Nz){
    hid_t file_id;
    hid_t dset_id_x, dset_id_y, dset_id_z;
    hid_t dspc_id_x, dspc_id_y, dspc_id_z;

    hsize_t xdim[1] = {Nx}, ydim[1] = {Ny}, zdim[1] = {Nz};

    herr_t status;

    const char* dsn_x = "x_grid";
    const char* dsn_y = "y_grid";
    const char* dsn_z = "z_grid";

    float dx = (shm_x_grid[Nx-1] - shm_x_grid[0]) / (Nx - 1);
    float dy = (shm_y_grid[Ny-1] - shm_y_grid[0]) / (Ny - 1); 
    float dz = (shm_z_grid[Nz-1] - shm_z_grid[0]) / (Nz - 1); 

    file_id = H5Fcreate(file_name.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    dspc_id_x = H5Screate_simple(1, xdim, NULL);
    dspc_id_y = H5Screate_simple(1, ydim, NULL);
    dspc_id_z = H5Screate_simple(1, zdim, NULL);

    dset_id_x = H5Dcreate(file_id, dsn_x, H5T_NATIVE_FLOAT, dspc_id_x, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id_y = H5Dcreate(file_id, dsn_y, H5T_NATIVE_FLOAT, dspc_id_y, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id_z = H5Dcreate(file_id, dsn_z, H5T_NATIVE_FLOAT, dspc_id_z, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    status = H5Dwrite(dset_id_x, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, shm_x_grid);
    status = H5Dwrite(dset_id_y, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, shm_y_grid);
    status = H5Dwrite(dset_id_z, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, shm_z_grid);

    // Also need attributes describing dimensionality
    hid_t attr_id;
    hid_t aspc_id;

    aspc_id = H5Screate(H5S_SCALAR);

    attr_id = H5Acreate(dset_id_x, "spacing", H5T_NATIVE_FLOAT, aspc_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attr_id, H5T_NATIVE_FLOAT, &dx);
    
    attr_id = H5Acreate(dset_id_y, "spacing", H5T_NATIVE_FLOAT, aspc_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attr_id, H5T_NATIVE_FLOAT, &dy);

    attr_id = H5Acreate(dset_id_z, "spacing", H5T_NATIVE_FLOAT, aspc_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attr_id, H5T_NATIVE_FLOAT, &dz);

    status = H5Fclose(file_id);
    status = H5Sclose(dspc_id_x);
    status = H5Sclose(dspc_id_y);
    status = H5Sclose(dspc_id_z);
    status = H5Sclose(aspc_id);
    status = H5Dclose(dset_id_x);
    status = H5Dclose(dset_id_y);
    status = H5Dclose(dset_id_z);
    status = H5Aclose(attr_id);
    return;
}