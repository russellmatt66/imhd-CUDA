#include <string>
#include <iostream>
#include <cstdlib>

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include "hdf5.h"

// int fluidvarNameMap(std::string fluidvar_name);

int main(int argc, char* argv[]){
    std::cout << "Inside viewframe_driver" << std::endl;
    /* Accept name of .h5 file to view */
    std::string fluidvar_name = argv[1]; // name of variable to render
    std::string filename_frame = argv[2];
    std::string filename_grid = argv[3];

    hsize_t Nx = 0, Ny = 0, Nz = 0;

    float dx = 0.0, dy = 0.0, dz = 0.0;
    float x_min = 0.0, x_max = 0.0, y_min = 0.0, y_max = 0.0, z_min = 0.0, z_max = 0.0;

    /* Error checking for `fluidvar_name` */

    // Open grid file and get dimensionality, and spacing attributes
    // It's just easier for this to be "spaghetti"
    // Spaghetti is only spaghetti if there is an equally simple way of doing things that is more compact and elegant
    // Here, functionalizing this would require side-effects or non-trivial memory management - overly complex
    // Open to hearing ways to improve this, however.
    hid_t grid_file_id, grid_dset_id, grid_attr_id, grid_dspc_id;
    
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

    // Need the values of the minimum and maximum grid points
    std::cout << "Getting min and max axis values" << std::endl;
    grid_dspc_id = H5Dget_space(grid_dset_id);

    hsize_t start[1] = {0};
    hsize_t count[1] = {1};

    status = H5Sselect_hyperslab(grid_dspc_id, H5S_SELECT_SET, start, NULL, count, NULL);
    status = H5Dread(grid_dset_id, H5T_NATIVE_FLOAT, H5S_ALL, grid_dspc_id, H5P_DEFAULT, &x_min);
    
    // start[0] = Nx - 1;
    // // status = H5Sselect_elements(grid_dspc_id, H5S_SELECT_SET, 1, offset);
    // status = H5Sselect_hyperslab(grid_dspc_id, H5S_SELECT_SET, start, NULL, count, NULL);
    // status = H5Dread(grid_dset_id, H5T_NATIVE_FLOAT, H5S_ALL, grid_dspc_id, H5P_DEFAULT, &x_max);
    // if (status < 0) {
    //     std::cerr << "Failed to read x_max" << std::endl;
    //     return 1;
    // }
    x_max = x_min + dx * (Nx - 1); // The above won't work, and won't tell me why not

    std::cout << "x_min = " << x_min << ", x_max = " << x_max << std::endl;

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

    std::cout << "Getting min and max axis values" << std::endl;  
    grid_dspc_id = H5Dget_space(grid_dset_id);

    start[0] = 0;
    // status = H5Sselect_elements(grid_dspc_id, H5S_SELECT_SET, 1, offset);
    status = H5Sselect_hyperslab(grid_dspc_id, H5S_SELECT_SET, start, NULL, count, NULL);
    status = H5Dread(grid_dset_id, H5T_NATIVE_FLOAT, H5S_ALL, grid_dspc_id, H5P_DEFAULT, &y_min);
    
    // start[0] = Ny - 1;
    // // status = H5Sselect_elements(grid_dspc_id, H5S_SELECT_SET, 1, offset);
    // status = H5Sselect_hyperslab(grid_dspc_id, H5S_SELECT_SET, start, NULL, count, NULL);
    // status = H5Dread(grid_dset_id, H5T_NATIVE_FLOAT, H5S_ALL, grid_dspc_id, H5P_DEFAULT, &y_max);
    y_max = y_min + dy * (Ny - 1); // The above won't work, and won't tell me why not

    std::cout << "y_min = " << y_min << ", y_max = " << y_max << std::endl;

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

    grid_dspc_id = H5Dget_space(grid_dset_id);

    start[0] = 0;
    // status = H5Sselect_elements(grid_dspc_id, H5S_SELECT_SET, 1, offset);
    status = H5Sselect_hyperslab(grid_dspc_id, H5S_SELECT_SET, start, NULL, count, NULL);
    status = H5Dread(grid_dset_id, H5T_NATIVE_FLOAT, H5S_ALL, grid_dspc_id, H5P_DEFAULT, &z_min);
    
    // start[0] = Nz - 1;
    // status = H5Sselect_elements(grid_dspc_id, H5S_SELECT_SET, 1, offset);
    // status = H5Sselect_hyperslab(grid_dspc_id, H5S_SELECT_SET, start, NULL, count, NULL);
    // status = H5Dread(grid_dset_id, H5T_NATIVE_FLOAT, H5S_ALL, grid_dspc_id, H5P_DEFAULT, &z_max);
    z_max = z_min + dz * (Nz - 1); // The above won't work, and won't tell me why not

    std::cout << "z_min = " << z_min << ", z_max = " << z_max << std::endl;

    // Open fluidvar file with HDF5 and get the frame data
    size_t fluid_data_size = sizeof(float) * Nx * Ny * Nz;

    std::string shm_name_fluidvar = "/shared_h_fluidvar";
    int shm_fd = shm_open(shm_name_fluidvar.data(), O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Inside viewframe_driver" << std::endl; 
        std::cerr << "Failed to create shared memory!" << std::endl;
        return EXIT_FAILURE;
    }
    
    ftruncate(shm_fd, fluid_data_size);

    float* shm_h_fluidvar = (float*)mmap(0, fluid_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_h_fluidvar == MAP_FAILED) {
        std::cerr << "mmap failed!" << std::endl;
        return EXIT_FAILURE;
    }

    hid_t file_id, dset_id;

    std::cout << "Opening file: " << filename_frame.data() << std::endl;
    file_id = H5Fopen(filename_frame.data(), H5F_ACC_RDONLY, H5P_DEFAULT);
    std::cout << file_id << std::endl;
    if (file_id < 0) {
        std::cout << "Failed to open the data file." << std::endl;
        return 1;
    }
    std::cout << "File: " << filename_frame.data() << " opened successfully" << std::endl;

    std::cout << "Opening dataset: " << fluidvar_name.data() << std::endl;
    dset_id = H5Dopen(file_id, fluidvar_name.data(), H5P_DEFAULT);
    if (dset_id < 0) {
        std::cout << "Failed to open the dataset." << std::endl;
        H5Fclose(file_id);
        return 1;
    }
    std::cout << "Dataset: " << fluidvar_name.data() << " opened successfully" << std::endl;

    status = H5Dread(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, shm_h_fluidvar);

    /* 
    Fork process to run `view_frame` 
    Needs:
    - Nx, Ny, Nz
    - dx, dy, dz
    - x_min, x_max, y_min, y_max, z_min, z_max
    - shm_name_fluidvar
    */
    std::cout << "Forking process to run `view_frame` binary" << std::endl;

    std::cout << "Value of (Nx, Ny, Nz) = (" << Nx << "," << Ny << "," << Nz << ")" << std::endl;
    std::cout << "Value of (dx, dy, dz) = (" << dx << "," << dy << "," << dz << ")" << std::endl;
    std::cout << "Value of (x_min, y_min, z_min) = (" << x_min << "," << y_min << "," << z_min << ")" << std::endl;
    std::cout << "Value of (x_max, y_max, z_max) = (" << x_max << "," << y_max << "," << z_max << ")" << std::endl;

    std::string viewframe_command = "./view_frame " + std::to_string(Nx) + " " + std::to_string(Ny) + " " + std::to_string(Nz) 
        + " " + std::to_string(dx) + " " + std::to_string(dy) + " " + std::to_string(dz) 
        + " " + std::to_string(x_min) + " " + std::to_string(x_max) + " " + std::to_string(y_min) + " " + std::to_string(y_max) 
        + " " + std::to_string(z_min) + " " + std::to_string(z_max) + " " + shm_name_fluidvar;  
    int ret = std::system(viewframe_command.data());

    // Memory Management
    // HDF5 objects
    H5Fclose(grid_file_id);
    H5Dclose(grid_dset_id);
    H5Sclose(grid_dspc_id);
    H5Aclose(grid_attr_id);
    H5Fclose(file_id);
    H5Dclose(dset_id);

    // Shared Memory
    shm_unlink(shm_name_fluidvar.data());
    munmap(shm_h_fluidvar, fluid_data_size);
    close(shm_fd);
    return 0;
}

// This returns the number of datacubes that you have to offset to get to the desired information
// int fluidvarNameMap(std::string fluidvar_name){
//     int key = 0;

//     if (fluidvar_name == "rho"){
//         key = 0;
//     }
//     else if (fluidvar_name == "rhovx"){
//         key = 1;
//     }
//     else if (fluidvar_name == "rhovy"){
//         key = 2;
//     }
//     else if (fluidvar_name == "rhovz"){
//         key = 3;
//     }
//     else if (fluidvar_name == "Bx"){
//         key = 4;
//     }
//     else if (fluidvar_name == "By"){
//         key = 5;
//     }
//     else if (fluidvar_name == "Bz"){
//         key = 6;
//     }
//     else if (fluidvar_name == "e"){
//         key = 7;
//     }

//     return key;
// }
