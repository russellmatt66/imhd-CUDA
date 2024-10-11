#include <iostream>
#include <string>

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include "hdf5.h" 

// Load fluidvar data from .h5 file into shared memory for VTK consumption 
int main(int argc, char* argv[]){
    std::cout << "Inside read_fluid_data" << std::endl;

    std::string shm_fluidvar_name = argv[1];
    std::string dset_name = argv[2];
    std::string filename_frame = argv[3];
    int Nx = atoi(argv[4]);
    int Ny = atoi(argv[5]);
    int Nz = atoi(argv[6]);

    /* Open fluidvar shared memory created by `driver.cpp` */
    int shm_fluidvar_fd = shm_open(shm_fluidvar_name.data(), O_RDWR, 0666);
    if (shm_fluidvar_fd == -1){
        std::cerr << "Inside read_fluid_data" << std::endl;
        std::cerr << "Failed to open shared memory for fluidvar data" << std::endl;
        return EXIT_FAILURE;
    }

    size_t fluid_data_size = sizeof(float) * Nx * Ny * Nz;
    float* shm_fluidvar = (float*)mmap(0, fluid_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fluidvar_fd, 0);

    if (shm_fluidvar == MAP_FAILED) {
        std::cerr << "mmap failed for fluidvar data!" << std::endl;
        return EXIT_FAILURE;
    }

    /* Read dset data into shared fluidvar memory */
    hid_t file_id, dset_id;

    herr_t status;

    std::cout << "Opening file: " << filename_frame.data() << std::endl;
    file_id = H5Fopen(filename_frame.data(), H5F_ACC_RDONLY, H5P_DEFAULT);
    std::cout << file_id << std::endl;
    if (file_id < 0) {
        std::cout << "Failed to open the data file." << std::endl;
        return 1;
    }
    std::cout << "File: " << filename_frame.data() << " opened successfully" << std::endl;

    std::cout << "Opening dataset: " << dset_name.data() << std::endl;
    dset_id = H5Dopen(file_id, dset_name.data(), H5P_DEFAULT);
    if (dset_id < 0) {
        std::cout << "Failed to open the dataset." << std::endl;
        H5Fclose(file_id);
        return 1;
    }
    std::cout << "Dataset: " << dset_name.data() << " opened successfully" << std::endl;

    status = H5Dread(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, shm_fluidvar);
    if (status < 0){
        std::cerr << "Failed to read dataset successfully" << std::endl;
    }
    std::cout << "Dataset read successfully" << std::endl;

    /* Close everything */
    H5Fclose(file_id);
    H5Dclose(dset_id);

    // Manage memory
    munmap(shm_fluidvar, fluid_data_size);
    close(shm_fluidvar_fd);
    return 0;
}