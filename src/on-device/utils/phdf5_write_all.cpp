/* 
DRIVER CODE FOR WRITING SIMULATION DATA TO STORAGE
*/
/* 
Library function for writing simulation data to storage using Parallel HDF5 
Because of the PHDF5 requirement to use MPI, this call must fork from the parent CUDA process which is launching the GPU kernels
This necessitates the data be in shared memory  
*/
#include <string>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <fstream>

#include "hdf5.h"
#include "mpi.h"

void writeH5FileAll(const std::string file_name, const float* output_data, const int Nx, const int Ny, const int Nz);

int main(int argc, char* argv[]){
    /* Parse arguments, and call writeH5FileAll with shared memory data */
    std::string file_name = argv[1];
    int Nx = atoi(argv[2]);
    int Ny = atoi(argv[3]);
    int Nz = atoi(argv[4]);
    std::string shm_name = argv[5];
    size_t data_size = atoi(argv[6]);

    int shm_fd = shm_open(shm_name.data(), O_RDWR, 0666);
    if (shm_fd == -1){
        std::cerr << "Inside phdf5_writeall" << std::endl;
        std::cerr << "Failed to open shared memory" << std::endl;
        return EXIT_FAILURE;
    }

    float* shm_h_fluidvar = (float*)mmap(0, data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_h_fluidvar == MAP_FAILED){
        std::cerr << "Inside phdf5_writeall" << std::endl;
        std::cerr << "Failed to connect pointer to shared memory" << std::endl;
        return EXIT_FAILURE;
    }

    // testWrite();
    // Parallel HDF5 that writes datasets in parallel
    writeH5FileAll(file_name, shm_h_fluidvar, Nx, Ny, Nz);

    // Write attributes to the datasets in serial because PHDF5 attribute writing is a novice trap
    // writeAttributes(filename, Nx, Ny, Nz);

    munmap(shm_h_fluidvar, data_size);
    close(shm_fd);
    return 0;
}

// EXAMPLE: https://cvw.cac.cornell.edu/parallel-io-libraries/phdf5/parallel_write.c 
void writeH5FileAll(const std::string file_name, const float* output_data, const int Nx, const int Ny, const int Nz){
    MPI_Init(NULL, NULL);
    int world_size, rank;

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Info info = MPI_INFO_NULL;
    
    MPI_Comm_size(comm, &world_size); 
    MPI_Comm_rank(comm, &rank);
    
    if (world_size > 8){
        std::cerr << "Error: total number of processes for data output must be no more than 8" << std::endl;
        MPI_Finalize();
        return;
    }

    std::cout << "Hello from process: " << rank << " out of " << world_size << std::endl;

    hid_t plist_id;
    hid_t file_id, dspc_id, mspace;
    hid_t attrdim_id, attrdim_dspc_id;
    hid_t attrstorage_id, attrstorage_dspc_id;
    hid_t strtype_id;

    // hid_t dset_id[8] = {0};
    // Because the above doesn't work
    hid_t dset_id_rho, dset_id_rhovx, dset_id_rhovy, dset_id_rhovz, dset_id_Bx, dset_id_By, dset_id_Bz, dset_id_e;
    // hid_t dset_id_attr_dim, dset_id_attr_storage;
    
    /* Can probably refactor this into an array */
    const char *dsn1 = "rho";
    const char *dsn2 = "rhovx";
    const char *dsn3 = "rhovy";
    const char *dsn4 = "rhovz";
    const char *dsn5 = "Bx";
    const char *dsn6 = "By";
    const char *dsn7 = "Bz";
    const char *dsn8 = "e";

    hsize_t cube_size = Nx * Ny * Nz;
    hsize_t dim[1] = {cube_size}; // 3D simulation data is stored in 1D  
    // hsize_t attrdim[1] = {3};
    // hsize_t attrstorage[1] = {1}; // Dimension of the storage pattern attribute
    // hsize_t cube_dimensions[3] = {Nx, Ny, Nz};
    
    herr_t status;

    // const char *dimension_names[3] = {"Nx", "Ny", "Nz"}; 
    // const char *storage_pattern[1] = {"Row-major, depth-minor: l = k * (Nx * Ny) + i * Ny + j"};
    
    std::cout << "filename is: " << file_name << std::endl;
    std::cout << "Where file is being written: " << file_name.data() << std::endl;

    // Creates an access template for the MPI communicator processes
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm, info);
    
    // Create the file
    file_id = H5Fcreate(file_name.data(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);

    // Create the dataspace, and dataset
    dspc_id = H5Screate_simple(1, dim, NULL);
    mspace = H5Screate_simple(1, dim, NULL);

    dset_id_rho = H5Dcreate(file_id, dsn1, H5T_NATIVE_FLOAT, dspc_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id_rhovx = H5Dcreate(file_id, dsn2, H5T_NATIVE_FLOAT, dspc_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id_rhovy = H5Dcreate(file_id, dsn3, H5T_NATIVE_FLOAT, dspc_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id_rhovz = H5Dcreate(file_id, dsn4, H5T_NATIVE_FLOAT, dspc_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id_Bx = H5Dcreate(file_id, dsn5, H5T_NATIVE_FLOAT, dspc_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id_By = H5Dcreate(file_id, dsn6, H5T_NATIVE_FLOAT, dspc_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id_Bz = H5Dcreate(file_id, dsn7, H5T_NATIVE_FLOAT, dspc_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id_e = H5Dcreate(file_id, dsn8, H5T_NATIVE_FLOAT, dspc_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    // Write to the dataset
    // Handles the case where num_procs != 8
    for (int irank = rank; irank < 8; irank += world_size){
        if (irank == 0){
            std::cout << "Process " << rank << " writing rho dataset" << std::endl;
            status = H5Dwrite(dset_id_rho, H5T_NATIVE_FLOAT, mspace, dspc_id, plist_id, output_data + irank * cube_size);
        }
        else if (irank == 1){
            std::cout << "Process " << rank << " writing rhovx dataset" << std::endl;
            status = H5Dwrite(dset_id_rhovx, H5T_NATIVE_FLOAT, mspace, dspc_id, plist_id, output_data + irank * cube_size);
        }
        else if (irank == 2){
            std::cout << "Process " << rank << " writing rhovy dataset" << std::endl;
            status = H5Dwrite(dset_id_rhovy, H5T_NATIVE_FLOAT, mspace, dspc_id, plist_id, output_data + irank * cube_size);
        }
        else if (irank == 3){
            std::cout << "Process " << rank << " writing rhovz dataset" << std::endl;
            status = H5Dwrite(dset_id_rhovz, H5T_NATIVE_FLOAT, mspace, dspc_id, plist_id, output_data + irank * cube_size);
        }
        else if (irank == 4){
            std::cout << "Process " << rank << " writing Bx dataset" << std::endl;
            status = H5Dwrite(dset_id_Bx, H5T_NATIVE_FLOAT, mspace, dspc_id, plist_id, output_data + irank * cube_size);
        }
        else if (irank == 5){
            std::cout << "Process " << rank << " writing By dataset" << std::endl;
            status = H5Dwrite(dset_id_By, H5T_NATIVE_FLOAT, mspace, dspc_id, plist_id, output_data + irank * cube_size);
        }
        else if (irank == 6){
            std::cout << "Process " << rank << " writing Bz dataset" << std::endl;
            status = H5Dwrite(dset_id_Bz, H5T_NATIVE_FLOAT, mspace, dspc_id, plist_id, output_data + irank * cube_size);
        }
        else if (irank == 7){
            std::cout << "Process " << rank << " writing e dataset" << std::endl;
            status = H5Dwrite(dset_id_e, H5T_NATIVE_FLOAT, mspace, dspc_id, plist_id, output_data + irank * cube_size);
        }
    }
    // MPI_Barrier(comm);

    status = H5Dclose(dset_id_rho);
    status = H5Dclose(dset_id_rhovx);
    status = H5Dclose(dset_id_rhovy);
    status = H5Dclose(dset_id_rhovz);
    status = H5Dclose(dset_id_Bx);
    status = H5Dclose(dset_id_By);
    status = H5Dclose(dset_id_Bz);
    status = H5Dclose(dset_id_e);

    std::cout << "Process: " << rank << " Closing property list" << std::endl;
    status = H5Pclose(plist_id);

    std::cout << "Process: " << rank << " Closing string type" << std::endl;
    // status = H5Tclose(strtype_id);
    
    std::cout << "Process: " << rank << " Closing attribute dimension" << std::endl;
    // status = H5Aclose(attrdim_id);
    
    std::cout << "Process: " << rank << " Closing dataspace, memspace, and attribute dataspace" << std::endl;
    status = H5Sclose(dspc_id);
    status = H5Sclose(mspace);
    // status = H5Sclose(attrdim_dspc_id);
    
    std::cout << "Process: " << rank << " Closing file" << std::endl;
    status = H5Fclose(file_id);

    if (rank == 0){
        std::cout << ".h5 file written" << std::endl;
    }
    
    MPI_Finalize();
    return;
}

