#include "phdf5.hpp"
#include "mpi.h"

#include <string>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <fstream>

// PHDF5 functions
void writeH5FileAll(const std::string filename, const float* output_data, const int Nx, const int Ny, const int Nz){
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
    
    std::cout << "filename is: " << filename << std::endl;
    std::cout << "Where file is being written: " << filename.data() << std::endl;

    // Creates an access template for the MPI communicator processes
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm, info);
    
    // Create the file
    file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
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
        if (rank == 0){
            std::cout << "Process " << rank << " writing rho dataset" << std::endl;
            status = H5Dwrite(dset_id_rho, H5T_NATIVE_FLOAT, mspace, dspc_id, plist_id, output_data + rank * cube_size);
        }
        else if (rank == 1){
            std::cout << "Process " << rank << " writing rhovx dataset" << std::endl;
            status = H5Dwrite(dset_id_rhovx, H5T_NATIVE_FLOAT, mspace, dspc_id, plist_id, output_data + rank * cube_size);
        }
        else if (rank == 2){
            std::cout << "Process " << rank << " writing rhovy dataset" << std::endl;
            status = H5Dwrite(dset_id_rhovy, H5T_NATIVE_FLOAT, mspace, dspc_id, plist_id, output_data + rank * cube_size);
        }
        else if (rank == 3){
            std::cout << "Process " << rank << " writing rhovz dataset" << std::endl;
            status = H5Dwrite(dset_id_rhovz, H5T_NATIVE_FLOAT, mspace, dspc_id, plist_id, output_data + rank * cube_size);
        }
        else if (rank == 4){
            std::cout << "Process " << rank << " writing Bx dataset" << std::endl;
            status = H5Dwrite(dset_id_Bx, H5T_NATIVE_FLOAT, mspace, dspc_id, plist_id, output_data + rank * cube_size);
        }
        else if (rank == 5){
            std::cout << "Process " << rank << " writing By dataset" << std::endl;
            status = H5Dwrite(dset_id_By, H5T_NATIVE_FLOAT, mspace, dspc_id, plist_id, output_data + rank * cube_size);
        }
        else if (rank == 6){
            std::cout << "Process " << rank << " writing Bz dataset" << std::endl;
            status = H5Dwrite(dset_id_Bz, H5T_NATIVE_FLOAT, mspace, dspc_id, plist_id, output_data + rank * cube_size);
        }
        else if (rank == 7){
            std::cout << "Process " << rank << " writing e dataset" << std::endl;
            status = H5Dwrite(dset_id_e, H5T_NATIVE_FLOAT, mspace, dspc_id, plist_id, output_data + rank * cube_size);
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

    MPI_Finalize();
    std::cout << ".h5 file written" << std::endl;
    return;
}

int callPHDF5(const std::string file_name, const int Nx, const int Ny, const int Nz, const std::string shm_name, const size_t data_size, const std::string num_proc, const std::string phdf5_bin_name){
    std::string mpirun_command = "mpirun -np " + num_proc + " ./" + phdf5_bin_name  
                                    + " " + file_name + " " + std::to_string(Nx) + " "
                                    + std::to_string(Ny) + " " + std::to_string(Nz) + " "
                                    + shm_name + " " + std::to_string(data_size);

    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        std::cout << "Current working directory: " << cwd << std::endl;
    } else {
       std::cerr << "Failed to get current working directory" << std::endl;
    }

    // Fork to PHDF5 output binary 
    std::cout << "Executing command: " << mpirun_command << std::endl;
    int ret = std::system(mpirun_command.data()); 
    return ret;
}

int callAttributes(const std::string file_name, const int Nx, const int Ny, const int Nz, const std::string attr_bin_name){
    std::string addatt_command = "./" + attr_bin_name + " " + file_name + " " + std::to_string(Nx) + " "
                                    + std::to_string(Ny) + " " + std::to_string(Nz);
    
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        std::cout << "Current working directory: " << cwd << std::endl;
    } else {
       std::cerr << "Failed to get current working directory" << std::endl;
    }

    std::cout << "Executing command: " << addatt_command << std::endl;
    int ret = std::system(addatt_command.data()); 
    return ret;
}

void writeAttributes(const std::string file_name, const int Nx, const int Ny, const int Nz){
    hid_t file_id, dset_id;

    herr_t status;

    file_id = H5Fopen(file_name.data(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error opening file: " << file_name << std::endl;
        return;
    }

    const char *dset_names[8] = {"rho", "rhovx", "rhovy", "rhovz", "Bx", "By", "Bz", "e"};

    for (int idset = 0; idset < 8; idset++){
        std::cout << "Opening dataset " << dset_names[idset] << " for attribute writing" << std::endl;
        dset_id = H5Dopen(file_id, dset_names[idset], H5P_DEFAULT);
        if (dset_id < 0) {
            std::cerr << "Error opening dataset: " << dset_names[idset] << std::endl;
            status = H5Fclose(file_id);
            return;
        }
        addAttributes(dset_id, Nx, Ny, Nz);
        verifyAttributes(dset_id);
        // status = H5Dclose(dset_id);
    }

    std::cout << "Closing dset_id, and file_id" << std::endl;
    status = H5Dclose(dset_id); 
    status = H5Fclose(file_id);
    return;
}

void addAttributes(const hid_t dset_id, const int Nx, const int Ny, const int Nz){
    hid_t attrdim_id, attrdim_dspc_id;
    hid_t attrstorage_id, attrstorage_dspc_id;
    hid_t strtype_id;

    hsize_t attrdim[1] = {3};
    hsize_t attrstorage[1] = {1}; // Dimension of the storage pattern attribute
    hsize_t cube_dimensions[3] = {Nx, Ny, Nz};

    herr_t status;

    const char *dimension_names[3] = {"Nx", "Ny", "Nz"}; 
    const char *storage_pattern[1] = {"Row-major, depth-minor: l = k * (Nx * Ny) + i * Ny + j"};

    // Need to store dimensionality of data
    attrdim_dspc_id = H5Screate_simple(1, attrdim, NULL);
    attrdim_id = H5Acreate(dset_id, "cubeDimensions", H5T_NATIVE_INT, attrdim_dspc_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attrdim_id, H5T_NATIVE_INT, cube_dimensions);    
    if (status < 0) {
        std::cerr << "Error writing attribute 'cubeDimensions'" << std::endl;
    }

    // To add an attribute that names which variable is which
    strtype_id = H5Tcopy(H5T_C_S1);
    H5Tset_size(strtype_id, H5T_VARIABLE);

    attrdim_id = H5Acreate(dset_id, "cubeDimensionsNames", strtype_id, attrdim_dspc_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attrdim_id, strtype_id, dimension_names);
    if (status < 0) {
        std::cerr << "Error writing attribute 'cubeDimensionsNames'" << std::endl;
    }

    // Lastly, need to add an attribute for the storage pattern of the cube
    attrstorage_dspc_id = H5Screate_simple(1, attrstorage, NULL);
    attrstorage_id = H5Acreate(dset_id, "storagePattern", strtype_id, attrstorage_dspc_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attrstorage_id, strtype_id, storage_pattern);
    if (status < 0) {
        std::cerr << "Error writing attribute 'storagePattern'" << std::endl;
    }

    status = H5Tclose(strtype_id);
    status = H5Aclose(attrdim_id);
    status = H5Aclose(attrstorage_id);
    status = H5Sclose(attrdim_dspc_id);
    status = H5Sclose(attrstorage_dspc_id);
    return;
}

void verifyAttributes(const hid_t dset_id){
    htri_t exists;

    exists = H5Aexists(dset_id, "cubeDimensions");
    std::cout << "Attribute 'cubeDimensions' exists: " << (exists ? "Yes" : "No") << std::endl;

    exists = H5Aexists(dset_id, "cubeDimensionsNames");
    std::cout << "Attribute 'cubeDimensionsNames' exists: " << (exists ? "Yes" : "No") << std::endl;

    exists = H5Aexists(dset_id, "storagePattern");
    std::cout << "Attribute 'storagePattern' exists: " << (exists ? "Yes" : "No") << std::endl;
}