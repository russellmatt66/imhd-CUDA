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

#include "hdf5.h"
#include "mpi.h"

void writeH5FileAll(const std::string filename, const float* output_data, const int Nx, const int Ny, const int Nz);
// void createH5File(const std::string filename);

int main(int argc, char* argv[]){
    /* Parse arguments, and call writeH5FileAll with shared memory data */
    std::string filename = argv[1];
    int Nx = int(argv[2]);
    int Ny = int(argv[3]);
    int Nz = int(argv[4]);
    return 0;
}

/* It should work to just change file access mode from `H5F_ACC_TRUNC` to `H5F_ACC_RDWR` in `writeH5FileAll` */
// void createH5File(const std::string filename){

// }

void writeH5FileAll(const std::string filename, const float* output_data, const int Nx, const int Ny, const int Nz){
    MPI_Init(NULL, NULL);
    int world_size, rank;
    
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Info info = MPI_INFO_NULL;
    
    MPI_Comm_size(comm, &world_size); 
    MPI_Comm_rank(comm, &rank);

    std::cout << "Hello from process: " << rank << " out of " << world_size << std::endl;
    // printf("Hello from process %d out of %d\n", rank, world_size);

    hid_t plist_id;
    hid_t file_id, dspc_id;
    hid_t attrdim_id, attrdim_dspc_id;
    hid_t attrstorage_id, attrstorage_dspc_id;
    hid_t strtype_id;

    hid_t dset_id[8] = {0};

    hsize_t cube_size = Nx * Ny * Nz;
    hsize_t dim[1] = {cube_size}; // 3D simulation data is stored in 1D  
    hsize_t attrdim[1] = {3};
    hsize_t attrstorage[1] = {1};
    hsize_t cube_dimensions[3] = {Nx, Ny, Nz};
    
    herr_t status;

    const char *dimension_names[3] = {"Nx", "Ny", "Nz"}; 
    const char *storage_pattern[1] = {"Row-major, depth-minor: l = k * (Nx * Ny) + i * Ny + j"};
    
    std::string dset_name[8] = {""};

    std::cout << "filename is: " << filename << std::endl;
    std::cout << "Where file is being written: " << filename.data() << std::endl;

    // Creates an access template for the MPI communicator processes
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm, info);

    // Create the file
    file_id = H5Fcreate(filename.data(), H5F_ACC_RDWR, H5P_DEFAULT, plist_id);

    // Create the dataspace, and dataset
    dspc_id = H5Screate_simple(1, dim, NULL);

    for (int irank = rank; irank < 8; irank += world_size){
        dset_name[irank] = get_dset_name(irank);
        dset_id[irank] = H5Dcreate(file_id, dset_name[irank].data(), H5T_NATIVE_FLOAT, dspc_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    }

    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);

    // Write to the dataset
    for (int irank = rank; irank < 8; irank += world_size){
        status = H5Dwrite(dset_id[irank], H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, plist_id, output_data + irank * cube_size); // so that each process writes a different fluid variable
    
        // Add attribute for the dimensions of the data cube 
        attrdim_dspc_id = H5Screate_simple(1, attrdim, NULL);
        attrdim_id = H5Acreate(dset_id[irank], "cubeDimensions", H5T_NATIVE_FLOAT, attrdim_dspc_id, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Awrite(attrdim_id, H5T_NATIVE_FLOAT, cube_dimensions);

        // Add an attribute which names which dimension is which
        strtype_id = H5Tcopy(H5T_C_S1);
        H5Tset_size(strtype_id, H5T_VARIABLE);

        attrdim_id = H5Acreate(dset_id[irank], "cubeDimensionsNames", strtype_id, attrdim_dspc_id, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Awrite(attrdim_id, strtype_id, dimension_names);

        // Add an attribute for the storage pattern of the cube
        attrstorage_dspc_id = H5Screate_simple(1, attrstorage, NULL);
        attrstorage_id = H5Acreate(dset_id[irank], "storagePattern", strtype_id, attrstorage_dspc_id, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Awrite(attrstorage_id, strtype_id, storage_pattern);
    }

    // Close everything
    for (int irank = rank; irank < 8; irank += world_size){
        status = H5Dclose(dset_id[irank]);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    status = H5Pclose(plist_id);
    status = H5Tclose(strtype_id);
    status = H5Aclose(attrdim_id);
    status = H5Sclose(dspc_id);
    status = H5Sclose(attrdim_dspc_id);
    status = H5Fclose(file_id);

    MPI_Finalize();
    std::cout << ".h5 file written" << std::endl;
    return;
}

std::string get_dset_name(int rank){
    std::string dset_name = "";

    switch (rank){
        case 0:
            dset_name = "rho";
            break;
        case 1:
            dset_name = "rhovx";
            break;
        case 2:
            dset_name = "rhovy";
            break;
        case 3:
            dset_name = "rhovz";
            break;
        case 4: 
            dset_name = "Bx";
            break;
        case 5:
            dset_name = "By";
            break;
        case 6:
            dset_name = "Bz";
            break;
        case 7:
            dset_name = "e";
            break;
    }

    return dset_name;
}