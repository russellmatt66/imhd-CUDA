/*
Writes the dimensionality, and storage pattern of the simulation datasets to the .h5 file created by `phdf5_write_all.cpp` 
*/
#include <iostream>

#include "hdf5.h"

void writeAttributes(const std::string filename, const int Nx, const int Ny, const int Nz);
void addAttributes(hid_t dset_id, const int Nx, const int Ny, const int Nz);
void verifyAttributes(hid_t dset_id);

int main(int argc, char* argv[]){
    std::string file_name = argv[1];
    int Nx = atoi(argv[2]);
    int Ny = atoi(argv[3]);
    int Nz = atoi(argv[4]);

    writeAttributes(file_name, Nx, Ny, Nz);
    return 0;
}

// PHDF5 attribute writing is a novice trap AFAIK
void writeAttributes(const std::string filename, const int Nx, const int Ny, const int Nz){
    hid_t file_id, dset_id;

    herr_t status;

    file_id = H5Fopen(filename.data(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error opening file: " << filename << std::endl;
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

void addAttributes(hid_t dset_id, const int Nx, const int Ny, const int Nz){
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

void verifyAttributes(hid_t dset_id){
    htri_t exists;

    exists = H5Aexists(dset_id, "cubeDimensions");
    std::cout << "Attribute 'cubeDimensions' exists: " << (exists ? "Yes" : "No") << std::endl;

    exists = H5Aexists(dset_id, "cubeDimensionsNames");
    std::cout << "Attribute 'cubeDimensionsNames' exists: " << (exists ? "Yes" : "No") << std::endl;

    exists = H5Aexists(dset_id, "storagePattern");
    std::cout << "Attribute 'storagePattern' exists: " << (exists ? "Yes" : "No") << std::endl;
}
