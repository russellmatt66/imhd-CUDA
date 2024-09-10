/* 
Proof of Concept:
Write Initial Conditions (any device data) out with HDF5 
*/
#include <string>

#include "hdf5.h"

void writeH5(const std::string filename, const float* output_data, const int Nx, const int Ny, const int Nz);

int main(int argc, char* argv[]){
    // Specify CUDA boilerplate

    // Call Initial Conditions Kernel

    // Transfer Device data to Host

    // Write .h5 file out

    return 0;
}

/* This goes into library when finished */
void writeH5(const std::string filename, const float* output_data, const int Nx, const int Ny, const int Nz){
    
    hid_t file_id;
    hsize_t num_x = Nx, num_y = Ny, num_z = Nz;
    herr_t status;

    file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
    status = H5Fclose(file_id);
    return;
}