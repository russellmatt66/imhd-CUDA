#ifndef PHDF5_HPP
#define PHDF5_HPP

#include <string>

#include "hdf5.h"

// PHDF5 functions
void writeH5FileAll(const std::string file_name, const float* output_data, const int Nx, const int Ny, const int Nz);
int callPHDF5(const std::string file_name, const int Nx, const int Ny, const int Nz, const std::string shm_name, const size_t data_size, const std::string num_proc, const std::string phdf5_bin_name);
int callAttributes(const std::string file_name, const int Nx, const int Ny, const int Nz, const std::string attr_bin_name);
void writeAttributes(const std::string file_name, const int Nx, const int Ny, const int Nz);
void addAttributes(const hid_t dset_id, const int Nx, const int Ny, const int Nz);
void verifyAttributes(const hid_t dset_id);

#endif