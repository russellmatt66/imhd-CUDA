/* 
Proof of Concept:
Write Initial Conditions (any device data) out with HDF5 
*/
#include <string>
#include <iostream>

#include "../../../include/initialize_od.cuh"
#include "hdf5.h"

// Writes the data cube of a single fluid variable
void writeH5File(const std::string filename, const float* output_data, const int Nx, const int Ny, const int Nz);

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char* argv[]){
    // Launch with Python
    std::cout << "Inside h5write_serial" << std::endl;

    std::string path_to_data = argv[1];
    int Nx = atoi(argv[2]);
	int Ny = atoi(argv[3]);
	int Nz = atoi(argv[4]);
	int SM_mult_x = atoi(argv[5]);
	int SM_mult_y = atoi(argv[6]);
	int SM_mult_z = atoi(argv[7]);
	int num_threads_per_block_x = atoi(argv[8]);
	int num_threads_per_block_y = atoi(argv[9]);
	int num_threads_per_block_z = atoi(argv[10]);
	float J0 = atof(argv[11]);
	float D = atof(argv[12]);
	float x_min = atof(argv[13]);
	float x_max = atof(argv[14]);
	float y_min = atof(argv[15]);
	float y_max = atof(argv[16]);
	float z_min = atof(argv[17]);
	float z_max = atof(argv[18]);

    float dx = (x_max - x_min) / (Nx - 1);
	float dy = (y_max - y_min) / (Ny - 1);
	float dz = (z_max - z_min) / (Nz - 1);

    // CUDA boilerplate
    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	float *fluidvar;
	float *grid_x, *grid_y, *grid_z;

    int cube_size = Nx * Ny * Nz;
	int fluid_data_size = sizeof(float) * Nx * Ny * Nz;

	checkCuda(cudaMalloc(&fluidvar, 8 * fluid_data_size));
    checkCuda(cudaMalloc(&grid_x, sizeof(float) * Nx));
	checkCuda(cudaMalloc(&grid_y, sizeof(float) * Ny));
	checkCuda(cudaMalloc(&grid_z, sizeof(float) * Nz));

	dim3 execution_grid_dimensions(SM_mult_x * numberOfSMs, SM_mult_y * numberOfSMs, SM_mult_z * numberOfSMs);
	dim3 block_dims_grid(32, 16, 2); // 1024 threads per block
	dim3 block_dims_init(8, 4, 4); // 256 < 923 threads per block

    // Call Initial Conditions Kernel
    std::cout << "Initializing grid" << std::endl;
    InitializeGrid<<<execution_grid_dimensions, block_dims_grid>>>(x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz,
															grid_x, grid_y, grid_z, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());

    std::cout << "Initializing simulation data" << std::endl;
	ScrewPinch<<<execution_grid_dimensions, block_dims_init>>>(fluidvar, J0, grid_x, grid_y, grid_z, Nx, Ny, Nz); // Screw-pinch
    checkCuda(cudaDeviceSynchronize());

    // Transfer Device data to Host
    float* h_fluidvar;

    h_fluidvar = (float*)malloc(fluid_data_size);
    
    std::cout << "Transferring device data to host" << std::endl;
    cudaMemcpy(h_fluidvar, fluidvar, fluid_data_size, cudaMemcpyDeviceToHost);
    checkCuda(cudaDeviceSynchronize());

    // Write .h5 file out
    std::cout << "path_to_data: " << path_to_data << std::endl;
    path_to_data += "fluid_data.h5";
    std::cout << "path_to_data being passed to writeH5File(): " << path_to_data << std::endl;

    std::cout << "Writing .h5 file" << std::endl;
    writeH5File(path_to_data, h_fluidvar, Nx, Ny, Nz);

    // Free data
    cudaFree(fluidvar);
    cudaFree(grid_x);
    cudaFree(grid_y);
    cudaFree(grid_z);
    free(h_fluidvar);
    return 0;
}

/* 
Proof of Concept for library function 

Library version will need to store all the data
*/
void writeH5File(const std::string filename, const float* output_data, const int Nx, const int Ny, const int Nz){
    hid_t file_id, dset_id, dspc_id;
    hid_t attrdim_id, attrdim_dspc_id;
    hid_t attrstorage_id, attrstorage_dspc_id;
    hid_t strtype_id;
    
    hsize_t dim[1] = {Nx * Ny * Nz}; // 3D simulation data is stored in 1D  
    hsize_t attrdim[1] = {3};
    hsize_t attrstorage[1] = {1};
    hsize_t cube_dimensions[3] = {Nx, Ny, Nz};
    
    herr_t status;

    const char *dimension_names[3] = {"Nx", "Ny", "Nz"}; 
    const char *storage_pattern[1] = {"Row-major, depth-minor: l = k * (Nx * Ny) + i * Ny + j"};

    std::cout << "filename is: " << filename << std::endl;
    std::cout << "Where file is being written: " << filename.data() << std::endl;

    // Create the file
    file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
    // Create the dataspace, and dataset
    dspc_id = H5Screate_simple(1, dim, NULL);
    dset_id = H5Dcreate(file_id, "fluid_data", H5T_NATIVE_FLOAT, dspc_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    // Write to the dataset
    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, output_data);

    // Add attribute for the dimensions of the data cube 
    attrdim_dspc_id = H5Screate_simple(1, attrdim, NULL);
    attrdim_id = H5Acreate(dset_id, "cubeDimensions", H5T_NATIVE_FLOAT, attrdim_dspc_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attrdim_id, H5T_NATIVE_FLOAT, cube_dimensions);

    // Add an attribute which names which dimension is which
    strtype_id = H5Tcopy(H5T_C_S1);
    H5Tset_size(strtype_id, H5T_VARIABLE);

    attrdim_id = H5Acreate(dset_id, "cubeDimensionsNames", strtype_id, attrdim_dspc_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attrdim_id, strtype_id, dimension_names);

    // Add an attribute for the storage pattern of the cube
    attrstorage_dspc_id = H5Screate_simple(1, attrstorage, NULL);
    attrstorage_id = H5Acreate(dset_id, "storagePattern", strtype_id, attrstorage_dspc_id, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attrstorage_id, strtype_id, storage_pattern);

    // Close everything
    status = H5Tclose(strtype_id);
    status = H5Aclose(attrdim_id);
    status = H5Dclose(dset_id);
    status = H5Sclose(dspc_id);
    status = H5Sclose(attrdim_dspc_id);
    status = H5Fclose(file_id);
    
    std::cout << ".h5 file written" << std::endl;
    return;
}