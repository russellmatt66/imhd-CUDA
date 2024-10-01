/* 
Proof of Concept:
Write Initial Conditions (any device data) out with a fork to PHDF5 function
*/
#include <string>
#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>
#include <vector>
#include <cstdlib>

#include "hdf5.h"
#include "../../../include/initialize_od.cuh"

int callPHDF5(const std::string filename, const int Nx, const int Ny, const int Nz, const std::string shm_name, const size_t data_size, const std::string num_proc, const std::string phdf5_bin_name);
int callAttributes(const std::string file_name, const int Nx, const int Ny, const int Nz, const std::string attr_binary);
void writeAttributes(const std::string filename, const int Nx, const int Ny, const int Nz);
void addAttributes(hid_t dset_id, const int Nx, const int Ny, const int Nz);
void verifyAttributes(hid_t dset_id);

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
    if (argc < 19) {
        std::cerr << "Error: Insufficient arguments provided!" << std::endl;
        std::cerr << "Usage: <path_to_data> <Nx> <Ny> <Nz> <SM_mult_x> <SM_mult_y> <SM_mult_z> <num_threads_per_block_x> <num_threads_per_block_y> <num_threads_per_block_z> <J0> <D> <x_min> <x_max> <y_min> <y_max> <z_min> <z_max>" << std::endl;
        return EXIT_FAILURE;
    }

    // Launch with Python
    std::cout << "Inside h5write_par" << std::endl;

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
    std::string num_proc = argv[19];
    std::string phdf5_bin_name = argv[20];
    std::string attr_bin_name = argv[21];

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

    size_t cube_size = Nx * Ny * Nz;
    size_t fluid_var_size = sizeof(float) * cube_size;
    size_t fluid_data_size = 8 * fluid_var_size;

    checkCuda(cudaMalloc(&fluidvar, fluid_data_size));
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

    // TRANSFER DEVICE DATA TO SHARED MEMORY REGION
    std::string shm_name = "/shared_h_fluidvar";
    int shm_fd = shm_open(shm_name.data(), O_CREAT | O_RDWR, 0666);
        if (shm_fd == -1) {
        std::cerr << "Failed to create shared memory!" << std::endl;
        return EXIT_FAILURE;
    }
    ftruncate(shm_fd, fluid_data_size);
    float* shm_h_fluidvar = (float*)mmap(0, fluid_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_h_fluidvar == MAP_FAILED) {
        std::cerr << "mmap failed!" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Transferring device data to host" << std::endl;
    cudaMemcpy(shm_h_fluidvar, fluidvar, fluid_data_size, cudaMemcpyDeviceToHost);
    checkCuda(cudaDeviceSynchronize());

    // Write .h5 file out
    std::cout << "Writing .h5 file" << std::endl;
    std::cout << "path_to_data: " << path_to_data << std::endl;
    path_to_data += "fluid_data.h5";
    std::cout << "path_to_data being passed to writeH5File(): " << path_to_data << std::endl;

    // Fork PHDF5 dataset writing with mpirun system call 
    std::cout << "Writing datasets with PHDF5" << std::endl;
    int ret = callPHDF5(path_to_data, Nx, Ny, Nz, shm_name, fluid_data_size, num_proc, phdf5_bin_name);
    if (ret != 0) {
        std::cerr << "Error executing PHDF5 command" << std::endl;
    }

    // Write attributes serially to datasets - PHDF5 attribute writing is novice trap AFAIK
    std::cout << "Writing dataset attributes with HDF5" << std::endl;
    ret = callAttributes(path_to_data, Nx, Ny, Nz, attr_bin_name); // trying to write attributes in current context doesn't work
    if (ret != 0) {
        std::cerr << "Error executing attribute command" << std::endl;
    }

    // Free data
    cudaFree(fluidvar);
    cudaFree(grid_x);
    cudaFree(grid_y);
    cudaFree(grid_z);
    munmap(shm_h_fluidvar, 8 * fluid_data_size);
    shm_unlink(shm_name.data());
    return 0;
}

// PHDF5 needs to be executed in MPI environment
int callPHDF5(const std::string file_name, const int Nx, const int Ny, const int Nz, 
                const std::string shm_name, const size_t data_size, 
                const std::string num_proc, const std::string phdf5_bin_name){
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

// Running the code in this binary in the context of this process does not seem to add attributes to file_name, so give it a separate process
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

// The below was moved to a separate binary because the attributes weren't actually getting written from the current context
// void addAttributes(hid_t dset_id, const int Nx, const int Ny, const int Nz){
//     hid_t attrdim_id, attrdim_dspc_id;
//     hid_t attrstorage_id, attrstorage_dspc_id;
//     hid_t strtype_id;

//     hsize_t attrdim[1] = {3};
//     hsize_t attrstorage[1] = {1}; // Dimension of the storage pattern attribute
//     hsize_t cube_dimensions[3] = {Nx, Ny, Nz};

//     herr_t status;

//     const char *dimension_names[3] = {"Nx", "Ny", "Nz"}; 
//     const char *storage_pattern[1] = {"Row-major, depth-minor: l = k * (Nx * Ny) + i * Ny + j"};

//     // Need to store dimensionality of data
//     attrdim_dspc_id = H5Screate_simple(1, attrdim, NULL);
//     attrdim_id = H5Acreate(dset_id, "cubeDimensions", H5T_NATIVE_INT, attrdim_dspc_id, H5P_DEFAULT, H5P_DEFAULT);
//     status = H5Awrite(attrdim_id, H5T_NATIVE_INT, cube_dimensions);    
//     if (status < 0) {
//         std::cerr << "Error writing attribute 'cubeDimensions'" << std::endl;
//     }

//     // To add an attribute that names which variable is which
//     strtype_id = H5Tcopy(H5T_C_S1);
//     H5Tset_size(strtype_id, H5T_VARIABLE);

//     attrdim_id = H5Acreate(dset_id, "cubeDimensionsNames", strtype_id, attrdim_dspc_id, H5P_DEFAULT, H5P_DEFAULT);
//     status = H5Awrite(attrdim_id, strtype_id, dimension_names);
//     if (status < 0) {
//         std::cerr << "Error writing attribute 'cubeDimensionsNames'" << std::endl;
//     }

//     // Lastly, need to add an attribute for the storage pattern of the cube
//     attrstorage_dspc_id = H5Screate_simple(1, attrstorage, NULL);
//     attrstorage_id = H5Acreate(dset_id, "storagePattern", strtype_id, attrstorage_dspc_id, H5P_DEFAULT, H5P_DEFAULT);
//     status = H5Awrite(attrstorage_id, strtype_id, storage_pattern);
//     if (status < 0) {
//         std::cerr << "Error writing attribute 'storagePattern'" << std::endl;
//     }

//     status = H5Tclose(strtype_id);
//     status = H5Aclose(attrdim_id);
//     status = H5Aclose(attrstorage_id);
//     status = H5Sclose(attrdim_dspc_id);
//     status = H5Sclose(attrstorage_dspc_id);
//     return;
// }

// // PHDF5 attribute writing is a novice trap AFAIK
// void writeAttributes(const std::string filename, const int Nx, const int Ny, const int Nz){
//     hid_t file_id, dset_id;

//     herr_t status;

//     file_id = H5Fopen(filename.data(), H5F_ACC_RDWR, H5P_DEFAULT);
//     if (file_id < 0) {
//         std::cerr << "Error opening file: " << filename << std::endl;
//         return;
//     }

//     const char *dset_names[8] = {"rho", "rhovx", "rhovy", "rhovz", "Bx", "By", "Bz", "e"};

//     for (int idset = 0; idset < 8; idset++){
//         std::cout << "Opening dataset " << dset_names[idset] << " for attribute writing" << std::endl;
//         dset_id = H5Dopen(file_id, dset_names[idset], H5P_DEFAULT);
//         if (dset_id < 0) {
//             std::cerr << "Error opening dataset: " << dset_names[idset] << std::endl;
//             status = H5Fclose(file_id);
//             return;
//         }
//         addAttributes(dset_id, Nx, Ny, Nz);
//         verifyAttributes(dset_id);
//         // status = H5Dclose(dset_id);
//     }

//     std::cout << "Closing dset_id, and file_id" << std::endl;
//     status = H5Dclose(dset_id); 
//     status = H5Fclose(file_id);
//     return;
// }

// void verifyAttributes(hid_t dset_id){
//     htri_t exists;

//     exists = H5Aexists(dset_id, "cubeDimensions");
//     std::cout << "Attribute 'cubeDimensions' exists: " << (exists ? "Yes" : "No") << std::endl;

//     exists = H5Aexists(dset_id, "cubeDimensionsNames");
//     std::cout << "Attribute 'cubeDimensionsNames' exists: " << (exists ? "Yes" : "No") << std::endl;

//     exists = H5Aexists(dset_id, "storagePattern");
//     std::cout << "Attribute 'storagePattern' exists: " << (exists ? "Yes" : "No") << std::endl;
// }
