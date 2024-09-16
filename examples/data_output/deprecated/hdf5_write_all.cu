/* 
Proof of Concept:
Write Initial Conditions (any device data) out with PHDF5 
SEE COMMENT ABOVE `writeH5FileAll` definition
*/
#include <string>
#include <iostream>
#include <fstream>

#include "../../../include/initialize_od.cuh"
#include "hdf5.h"
#include "mpi.h"

// Writes the data cubes of all the fluid variables using PHDF5
// void writeH5FileAll(const std::string filename, const float* output_data, const int Nx, const int Ny, const int Nz, int argc, char* argv[]);
void writeH5FileAll(const std::string filename, const float* output_data, const int Nx, const int Ny, const int Nz);
std::string get_dset_name(int rank);

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
    InitialConditions<<<execution_grid_dimensions, block_dims_init>>>(fluidvar, J0, grid_x, grid_y, grid_z, Nx, Ny, Nz); // Screw-pinch
    checkCuda(cudaDeviceSynchronize());

    // Transfer Device data to Host
    float* h_fluidvar;

    h_fluidvar = (float*)malloc(8 * fluid_data_size);
    
    std::cout << "Transferring device data to host" << std::endl;
    cudaMemcpy(h_fluidvar, fluidvar, 8 * fluid_data_size, cudaMemcpyDeviceToHost);
    checkCuda(cudaDeviceSynchronize());

    // Write .h5 file out
    std::cout << "path_to_data: " << path_to_data << std::endl;
    path_to_data += "fluid_data.h5";
    std::cout << "path_to_data being passed to writeH5File(): " << path_to_data << std::endl;

    std::cout << "Writing .h5 file" << std::endl;
    /* NEED TO PUT IN OWN BINARY, MPIRUN IS A FUNDAMENTAL ISSUE */
    // MPI_Init(&argc, &argv);
    // writeH5FileAll(path_to_data, h_fluidvar, Nx, Ny, Nz);
    // writeH5FileAll(path_to_data, h_fluidvar, Nx, Ny, Nz, argc, argv);
    // MPI_Finalize();

    // Free data
    cudaFree(fluidvar);
    cudaFree(grid_x);
    cudaFree(grid_y);
    cudaFree(grid_z);
    free(h_fluidvar);
    return 0;
}

/* 
PROOF OF CONCEPT for parallel library function 
FUNDAMENTAL PROBLEM is needing to call mpirun on the binary representing the CUDA code
- No way to refactor because each MPI child process will run the main driver code with all the CUDA kernels
- Only way would be to place the calls inside of an 'if (rank == 0) {}' block  
- This is an ugly solution, and it also requires additional synchronization and broadcasting of the simulation data to the processes
SOLUTION is to put this code in its own binary, and then call that from within CUDA code, with the simulation data copied from device to a shared memory region 
*/
// void writeH5FileAll(const std::string filename, const float* output_data, const int Nx, const int Ny, const int Nz, int argc, char* argv[]){
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
    file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);

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