/* 
Proof of Concept:
Write Initial Conditions (any device data) out with a fork to PHDF5 function
*/
#include <string>
#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include "../../../include/initialize_od.cuh"

void callPHDF5(const std::string filename, const int Nx, const int Ny, const int Nz, const std::string shm_name, const size_t data_size, const std::string num_proc);

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
    InitialConditions<<<execution_grid_dimensions, block_dims_init>>>(fluidvar, J0, grid_x, grid_y, grid_z, Nx, Ny, Nz); // Screw-pinch
    checkCuda(cudaDeviceSynchronize());

    // TRANSFER DEVICE DATA TO SHARED MEMORY REGION
    std::string shm_name = "/shared_h_fluidvar";
    int shm_fd = shm_open(shm_name.data(), O_CREAT | O_RDWR, 0666);
        if (shm_fd == -1) {
        std::cerr << "Failed to create shared memory!" << std::endl;
        return EXIT_FAILURE;
    }
    ftruncate(shm_fd, 8 * fluid_data_size);
    float* shm_h_fluidvar = (float*)mmap(0, fluid_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_h_fluidvar == MAP_FAILED) {
        std::cerr << "mmap failed!" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Transferring device data to host" << std::endl;
    cudaMemcpy(shm_h_fluidvar, fluidvar, fluid_data_size, cudaMemcpyDeviceToHost);
    checkCuda(cudaDeviceSynchronize());

    // Write .h5 file out
    std::cout << "path_to_data: " << path_to_data << std::endl;
    path_to_data += "fluid_data.h5";
    std::cout << "path_to_data being passed to writeH5File(): " << path_to_data << std::endl;

    /* FORK THE PHDF5 FUNCTION WITH MPIRUN SYSTEM CALL */
    std::cout << "Writing .h5 file" << std::endl;
    callPHDF5(path_to_data, Nx, Ny, Nz, shm_name, fluid_data_size, num_proc);

    // Free data
    cudaFree(fluidvar);
    cudaFree(grid_x);
    cudaFree(grid_y);
    cudaFree(grid_z);
    munmap(shm_h_fluidvar, 8 * fluid_data_size);
    shm_unlink(shm_name.data());
    return 0;
}

void callPHDF5(const std::string filename, const int Nx, const int Ny, const int Nz, 
                const std::string shm_name, const size_t data_size, 
                const std::string num_proc, const std::string phdf5_bin_name){
    std::string mpirun_command = "mpirun -np " + num_proc + " ./" + phdf5_bin_name  
                                    + " " + filename + " " + std::to_string(Nx) + " "
                                    + std::to_string(Ny) + " " + std::to_string(Nz) + " "
                                    + " " + shm_name + " " + std::to_string(data_size);

    /* Fork to PHDF5 output binary */ 
    return;
}