/* 
Proof of Concept:
Compute eigenvalues of the Flux Jacobians for a given state 
*/
#include <string>
#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>

#include "../../../include/initialize_od.cuh"

int callEigen(const std::string shm_name, const int Nx, const int Ny, const int Nz, const std::string bin_name);

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
    std::cout << "Inside fj_evs_cu" << std::endl;

    std::string eigen_bin_name = argv[1];
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
	int fluid_var_size = sizeof(float) * Nx * Ny * Nz;
    int fluid_data_size = 8 * fluid_var_size;

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
    ftruncate(shm_fd, fluid_data_size);
    float* shm_h_fluidvar = (float*)mmap(0, fluid_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_h_fluidvar == MAP_FAILED) {
        std::cerr << "mmap failed!" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Transferring device data to host" << std::endl;
    cudaMemcpy(shm_h_fluidvar, fluidvar, fluid_data_size, cudaMemcpyDeviceToHost);
    checkCuda(cudaDeviceSynchronize());

    std::cout << "Forking to flux jacobian eigenvalue computation binary" << std::endl;
    int ret = callEigen(shm_name, Nx, Ny, Nz, eigen_bin_name);
    if (ret != 0) {
        std::cerr << "Error executing Eigen command" << std::endl;
    }

    // Free data
    std::cout << "Freeing data" << std::endl;
    cudaFree(fluidvar);
    cudaFree(grid_x);
    cudaFree(grid_y);
    cudaFree(grid_z);
    munmap(shm_h_fluidvar, 8 * fluid_data_size);
    shm_unlink(shm_name.data());  
    return 0;
}

int callEigen(const std::string shm_name, const int Nx, const int Ny, const int Nz, const std::string bin_name){
    std::string eigen_command = "./" + bin_name + " " + shm_name + " " + std::to_string(Nx) + " "
                                    + std::to_string(Ny) + " " + std::to_string(Nz);
    
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        std::cout << "Current working directory: " << cwd << std::endl;
    } else {
       std::cerr << "Failed to get current working directory" << std::endl;
    }

    std::cout << "Executing command: " << eigen_command << std::endl;
    int ret = std::system(eigen_command.data()); 
    return ret;
    return 0;
}