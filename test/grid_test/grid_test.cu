#include "grid.cuh"
#include "grid.cu"

// // https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
// #define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
// {
//    if (code != cudaSuccess) 
//    {
//       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//       if (abort) exit(code);
//    }
// }

int main(int argc, char* argv[]){
    int Nx = atoi(argv[1]);
    int Ny = atoi(argv[2]);
    int Nz = atoi(argv[3]);

    Grid3D cartesian_grid = Grid3D(Nx, Ny, Nz);

    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    int num_threads_per_block_x = 1024;
    int num_threads_per_block_y = 1024;
    int num_threads_per_block_z = 1024;

    int num_blocks_x = numberOfSMs;
    int num_blocks_y = numberOfSMs;
    int num_blocks_z = numberOfSMs;

    dim3 grid_dimensions(num_blocks_x, num_blocks_y, num_blocks_z);
    dim3 block_dimensions(num_threads_per_block_x, num_threads_per_block_y, num_threads_per_block_z);

    float x_min = -1.0, x_max = 1.0, y_min = -1.0, y_max = 1.0, z_min = -1.0, z_max = 1.0;

    printf("Initializing grid\n");
    InitializeGrid<<<grid_dimensions, block_dimensions>>>(&cartesian_grid, x_min, x_max, y_min, y_max, z_min, z_max,
        Nx, Ny, Nz);
    checkCuda(cudaDeviceSynchronize());

    printf("Printing grid\n");
    PrintGrid<<<grid_dimensions, block_dimensions>>>(&cartesian_grid, Nx, Ny, Nz);
    checkCuda(cudaDeviceSynchronize());
    printf("Finished printing grid\n");

    freeGrid3D(&cartesian_grid);
    return 0;
}