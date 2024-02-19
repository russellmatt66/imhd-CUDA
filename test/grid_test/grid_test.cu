#include "grid.cuh"
#include "grid.cu"

/* 
For some reason, I cannot get the struct to work.  
I have tried multiple times, and one of two things occurs:
(1) PrintGrid does not print anything
(2) Segfault
The current version segfaults, which is due to the allocations I'm making.

The correct approach is to allocate the structure on host, and then copy it to device. This results in (1).

The solution is to just use linear arrays for everything, or try to get the simplest possible case with a single pointer working.
 */
int main(int argc, char* argv[]){
    int Nx = atoi(argv[1]);
    int Ny = atoi(argv[2]);
    int Nz = atoi(argv[3]);

    float x_min = -1.0, x_max = 1.0, y_min = -1.0, y_max = 1.0, z_min = -1.0, z_max = 1.0;

    float dx = (x_max - x_min) / (Nx - 1);
    float dy = (y_max - y_min) / (Ny - 1);
    float dz = (z_max - z_min) / (Nz - 1);

    // Grid3D cartesian_grid = Grid3D(Nx, Ny, Nz);
    Grid3D* cartesian_grid;
    cartesian_grid = (Grid3D*)malloc(sizeof(Grid3D));
    createGrid3D(cartesian_grid, Nx, Ny, Nz); // allocates memory to structure pointers

    for (int i = 0; i < Nx; i++){
        cartesian_grid->x_[i] = x_min + i * dx;
    }
    for (int j = 0; j < Ny; j++){
        cartesian_grid->y_[j] = y_min + j * dy;
    }
    for (int k = 0; k < Nz; k++){
        cartesian_grid->z_[k] = z_min + k * dz;
    }

    Grid3D* d_cartesian_grid;
    checkCuda(cudaMalloc(&d_cartesian_grid, sizeof(Grid3D)));
    checkCuda(cudaMalloc(&(d_cartesian_grid->x_), sizeof(float) * Nx));
    checkCuda(cudaMalloc(&(d_cartesian_grid->y_), sizeof(float) * Ny));
    checkCuda(cudaMalloc(&(d_cartesian_grid->z_), sizeof(float) * Nz));

    // cudaMemcpy(d_cartesian_grid, cartesian_grid, sizeof(Grid3D), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cartesian_grid->x_, cartesian_grid->x_, Nx * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cartesian_grid->y_, cartesian_grid->y_, Ny * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cartesian_grid->z_, cartesian_grid->z_, Nz * sizeof(float), cudaMemcpyHostToDevice);

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

    // printf("Initializing grid\n");
    // InitializeGrid<<<grid_dimensions, block_dimensions>>>(d_cartesian_grid, x_min, x_max, y_min, y_max, z_min, z_max,
    //     Nx, Ny, Nz);
    // checkCuda(cudaDeviceSynchronize());

    printf("Printing grid\n");
    PrintGrid<<<grid_dimensions, block_dimensions>>>(d_cartesian_grid, Nx, Ny, Nz);
    checkCuda(cudaDeviceSynchronize());
    printf("Finished printing grid\n");

    freeGrid3D(cartesian_grid);
    return 0;
}