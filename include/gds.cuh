#ifndef GDS_FNC_CUH
#define GDS_FNC_CUH

// GPU Direct Storage (GDS) writes/reads data to/from storage directly from the GPU

void writeDataGDS(const char* filename, const float* data, const int size);
void writeFluidDataGDS(const float* rho, const float* rhov_x, const float* rhov_y, const float* rhov_z, 
    const float* Bx, const float* By, const float* Bz, const float* e, 
    const int size, const int nt);

void writeGridBasisGDS(const char* filename, const float* x_grid, const float* y_grid, const float* z_grid, const int Nx, const int Ny, const int Nz);
// Writes pattern for writeGridGDS
__global__ void WriteGridBuffer(float* buffer, const float* x_grid, const float* y_grid, const float* z_grid, const int Nx, const int Ny, const int Nz);
void writeGridGDS(const char* filename, const float* grid_data, const int Nx, const int Ny, const int Nz);

#endif