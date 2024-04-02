#ifndef GDS_FNC_CUH
#define GDS_FNC_CUH

void writeDataGDS(const char* filename, const float* data, const int size);
void writeGridBasisGDS(const char* filename, const float* x_grid, const float* y_grid, const float* z_grid, const int Nx, const int Ny, const int Nz);
void writeGridGDS(const char* filename, const float* grid_data, const int Nx, const int Ny, const int Nz);

__global__ void WriteGridBuffer(float* buffer, const float* x_grid, const float* y_grid, const float* z_grid, const int Nx, const int Ny, const int Nz);

#endif