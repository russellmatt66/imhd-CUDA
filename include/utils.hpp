#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <vector>

void writeGrid(const std::vector<std::string> file_names, 
    const float* x_grid, const float* y_grid, const float* z_grid, const size_t Nx, const size_t Ny, const size_t Nz);
    
void writeCSV(const std::string file_name, const float* x_grid, const float* y_grid, const float* z_grid, const size_t Nx, const size_t Ny, const size_t Nz,
    const size_t x_begin, const size_t x_end, const size_t y_begin, const size_t y_end, const size_t z_begin, const size_t z_end);

void writeFluidVars(const std::vector<std::string> file_names, 
    const float* rho, const float* rhov_z, 
    const size_t Nx, const size_t Ny, const size_t Nz);

void writeFVCSV(const std::string file_name, 
    const float* rho, const float* rhov_z, 
    const size_t Nx, const size_t Ny, const size_t Nz,
    const size_t x_begin, const size_t x_end, const size_t y_begin, const size_t y_end, const size_t z_begin, const size_t z_end);

#endif