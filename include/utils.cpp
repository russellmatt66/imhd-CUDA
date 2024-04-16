#include <thread>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>

#include "utils.hpp"

#define IDX3D(i, j, k, Nx, Ny, Nz) (k * (Nx * Ny) + i * Ny + j) // row-major, column-minor order

void writeGrid(const std::vector<std::string> file_names, const float* x_grid, const float* y_grid, const float* z_grid, const size_t Nx, const size_t Ny, const size_t Nz){
    /* 
    To avoid issues with asynchronous execution, e.g., needing a semaphore, each thread is going to write to its own `.csv` file
    These files will then be merged together with cuDF
    */
    if (file_names.size() != 8){
        std::cout << "Not providing the right number of file names to writeGrid" << std::endl;
        std::cout << "The correct number is 8" << std::endl;
    }
    /* 
    DOESN'T WRITE ALL THE DATA 
    Each thread writes a cube of data, the union of which does not form a complete representation of the original data
    The solution is to partition the workload into slabs so that each thread completely writes a given number of the xy-planes
    */
    // std::thread w1(writeCSV, file_names[0], x_grid, y_grid, z_grid, Nx, Ny, Nz, 0, Nx / 8, 0, Ny / 8, 0, Nz / 8);
    // std::thread w2(writeCSV, file_names[1], x_grid, y_grid, z_grid, Nx, Ny, Nz, Nx / 8, Nx / 4, Ny / 8, Ny / 4, Nz / 8, Nz / 4);
    // std::thread w3(writeCSV, file_names[2], x_grid, y_grid, z_grid, Nx, Ny, Nz, Nx / 4, 3 * Nx / 8, Ny / 4, 3 * Ny / 8, Nz / 4, 3 * Nz / 8);
    // std::thread w4(writeCSV, file_names[3], x_grid, y_grid, z_grid, Nx, Ny, Nz, 3 * Nx / 8, Nx / 2, 3 * Ny / 8, Ny / 2, 3 * Nz / 8, Nz / 2);
    // std::thread w5(writeCSV, file_names[4], x_grid, y_grid, z_grid, Nx, Ny, Nz, Nx / 2, 5 * Nx / 8, Ny / 2, 5 * Ny / 8, Nz / 2, 5 * Nz / 8);
    // std::thread w6(writeCSV, file_names[5], x_grid, y_grid, z_grid, Nx, Ny, Nz, 5 * Nx / 8, 3 * Nx / 4, 5 * Ny / 8, 3 * Ny / 4, 5 * Nz / 8, 3 * Nz / 4);
    // std::thread w7(writeCSV, file_names[6], x_grid, y_grid, z_grid, Nx, Ny, Nz, 3 * Nx / 4, 7 * Nx / 8, 3 * Ny / 4, 7 * Ny / 8, 3 * Nz / 4, 7 * Nz / 8);
    // std::thread w8(writeCSV, file_names[7], x_grid, y_grid, z_grid, Nx, Ny, Nz, 7 * Nx / 8, Nx, 7 * Ny / 8, Ny, 7 * Nz / 8, Nz);

    std::thread w1(writeCSV, file_names[0], x_grid, y_grid, z_grid, Nx, Ny, Nz, 0, Nx, 0, Ny, 0, Nz / 8);
    std::thread w2(writeCSV, file_names[1], x_grid, y_grid, z_grid, Nx, Ny, Nz, 0, Nx, 0, Ny, Nz / 8, Nz / 4);
    std::thread w3(writeCSV, file_names[2], x_grid, y_grid, z_grid, Nx, Ny, Nz, 0, Nx, 0, Ny, Nz / 4, 3 * Nz / 8);
    std::thread w4(writeCSV, file_names[3], x_grid, y_grid, z_grid, Nx, Ny, Nz, 0, Nx, 0, Ny, 3 * Nz / 8, Nz / 2);
    std::thread w5(writeCSV, file_names[4], x_grid, y_grid, z_grid, Nx, Ny, Nz, 0, Nx, 0, Ny, Nz / 2, 5 * Nz / 8);
    std::thread w6(writeCSV, file_names[5], x_grid, y_grid, z_grid, Nx, Ny, Nz, 0, Nx, 0, Ny, 5 * Nz / 8, 3 * Nz / 4);
    std::thread w7(writeCSV, file_names[6], x_grid, y_grid, z_grid, Nx, Ny, Nz, 0, Nx, 0, Ny, 3 * Nz / 4, 7 * Nz / 8);
    std::thread w8(writeCSV, file_names[7], x_grid, y_grid, z_grid, Nx, Ny, Nz, 0, Nx, 0, Ny, 7 * Nz / 8, Nz);

    w1.join(); w2.join(); w3.join(); w4.join(); w5.join(); w6.join(); w7.join(); w8.join();
    return;
}

void writeCSV(const std::string file_name, const float* x_grid, const float* y_grid, const float* z_grid, const size_t Nx, const size_t Ny, const size_t Nz,
    const size_t x_begin, const size_t x_end, size_t y_begin, size_t y_end, size_t z_begin, size_t z_end){
    std::remove(file_name.data()); // Remove old data file for clean slate
    std::ofstream data_file;
    data_file.open(file_name);
    
    data_file << "x,y,z,i,j,k" << std::endl;

    for (size_t k = z_begin; k < z_end; k++){
        for (size_t i = x_begin; i < x_end; i++){
            for (size_t j = y_begin; j < y_end; j++){
                data_file << x_grid[i] << "," << y_grid[j] << "," << z_grid[k] << "," << i << "," << j << "," << k << std::endl;
            }
        }
    }
    return;
}

/* EXTEND TO ALL FLUID VARS */
void writeFluidVars(const std::vector<std::string> file_names, 
    const float* rho, const size_t Nx, const size_t Ny, const size_t Nz){
        /* 
        To avoid issues with asynchronous execution, e.g., needing a semaphore, each thread is going to write to its own `.csv` file
        These files will then be merged together with cuDF
        */
        if (file_names.size() != 8){
            std::cout << "Not providing the right number of file names to writeFluidVars" << std::endl;
            std::cout << "The correct number is 8" << std::endl;
        }
        /* 
        DOESN'T WRITE ALL THE DATA 
        Each thread writes a cube of data, the union of which does not form a complete representation of the original data
        */
        // std::thread w1(writeFVCSV, file_names[0], rho, Nx, Ny, Nz, 0, Nx / 8, 0, Ny / 8, 0, Nz / 8);
        // std::thread w2(writeFVCSV, file_names[1], rho, Nx, Ny, Nz, Nx / 8, Nx / 4, Ny / 8, Ny / 4, Nz / 8, Nz / 4);
        // std::thread w3(writeFVCSV, file_names[2], rho, Nx, Ny, Nz, Nx / 4, 3 * Nx / 8, Ny / 4, 3 * Ny / 8, Nz / 4, 3 * Nz / 8);
        // std::thread w4(writeFVCSV, file_names[3], rho, Nx, Ny, Nz, 3 * Nx / 8, Nx / 2, 3 * Ny / 8, Ny / 2, 3 * Nz / 8, Nz / 2);
        // std::thread w5(writeFVCSV, file_names[4], rho, Nx, Ny, Nz, Nx / 2, 5 * Nx / 8, Ny / 2, 5 * Ny / 8, Nz / 2, 5 * Nz / 8);
        // std::thread w6(writeFVCSV, file_names[5], rho, Nx, Ny, Nz, 5 * Nx / 8, 3 * Nx / 4, 5 * Ny / 8, 3 * Ny / 4, 5 * Nz / 8, 3 * Nz / 4);
        // std::thread w7(writeFVCSV, file_names[6], rho, Nx, Ny, Nz, 3 * Nx / 4, 7 * Nx / 8, 3 * Ny / 4, 7 * Ny / 8, 3 * Nz / 4, 7 * Nz / 8);
        // std::thread w8(writeFVCSV, file_names[7], rho, Nx, Ny, Nz, 7 * Nx / 8, Nx, 7 * Ny / 8, Ny, 7 * Nz / 8, Nz);

        std::thread w1(writeFVCSV, file_names[0], rho, Nx, Ny, Nz, 0, Nx, 0, Ny, 0, Nz / 8);
        std::thread w2(writeFVCSV, file_names[1], rho, Nx, Ny, Nz, 0, Nx, 0, Ny, Nz / 8, Nz / 4);
        std::thread w3(writeFVCSV, file_names[2], rho, Nx, Ny, Nz, 0, Nx, 0, Ny, Nz / 4, 3 * Nz / 8);
        std::thread w4(writeFVCSV, file_names[3], rho, Nx, Ny, Nz, 0, Nx, 0, Ny, 3 * Nz / 8, Nz / 2);
        std::thread w5(writeFVCSV, file_names[4], rho, Nx, Ny, Nz, 0, Nx, 0, Ny, Nz / 2, 5 * Nz / 8);
        std::thread w6(writeFVCSV, file_names[5], rho, Nx, Ny, Nz, 0, Nx, 0, Ny, 5 * Nz / 8, 3 * Nz / 4);
        std::thread w7(writeFVCSV, file_names[6], rho, Nx, Ny, Nz, 0, Nx, 0, Ny, 3 * Nz / 4, 7 * Nz / 8);
        std::thread w8(writeFVCSV, file_names[7], rho, Nx, Ny, Nz, 0, Nx, 0, Ny, 7 * Nz / 8, Nz);

        w1.join(); w2.join(); w3.join(); w4.join(); w5.join(); w6.join(); w7.join(); w8.join();
        return;
    }

/* EXTEND TO ALL FLUID VARS */
void writeFVCSV(const std::string file_name, const float* rho, const size_t Nx, const size_t Ny, const size_t Nz,
    const size_t x_begin, const size_t x_end, const size_t y_begin, const size_t y_end, const size_t z_begin, const size_t z_end){
        std::remove(file_name.data()); // Remove old data file for clean slate
        std::ofstream data_file;
        data_file.open(file_name);

        data_file << "rho,i,j,k" << std::endl; 

        for (size_t k = z_begin; k < z_end; k++){
            for (size_t i = x_begin; i < x_end; i++){
                for (size_t j = y_begin; j < y_end; j++){
                    data_file << rho[IDX3D(i, j, k, Nx, Ny, Nz)] << "," << i << "," << j << "," << k << std::endl;
                }
            }
        }

        return;
    }
