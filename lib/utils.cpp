#include <string>
#include <iostream>
#include <unistd.h>

#include "utils.hpp"

int callBinary_AttrWrite(const std::string file_name, const int Nx, const int Ny, const int Nz, const std::string attr_bin_name){
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

int callBinary_PHDF5Write(const std::string file_name, const int Nx, const int Ny, const int Nz, 
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

int callBinary_EigenSC(const std::string shm_name, const int Nx, const int Ny, const int Nz, const std::string bin_name, 
    const float dt, const float dx, const float dy, const float dz, 
    const std::string shm_name_gridx, const std::string shm_name_gridy, const std::string shm_name_gridz){
    std::string eigen_command = "./" + bin_name + " " + shm_name + " " + std::to_string(Nx) + " "
                                    + std::to_string(Ny) + " " + std::to_string(Nz) + " " + std::to_string(dt)
                                    + " " + std::to_string(dx) + " " + std::to_string(dy) + " " + std::to_string(dz)
                                    + " " + shm_name_gridx + " " + shm_name_gridy + " " + shm_name_gridz;
    
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