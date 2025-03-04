#include <string>
#include <fstream>
#include <iostream>
#include <unistd.h>

#include "utils.hpp"

// PHDF5 output functions
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
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        std::cout << "Current working directory: " << cwd << std::endl;
    } else {
       std::cerr << "Failed to get current working directory" << std::endl;
    }

    std::string mpirun_command = "mpirun -np " + num_proc + " ./" + phdf5_bin_name  
                                    + " " + file_name + " " + std::to_string(Nx) + " "
                                    + std::to_string(Ny) + " " + std::to_string(Nz) + " "
                                    + shm_name + " " + std::to_string(data_size);

    // Fork to PHDF5 output binary 
    std::cout << "Executing command: " << mpirun_command << std::endl;
    int ret = std::system(mpirun_command.data()); 
    return ret;
}

int callBinary_WriteGrid(const std::string bin_name, const std::string file_name, const std::string shm_name_gridx, const std::string shm_name_gridy, const std::string shm_name_gridz, 
    const int Nx, const int Ny, const int Nz){
    std::string writegrid_command = "./" + bin_name + " " + std::to_string(Nx) + " " + std::to_string(Ny) + " " + std::to_string(Nz) + " " 
        + " " + shm_name_gridx + " " + shm_name_gridy + " " + shm_name_gridz + " " + file_name; 
    
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        std::cout << "Current working directory: " << cwd << std::endl;
    } else {
       std::cerr << "Failed to get current working directory" << std::endl;
    }

    std::cout << "Executing command: " << writegrid_command << std::endl;
    int ret = std::system(writegrid_command.data());
    return ret;
}

// Determines CFL number at every point in the domain using eigenvalue solve
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
}

/* 
WRITE
Determines CFL number at every point in the domain using analytic expressions for the eigenvalues
Adaptive method that returns value of new timestep necessary to maintain stability
Choice is made based on safety factor  
*/
float adaptiveTimeStep(const float dt){
    float new_dt = dt;
    /* WRITE */
    return new_dt;
}


// This is used in benchmarking, and debug versions of code
// Python launcher does parsing of input file normally
// Extra layer is unwanted when it comes to debugging / benchmarking 
void parseInputFileDebug(std::vector<float>& inputs, const std::string input_file){
    /* READ THROUGH input_file AND PUT DATA INTO inputs */
    std::ifstream input_file_stream(input_file);
    std::string line;
    int i = 0;
    while(std::getline(input_file_stream, line)){
        inputs[i] = std::atof(line.data());
        i++;
    }
    return;
}