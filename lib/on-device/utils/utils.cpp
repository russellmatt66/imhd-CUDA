#include <string>
#include <fstream>
#include <iostream>
#include <cstddef>

#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

#include "utils.hpp"

/* 
TODO:
(1) Research this matter, and find best practice
There's a modern C++ language feature to do this better, but this works for now 
*/
float* SHMAllocator(const std::string shm_name, const size_t data_size){
    int shm_fd = shm_open(shm_name.data(), O_CREAT | O_RDWR, 0666);

    if (shm_fd == -1) {
        std::cerr << "Failed to create (host) shared memory!" << std::endl;
        return NULL; // This connects to an external layer to check for `EXIT_FAILURE`
    }

    ftruncate(shm_fd, data_size);

    float* shm_h = (float*)mmap(0, data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

    if (shm_h == MAP_FAILED) {
        std::cerr << "mmap failed!" << std::endl;
        return NULL; // This connects to an external layer to check for `EXIT_FAILURE`
    }

    return shm_h;
}

/* 
TODO:
(1) Serial HDF5 output
*/
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

/* 
TODO:
(1) Replace with analytic expressions for the eigenvalues
*/
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
float adaptiveTimeStep(const float dt, const float* h_fluidvars){
    float new_dt = dt;
    /* WRITE */
    return new_dt;
}


// This is used in benchmarking, and debug versions of code
/* PERHAPS THIS ABOVE EXPRESSION SUGGESTS THAT THESE CONCERNS SHOULD BE SEPARATED?
x There are reasons not too, for example, maybe the specific changes can all be local so this can serve globally to load values into memory. 
o If this is purely used in those contexts, then there are strong reasons to move it out of here. 
The tradeoff is having to expand `debug/` and `bench/` build systems to accomodate `lib/` sub-directories
Maybe there should be a global `lib/utils`?
*/
// Python launcher does parsing of input file normally
// Extra layer is unwanted when it comes to debugging / benchmarking 
void parseInputFileDebug(std::vector<float>& inputs, const std::string input_file){
    /* READ THROUGH input_file AND PUT DATA INTO inputs */
    std::ifstream input_file_stream(input_file);
    std::string line;
    int i = 0;
    while(std::getline(input_file_stream, line)){
        inputs[i] = std::atof(line.data()); // quick and dirty - everything is fair game to be a float in the appropriate `input_file`
        i++;
    }
    return;
}