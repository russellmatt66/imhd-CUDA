#include <vector>
#include <string>

int callBinary_AttrWrite(const std::string file_name, const int Nx, const int Ny, const int Nz, const std::string attr_bin_name);

int callBinary_PHDF5Write(const std::string file_name, const int Nx, const int Ny, const int Nz, 
                const std::string shm_name, const size_t data_size, 
                const std::string num_proc, const std::string phdf5_bin_name);

int callBinary_WriteGrid(const std::string bin_name, const std::string file_name, const std::string shm_name_gridx, const std::string shm_name_gridy, const std::string shm_name_gridz, 
    const int Nx, const int Ny, const int Nz);

int callBinary_EigenSC(const std::string shm_name, const int Nx, const int Ny, const int Nz, const std::string bin_name, 
    const float dt, const float dx, const float dy, const float dz, 
    const std::string shm_name_gridx, const std::string shm_name_gridy, const std::string shm_name_gridz);

float adaptiveTimeStep(const float dt);

void parseInputFileDebug(std::vector<float>& inputs, const std::string input_file);
// void parseInputFileBench(std::vector<float>& inputs, const std::string input_file);