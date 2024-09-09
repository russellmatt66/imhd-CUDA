/* 
Proof of Concept:
Write Initial Conditions (any device data) out with HDF5 
*/
#include <string>

void writeH5(const std::string filename, const float* output_data, const int Nx, const int Ny, const int Nz);

int main(int argc, char* argv[]){
    // Specify CUDA boilerplate

    // Call Initial Conditions Kernel

    // Transfer Device data to Host

    // Write .h5 file out

    return 0;
}

/* This goes into library when finished */
void writeH5(const std::string filename, const float* output_data, const int Nx, const int Ny, const int Nz){
    return;
}