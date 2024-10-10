#include <iostream>
#include <string>

#include "hdf5.h" 

// Load fluidvar data from .h5 file into shared memory for VTK consumption 
int main(int argc, char* argv[]){
    std::cout << "Inside read_fluid_data" << std::endl;

    std::string shm_fluidvar_name = argv[1];
    std::string dset_name = argv[2];
    std::string file_name = argv[3];
    int Nx = atoi(argv[4]);
    int Ny = atoi(argv[5]);
    int Nz = atoi(argv[6]);

    /* Open fluidvar shared memory */

    /* Read dset data into shared fluidvar memory */

    /* Close everything */

    // Manage memory
    
    return 0;
}