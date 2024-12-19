#include <iostream>
#include <string>

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char* argv[]){
    std::cout << "Inside read_fluid_plane" << std::endl;

    std::string shm_fluidvar_name = argv[1];
    std::string dset_name = argv[2];
    std::string filename_frame = argv[3];
    
    size_t N_1 = atoi(argv[4]);
    size_t N_2 = atoi(argv[5]);

    size_t plane_val = atoi(argv[6]);

    /* Open shared memory representing the location where the variable data goes */
    /* Open .h5 file representing the frame */
    /* Read from the corresponding locations into shared memory */ 
    
    return 0;
}