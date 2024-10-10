#include <iostream>
#include <string>

#include <vtkCamera.h>
#include <vtkColorTransferFunction.h>
#include <vtkFFMPEGWriter.h>
#include <vtkImageImport.h>
#include <vtkNew.h>
#include <vtkPiecewiseFunction.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkWindowToImageFilter.h>

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char* argv[]){
    std::cout << "Inside make_movie driver" << std::endl;

    std::string path_to_data = argv[1];
    std::string dset_name = argv[2]; // specific fluid variable to visualize
    std::string filename_grid = argv[3];
    size_t Nt = atoi(argv[4]);

    /* Get mesh dimensions */
    // Allocate shared memory that will store the necessary attributes
    // = Nx, Ny, Nz, dx, dy, dz
    // Shared memory = workaround b/c VTK and .h5 files don't mix
    std::string shm_gridattr_name = "/shm_grid_attributes";
    int shm_fd_gridattr = shm_open(shm_gridattr_name.data(), O_CREAT | O_RDWR, 0666);
    if (shm_fd_gridattr == -1){
        std::cerr << "Failed to create shared memory for grid attributes" << std::endl;
        return 1;
    }

    size_t gridattr_data_size = sizeof(float) * 6;
    ftruncate(shm_fd_gridattr, gridattr_data_size); // Nx, Ny, Nz, ...

    float* shm_gridattr = (float*)mmap(0, gridattr_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_gridattr, 0);
    if (shm_gridattr == MAP_FAILED) {
        std::cerr << "Failed to map shared memory for grid attributes" << std::endl;
        return 1;
    }

    /* Fork process to obtain necessary grid attributes */
    std::string gridattr_command = "./read_grid_data " + shm_gridattr_name + " " + filename_grid + " " + std::to_string(gridattr_data_size);
    std::cout << "Forking to process for obtaining grid attributes" << std::endl;
    std::system(gridattr_command.data());

    int Nx = (int)shm_gridattr[0], Ny = (int)shm_gridattr[1], Nz = (int)shm_gridattr[2];
    float dx = shm_gridattr[3], dy = shm_gridattr[4], dz = shm_gridattr[5];

    std::cout << "Returned successfully from process. Attribute values are: " << std::endl;
    std::cout << "(Nx, Ny, Nz) = " << "(" << Nx << "," << Ny << "," << Nz << ")" << std::endl;
    std::cout << "(dx, dy, dz) = " << "(" << dx << "," << dy << "," << dz << ")" << std::endl; 

    // Allocate shared memory that will contain the fluidvar data 
    std::string shm_fluidvar_name = "/shm_fluidvar_data";
    int shm_fd_fluidvar = shm_open(shm_fluidvar_name.data(), O_CREAT | O_RDWR, 0666);
    if (shm_fd_fluidvar == -1){
        std::cerr << "Failed to create shared memory for fluidvar data" << std::endl;
        return 1;
    }

    size_t fluid_data_size = sizeof(float) * Nx * Ny * Nz;
    ftruncate(shm_fd_fluidvar, fluid_data_size);

    float* shm_fluidvar = (float*)mmap(0, fluid_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_fluidvar, 0);
    if (shm_fluidvar == MAP_FAILED) {
        std::cerr << "Failed to map shared memory for fluidvar data" << std::endl;
        return 1;
    }

    // VTK 
    vtkNew<vtkImageImport> frame_data;
    frame_data->SetDataSpacing(dx, dy, dz);
    frame_data->SetDataOrigin(0, 0, 0);
    frame_data->SetWholeExtent(0, Nx - 1, 0, Ny - 1, 0, Nz - 1);
    frame_data->SetDataExtentToWholeExtent();
    frame_data->SetDataScalarTypeToFloat();  
    frame_data->SetNumberOfScalarComponents(1);

    /* Loop over frames, and make video */
    std::string load_fluidvar_command = "";
    for (int i = 0; i <= Nt; i++){
        std::cout << "Writing frame " << i << " to video" << std::endl;
        /* Fork process to load fluidvar data from .h5 file */
        std::system(load_fluidvar_command.data());
        frame_data->SetImportVoidPointer(shm_fluidvar);
    }

    // Manage memory
    shm_unlink(shm_gridattr_name.data());
    shm_unlink(shm_fluidvar_name.data());
    munmap(shm_gridattr, gridattr_data_size);
    munmap(shm_fluidvar, fluid_data_size);
    close(shm_fd_gridattr);
    close(shm_fd_fluidvar);
    return 0;
}