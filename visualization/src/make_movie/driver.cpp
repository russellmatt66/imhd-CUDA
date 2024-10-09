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

    std::string fluidvar_name = argv[1]; // name of dataset to open in fluidvar .h5 file
    std::string path_to_data = argv[2];
    std::string gridfile_name = argv[3];
    size_t Nt = atoi(argv[4]); 

    /* Get mesh dimensions */
    // Allocate shared memory that will store the necessary attributes
    // = Nx, Ny, Nz, dx, dy, dz
    // Shared memory = workaround b/c VTK and .h5 files don't mix
    std::string shm_gridattr_name = "/grid_attributes";
    int shm_fd_gridattr = shm_open(shm_gridattr_name.data(), O_CREAT | O_RDWR, 0666);
    if (shm_fd_gridattr == -1){
        std::cerr << "Failed to create shared memory" << std::endl;
        return 1;
    }

    size_t sizeof_gridattr = sizeof(float) * 6;
    ftruncate(shm_fd_gridattr, sizeof_gridattr); // Nx, Ny, Nz, ...

    float* shm_gridattr = (float*)mmap(0, sizeof_gridattr, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_gridattr, 0);
    if (shm_gridattr == MAP_FAILED) {
        std::cerr << "Failed to map shared memory" << std::endl;
        return 1;
    }

    /* Initialize */
    // vtkNew<vtkImageImport> frame_data;
    // imageImport->SetDataSpacing(dx, dy, dz);
    // imageImport->SetDataOrigin(0, 0, 0);
    // imageImport->SetWholeExtent(0, Nx - 1, 0, Ny - 1, 0, Nz - 1);
    // imageImport->SetDataExtentToWholeExtent();
    // imageImport->SetDataScalarTypeToFloat();  
    // imageImport->SetNumberOfScalarComponents(1);

    /* Loop over frames, and make video */
    // for (int i = 0; i < Nt; i++){
        // frame_data->SetImportVoidPointer();
    // }

    return 0;
}