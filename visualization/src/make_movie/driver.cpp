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
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>
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
    int ret = std::system(gridattr_command.data());
    if (ret != 0) {
        std::cerr << "Error executing command: " << gridattr_command << std::endl;
        return 1;
    }

    int Nx = (int)shm_gridattr[0], Ny = (int)shm_gridattr[1], Nz = (int)shm_gridattr[2];
    float dx = shm_gridattr[3], dy = shm_gridattr[4], dz = shm_gridattr[5];

    std::cout << "Returned successfully from process. Attribute values are: " << std::endl;
    std::cout << "(Nx, Ny, Nz) = " << "(" << Nx << "," << Ny << "," << Nz << ")" << std::endl;
    std::cout << "(dx, dy, dz) = " << "(" << dx << "," << dy << "," << dz << ")" << std::endl; 

    // Allocate shared memory that will contain the fluidvar data 
    std::cout << "Allocating shared memory for fluidvar data" << std::endl;
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
    std::cout << "Shared memory for fluidvar data successfully allocated" << std::endl;


    // VTK
    std::cout << "Instantiating VTK objects" << std::endl; 

    std::cout << "Instantiating frame_data" << std::endl;
    vtkNew<vtkImageImport> frame_data;
    frame_data->SetDataSpacing(dx, dy, dz);
    frame_data->SetDataOrigin(0, 0, 0);
    frame_data->SetWholeExtent(0, Nx - 1, 0, Ny - 1, 0, Nz - 1);
    frame_data->SetDataExtentToWholeExtent();
    frame_data->SetDataScalarTypeToFloat();  
    frame_data->SetNumberOfScalarComponents(1);
    std::cout << "frame_data instantiated" << std::endl;

    std::cout << "Instantiating volumeMapper" << std::endl;
    vtkNew<vtkSmartVolumeMapper> volumeMapper;
    volumeMapper->SetInputConnection(frame_data->GetOutputPort());
    std::cout << "volumeMapper instantiated, input connection to frame_data set" << std::endl;

    std::cout << "Instantiating volumeActor" << std::endl;
    vtkNew<vtkVolume> volumeActor;
    volumeActor->SetMapper(volumeMapper);
    std::cout << "volumeActor instantiated, mapper set to volumeMapper" << std::endl;

    /* Needs wrapper that does the scaling using min and max values */
    std::cout << "Instantiating colorTransfer" << std::endl;
    vtkNew<vtkColorTransferFunction> colorTransfer;
    colorTransfer->AddRGBPoint(0, 1.0, 0.0, 0.0);   // Red
    colorTransfer->AddRGBPoint(0.1667, 1.0, 0.5, 0.0); // Orange
    colorTransfer->AddRGBPoint(0.3333, 1.0, 1.0, 0.0); // Yellow
    colorTransfer->AddRGBPoint(0.5, 0.0, 1.0, 0.0);    // Green
    colorTransfer->AddRGBPoint(0.6667, 0.0, 0.0, 1.0); // Blue
    colorTransfer->AddRGBPoint(0.8333, 0.5, 0.0, 1.0); // Indigo
    colorTransfer->AddRGBPoint(1, 1.0, 0.0, 1.0);       // Violet
    std::cout << "colorTransfer instantiated" << std::endl;

    /* Needs wrapper that does the scaling using min and max values */
    std::cout << "Instantiating opacityTransfer" << std::endl;
    vtkNew<vtkPiecewiseFunction> opacityTransfer;
    opacityTransfer->AddPoint(0, 0.0);  // Adjust opacity points based on your data
    opacityTransfer->AddPoint(1, 1.0);
    std::cout << "opacityTransfer instantiated" << std::endl;

    volumeActor->GetProperty()->SetScalarOpacity(opacityTransfer);
    volumeActor->GetProperty()->SetColor(colorTransfer);

    std::cout << "Instantiating renderer, and renderWindow" << std::endl;
    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkRenderWindow> renderWindow;
    renderWindow->AddRenderer(renderer);
    std::cout << "renderer, and renderWindow instantiated" << std::endl;

    std::cout << "Instantiating camera" << std::endl;
    vtkNew<vtkCamera> camera;
    camera->SetPosition(10, 10, 10);
    camera->SetFocalPoint(0, 0, 0);
    camera->SetViewUp(1, 0, 0);
    std::cout << "camera instantiated" << std::endl;

    std::cout << "Connecting camera, and volumeActor to renderer" << std::endl;
    renderer->SetActiveCamera(camera);
    renderer->SetBackground(0.0, 0.0, 0.0);
    renderer->AddVolume(volumeActor);
    std::cout << "camera and volumeActor connected to renderer" << std::endl;
    
    std::cout << "Instantiating windowToImageFilter" << std::endl;
    vtkNew<vtkWindowToImageFilter> windowToImageFilter;
    windowToImageFilter->SetInput(renderWindow);
    windowToImageFilter->Update();
    std::cout << "windowToImageFilter instantiated - renderWindow connected" << std::endl;

    std::cout << "Instantiating videoWriter" << std::endl;
    std::string video_filename = "../" + dset_name + "_" + std::to_string(Nx) + std::to_string(Ny) + std::to_string(Nz) + ".mp4"; // Being run from inside 'build/'
    vtkNew<vtkFFMPEGWriter> videoWriter;
    videoWriter->SetInputConnection(windowToImageFilter->GetOutputPort());
    videoWriter->SetFileName(video_filename.data());
    videoWriter->Start();
    std::cout << "videoWriter instantiated" << std::endl;

    /* Loop over frames, and make video */
    std::string load_fluidvar_command = "";
    std::string fluidvar_filename = "";

    for (int i = 0; i <= Nt; i++){
        std::cout << "Writing frame " << i << " to video" << std::endl;

        /* Fork process to load fluidvar data from .h5 file */
        fluidvar_filename = path_to_data + "fluidvars_" + std::to_string(i) + ".h5";
        std::cout << "fluidvar_filename is: " << fluidvar_filename << std::endl;

        load_fluidvar_command = "./read_fluid_data " + fluidvar_filename + " " + dset_name + " " + fluidvar_filename 
            + " " + std::to_string(Nx) + " " + std::to_string(Ny) + " " + std::to_string(Nz);
        std::cout << "Forking to read_fluid_data binary with: " << load_fluidvar_command << std::endl;
        ret = std::system(load_fluidvar_command.data());
        if (ret != 0) {
           std::cerr << "Error executing command: " << load_fluidvar_command << std::endl;
            return 1;
        }

        /* Write frame to video */
        frame_data->SetImportVoidPointer(shm_fluidvar);
        renderWindow->Render();

        windowToImageFilter->Modified();
        
        videoWriter->Write();
    }

    std::cout << "Frames successfully written, ending videoWriter" << std::endl;
    videoWriter->End();

    // Manage memory
    std::cout << "Freeing shared memory" << std::endl;
    shm_unlink(shm_gridattr_name.data());
    shm_unlink(shm_fluidvar_name.data());
    munmap(shm_gridattr, gridattr_data_size);
    munmap(shm_fluidvar, fluid_data_size);
    close(shm_fd_gridattr);
    close(shm_fd_fluidvar);

    return 0;
}