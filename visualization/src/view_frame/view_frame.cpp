/*
Based on https://examples.vtk.org/site/Cxx/Images/ImageImport/
*/

#include <vtkCamera.h>
#include <vtkColorTransferFunction.h>
#include <vtkImageData.h>
#include <vtkImageImport.h>
#include <vtkNew.h>
#include <vtkPiecewiseFunction.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>

#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

int main(int argc, char* argv[]){
    std::cout << "Inside view_frame" << std::endl;

    int Nx = atoi(argv[1]);
    int Ny = atoi(argv[2]);
    int Nz = atoi(argv[3]);

    float dx = atof(argv[4]);
    float dy = atof(argv[5]);
    float dz = atof(argv[6]);
    
    float x_min = atof(argv[7]);
    float x_max = atof(argv[8]);
    float y_min = atof(argv[9]);
    float y_max = atof(argv[10]);
    float z_min = atof(argv[11]);
    float z_max = atof(argv[12]);

    std::string shm_name_fluidvar = argv[13];

    size_t fluid_data_size = sizeof(float) * Nx * Ny * Nz;

    int shm_fd = shm_open(shm_name_fluidvar.data(), O_RDWR, 0666);
    if (shm_fd == -1){
        std::cerr << "Inside view_frame" << std::endl;
        std::cerr << "Failed to open shared memory for fluidvar data" << std::endl;
        return EXIT_FAILURE;
    }

    float* shm_fluidvar = (float*)mmap(0, fluid_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

    vtkNew<vtkImageImport> imageImport;
    imageImport->SetDataSpacing(dx, dy, dz);
    imageImport->SetDataOrigin(0, 0, 0);
    imageImport->SetWholeExtent(0, Nx - 1, 0, Ny - 1, 0, Nz - 1);
    imageImport->SetDataExtentToWholeExtent();
    imageImport->SetDataScalarTypeToFloat();  
    imageImport->SetNumberOfScalarComponents(1);
    imageImport->SetImportVoidPointer(shm_fluidvar);
    imageImport->Update();

    // Create volume mapper
    vtkNew<vtkSmartVolumeMapper> volumeMapper;
    volumeMapper->SetInputConnection(imageImport->GetOutputPort());

    // Create volume actor
    vtkNew<vtkVolume> volumeActor;
    volumeActor->SetMapper(volumeMapper);

    // Setup color transfer function
    vtkNew<vtkColorTransferFunction> colorTransfer;
    colorTransfer->AddRGBPoint(0, 1.0, 0.0, 0.0);   // Red
    colorTransfer->AddRGBPoint(0.1667, 1.0, 0.5, 0.0); // Orange
    colorTransfer->AddRGBPoint(0.3333, 1.0, 1.0, 0.0); // Yellow
    colorTransfer->AddRGBPoint(0.5, 0.0, 1.0, 0.0);    // Green
    colorTransfer->AddRGBPoint(0.6667, 0.0, 0.0, 1.0); // Blue
    colorTransfer->AddRGBPoint(0.8333, 0.5, 0.0, 1.0); // Indigo
    colorTransfer->AddRGBPoint(1, 1.0, 0.0, 1.0);       // Violet
    
    // Setup opacity transfer function
    vtkNew<vtkPiecewiseFunction> opacityTransfer;
    opacityTransfer->AddPoint(0, 0.0);  // Adjust opacity points based on your data
    opacityTransfer->AddPoint(1, 1.0);
    
    volumeActor->GetProperty()->SetScalarOpacity(opacityTransfer);
    volumeActor->GetProperty()->SetColor(colorTransfer);

    // Set up the Renderer, RenderWindow, and Interactor
    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkRenderWindow> renderWindow;
    renderWindow->AddRenderer(renderer);
    vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
    renderWindowInteractor->SetRenderWindow(renderWindow);

    vtkNew<vtkCamera> camera;
    camera->SetPosition(10, 10, 10); // X, Y, Z coordinates; adjust as needed
    camera->SetFocalPoint(0, 0, 0); // Center of the volume
    camera->SetViewUp(1, 0, 0); // z-axis is along pinch
    renderer->SetActiveCamera(camera);

    // Add the volume actor to the scene
    renderer->AddVolume(volumeActor);
    renderer->SetBackground(0.0, 0.0, 0.0); // Background color

    // Render and interact
    renderWindow->Render();
    renderWindowInteractor->Start();

    munmap(shm_fluidvar, fluid_data_size);
    close(shm_fd);
    return EXIT_SUCCESS;
}