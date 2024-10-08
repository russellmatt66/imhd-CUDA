/*
Based on https://examples.vtk.org/site/Cxx/Images/ImageImport/
*/

#include <vtkImageActor.h>
#include <vtkImageData.h>
#include <vtkImageImport.h>
#include <vtkImageMapToColors.h>
#include <vtkInteractorStyleImage.h>
#include <vtkLookupTable.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkXMLImageDataWriter.h>

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

    vtkNew<vtkNamedColors> colors;

    // Convert the c-style image to a vtkImageData
    vtkNew<vtkImageImport> imageImport;
    imageImport->SetDataSpacing(dx, dy, dz);
    imageImport->SetDataOrigin(0, 0, 0);
    imageImport->SetWholeExtent(0, Nx - 1, 0, Ny - 1, 0, Nz - 1);
    imageImport->SetDataExtentToWholeExtent();
    imageImport->SetDataScalarTypeToFloat();  
    imageImport->SetNumberOfScalarComponents(1);
    imageImport->SetImportVoidPointer(shm_fluidvar);
    imageImport->Update();

    double scalarRange[2];
    imageImport->GetOutput()->GetScalarRange(scalarRange);
    std::cout << "Scalar range: [" << scalarRange[0] << ", " << scalarRange[1] << "]" << std::endl;

    // Create an actor
    vtkNew<vtkImageActor> actor;
    // actor->SetInputData(imageImport->GetOutput());

    // Create a lookup table to map scalar values to colors
    vtkNew<vtkLookupTable> lookupTable;
    lookupTable->SetRange(scalarRange);  // Set the range of your scalar values
    lookupTable->Build();

    // Create a mapper to map the image data through the lookup table
    vtkNew<vtkImageMapToColors> mapColors;
    mapColors->SetInputData(imageImport->GetOutput());
    mapColors->SetLookupTable(lookupTable);
    mapColors->Update();

    actor->SetInputData(mapColors->GetOutput());

    // Assign the color-mapped image to the actor
    // actor->GetMapper()->SetInputConnection(mapColors->GetOutputPort());

    // Setup renderer
    vtkNew<vtkRenderer> renderer;
    renderer->AddActor(actor);
    renderer->ResetCamera();
    renderer->SetBackground(1.0, 1.0, 1.0);

    // Setup render window
    vtkNew<vtkRenderWindow> renderWindow;
    renderWindow->AddRenderer(renderer);
    renderWindow->SetWindowName("ImageImport");

    // Setup render window interactor
    vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
    vtkNew<vtkInteractorStyleImage> style;

    renderWindowInteractor->SetInteractorStyle(style);

    // Render and start interaction
    renderWindowInteractor->SetRenderWindow(renderWindow);
    renderWindow->Render();
    renderWindowInteractor->Initialize();

    renderWindowInteractor->Start();

    munmap(shm_fluidvar, fluid_data_size);
    close(shm_fd);
    return EXIT_SUCCESS;
}