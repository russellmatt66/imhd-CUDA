// #include <vtkSmartPointer.h>
// #include <vtkImageData.h>
// #include <vtkPoints.h>
// #include <vtkPolyData.h>
// #include <vtkVertexGlyphFilter.h>
// #include <vtkPolyDataMapper.h>
// #include <vtkActor.h>
// #include <vtkImageActor.h>
// #include <vtkImageMapper3D.h>
// #include <vtkRenderer.h>
// #include <vtkRenderWindow.h>
// #include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkGPUVolumeRayCastMapper.h>
// #include <vtkAutoInit.h>

// VTK_MODULE_INIT(vtkRenderingOpenGL2); // VTK was built with vtkRenderingOpenGL2
// VTK_MODULE_INIT(vtkInteractionStyle);

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

int main(int argc, char* argv[]){
    size_t Nx = std::stoi(argv[1]);
    size_t Ny = std::stoi(argv[2]);
    size_t Nz = std::stoi(argv[3]);

    std::ifstream infile("../rho_ics.dat", std::ios::binary);
    if (!infile) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }

    std::vector<char> buffer(std::istreambuf_iterator<char>(infile), {});
    size_t num_points = buffer.size() / sizeof(float);
    float temp;
    std::vector<float> rho_data;
    
    size_t num_ones = 0.0;
    for (size_t l = 0; l < num_points; l++){
        std::memcpy(&temp, &buffer[l * sizeof(float)], sizeof(float));
        if (temp == 1.0){
            num_ones++;
        }
        rho_data.push_back(temp);
    }
    // infile.read(reinterpret_cast<char*>(rho_data.data()), sizeof(float) * Nx * Ny * Nz);
    infile.close();

    std::cout << "Number of points read: " << num_points << std::endl;
    std::cout << "Dimensions of input data, (Nx, Ny, Nz): (" << Nx << "," << Ny << "," << Nz << ")" << std::endl;
    std::cout << "Size of rho array: " << rho_data.size() << std::endl;
    std::cout << "Number of ones: " << num_ones << std::endl;

    vtkSmartPointer<vtkImageData> rhoData = vtkSmartPointer<vtkImageData>::New();
    rhoData->SetDimensions(Nx, Ny, Nz);

    vtkSmartPointer<vtkFloatArray> rhoPixels = vtkSmartPointer<vtkFloatArray>::New();
    rhoPixels->SetNumberOfComponents(1);
    rhoPixels->SetArray(rho_data.data(), Nx * Ny * Nz, 1);

    rhoData->GetPointData()->SetScalars(rhoPixels);

    // Set the spacing between pixels
    double spacing[3] = {1.0, 1.0, 1.0}; // Adjust as needed
    rhoData->SetSpacing(spacing);

    // Set the origin of the image
    double origin[3] = {0.0, 0.0, 0.0}; // Adjust as needed
    rhoData->SetOrigin(origin);

    // Update the image data
    rhoData->Modified();

     // Create a volume mapper
    vtkSmartPointer<vtkGPUVolumeRayCastMapper> volumeMapper = vtkSmartPointer<vtkGPUVolumeRayCastMapper>::New();
    volumeMapper->SetInputData(rhoData);

    // Create a volume property
    vtkSmartPointer<vtkVolumeProperty> volumeProperty = vtkSmartPointer<vtkVolumeProperty>::New();
    volumeProperty->ShadeOn(); // Enable shading for better visualization

    // Create a color transfer function
    vtkSmartPointer<vtkColorTransferFunction> colorTransferFunction = vtkSmartPointer<vtkColorTransferFunction>::New();
    // Add color mapping points to the transfer function

    // Create an opacity transfer function
    vtkSmartPointer<vtkPiecewiseFunction> opacityTransferFunction = vtkSmartPointer<vtkPiecewiseFunction>::New();
    // Add opacity mapping points to the transfer function

    // Set the color and opacity transfer functions to the volume property
    volumeProperty->SetColor(colorTransferFunction);
    volumeProperty->SetScalarOpacity(opacityTransferFunction);

    // Create a volume
    vtkSmartPointer<vtkVolume> volume = vtkSmartPointer<vtkVolume>::New();
    volume->SetMapper(volumeMapper);
    volume->SetProperty(volumeProperty);

    // Create a renderer
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->AddVolume(volume);

    // Create a render window
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);

    // Create an interactor
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    // Start the rendering loop
    renderWindow->Render();
    renderWindowInteractor->Start();

    return 0;
}