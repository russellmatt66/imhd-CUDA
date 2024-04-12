#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

int main(int argc, char* argv[]) {
    // Read data from .dat files into memory and store in vectors or arrays
    std::ifstream infile("../xyz_grid.dat");
    if (!infile.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        return 1;
    }


    std::vector<char> buffer(std::istreambuf_iterator<char>(infile), {});
    float x, y, z;
    std::vector<float> xCoords, yCoords, zCoords;
    size_t numPoints = buffer.size() / sizeof(float)/ 3;

    size_t num_zerozerozero = 0; // Having a problem where all three coordinates are 0 more than they should be
    for (size_t i = 0; i < numPoints; i++){
        std::memcpy(&x, &buffer[i * sizeof(float) * 3], sizeof(float));
        std::memcpy(&y, &buffer[i * sizeof(float) * 3 + sizeof(float)], sizeof(float));
        std::memcpy(&z, &buffer[i * sizeof(float) * 3 + 2 * sizeof(float)], sizeof(float));

        if (x == 0.0 && y == 0.0 && z == 0.0) {
            num_zerozerozero++;
        }
        // std::cout << "Pushing: " << x << "," << y << "," << z << std::endl;
        xCoords.push_back(x);
        yCoords.push_back(y);
        zCoords.push_back(z);
    }

    /* 
    NOTE:
    .dat files contains raw bytes, need to correctly interpet as floating-point values representing points on a cartesian grid
    */
    // std::vector<double> xCoords, yCoords, zCoords;
    // double x, y, z;
    // while (infile >> x >> y >> z) {
    //     std::cout << "Pushing " << x << "," << y << "," << z << std::endl; 
    //     xCoords.push_back(x); 
    //     yCoords.push_back(y);
    //     zCoords.push_back(z);
    //     numPoints++;
    // }
    
    infile.close();
    std::cout << "Number of points read: " << numPoints << std::endl;
    std::cout << "Number of 0,0,0 found: " << num_zerozerozero << std::endl;


    // Create a VTK Points object
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

    // Populate the vtkPoints object with the coordinates
    // Assuming data is stored in vectors xCoords, yCoords, zCoords
    for (int i = 0; i < numPoints; ++i) {
        points->InsertNextPoint(xCoords[i], yCoords[i], zCoords[i]);
    }

    // Create a VTK PolyData object and set its points
    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);

    // Convert points into vertices for visualization
    vtkSmartPointer<vtkVertexGlyphFilter> vertexFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
    vertexFilter->SetInputData(polydata);

    // Create a mapper
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(vertexFilter->GetOutputPort());

    // Create an actor
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    // Create a renderer
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->AddActor(actor);

    // Create a render window
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);

    // Create an interactor
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    // Start the interaction
    renderWindow->Render();
    renderWindowInteractor->Start();

    return 0;
}
