#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkDelimitedTextReader.h>
#include <vtkTable.h>
#include <vtkFloatArray.h>

#include <string>
#include <iostream>

void PrintTableContent(vtkSmartPointer<vtkTable> table);

int main(int argc, char* argv[]){
    std::string simdata_location = "../../data/sim_data.csv";
    // std::string fluiddata_location = "../data/";

    vtkSmartPointer<vtkDelimitedTextReader> reader = vtkSmartPointer<vtkDelimitedTextReader>::New();
    reader->SetFileName(simdata_location.data());
    reader->DetectNumericColumnsOn();
    reader->SetFieldDelimiterCharacters(",");
    reader->Update();

    // Check if the reader has valid output
    if (!reader->GetOutput())
    {
        std::cerr << "Error: Unable to read the CSV file." << std::endl;
        return EXIT_FAILURE;
    }

    vtkSmartPointer<vtkTable> simTable = reader->GetOutput();

    PrintTableContent(simTable);

    for (int i = 0; i < simTable->GetNumberOfColumns(); ++i)
    {
        std::cout << "Column " << i << ": " << simTable->GetColumnName(i) << std::endl;
        std::cout << "Column " << i << ": " << simTable->GetColumn(i) << std::endl;
    }

    // vtkFloatArray* x = vtkFloatArray::SafeDownCast(simTable->GetColumnByName("x"));
    // vtkFloatArray* y = vtkFloatArray::SafeDownCast(simTable->GetColumnByName("y"));
    // vtkFloatArray* z = vtkFloatArray::SafeDownCast(simTable->GetColumnByName("z"));
    // vtkFloatArray* rho = vtkFloatArray::SafeDownCast(simTable->GetColumnByName("rho"));
    
    // vtkSmartPointer<vtkFloatArray> x = vtkSmartPointer<vtkFloatArray>::New();
    // vtkSmartPointer<vtkFloatArray> y = vtkSmartPointer<vtkFloatArray>::New();
    // vtkSmartPointer<vtkFloatArray> z = vtkSmartPointer<vtkFloatArray>::New();
    // vtkSmartPointer<vtkFloatArray> rho = vtkSmartPointer<vtkFloatArray>::New();

    vtkSmartPointer<vtkFloatArray> x = vtkFloatArray::SafeDownCast(simTable->GetColumn(0));
    vtkSmartPointer<vtkFloatArray> y = vtkFloatArray::SafeDownCast(simTable->GetColumn(1));
    vtkSmartPointer<vtkFloatArray> z = vtkFloatArray::SafeDownCast(simTable->GetColumn(2));
    vtkSmartPointer<vtkFloatArray> rho = vtkFloatArray::SafeDownCast(simTable->GetColumn(6));

    if (!x || !y || !z || !rho)
    {
        std::cout << "!x: " << !x << std::endl; 
        std::cout << "!y: " << !y << std::endl; 
        std::cout << "!z: " << !z << std::endl; 
        std::cout << "!rho: " << !rho << std::endl; 
        std::cerr << "Error: One or more columns not found in the table." << std::endl;
        return EXIT_FAILURE;
    }

    vtkSmartPointer<vtkPoints> grid_points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkFloatArray> fluid_vals = vtkSmartPointer<vtkFloatArray>::New();

    std::cout << "Inserting values " << std::endl;
    for (vtkIdType l = 0; l < rho->GetNumberOfValues(); l++){
        std::cout << "Inserting l = " << l << std::endl;
        grid_points->InsertNextPoint(x->GetValue(l), y->GetValue(l), z->GetValue(l));
        fluid_vals->SetValue(l, rho->GetValue(l));
    }
    std::cout << "Values inserted successfully" << std::endl;

    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(grid_points);
    polydata->GetPointData()->SetScalars(fluid_vals);

    vtkSmartPointer<vtkVertexGlyphFilter> vertexGF = vtkSmartPointer<vtkVertexGlyphFilter>::New();
    vertexGF->SetInputData(polydata);

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(vertexGF->GetOutputPort());
    mapper->ScalarVisibilityOn();

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

        // Create renderer
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->AddActor(actor);
    renderer->SetBackground(1.0, 1.0, 1.0); // white background

    // Create render window
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);

    // Create render window interactor
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    // Adjust camera position
    renderer->ResetCamera();

    // Start the interaction
    renderWindow->Render();
    renderWindowInteractor->Start();

    // free(x);
    // free(y);
    // free(z);
    // free(rho);
    return 0;
}

void PrintTableContent(vtkSmartPointer<vtkTable> table){
    int numRows = table->GetNumberOfRows();
    int numCols = table->GetNumberOfColumns();

    std::cout << "numRows: " << numRows << std::endl;
    std::cout << "numCols: " << numCols << std::endl;
}
