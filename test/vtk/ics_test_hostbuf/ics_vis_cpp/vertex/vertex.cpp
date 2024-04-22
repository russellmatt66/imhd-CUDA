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
#include <vtkAbstractArray.h>
#include <vtkScalarBarActor.h>
#include <vtkLookupTable.h>

#include <string>
#include <iostream>

/* 
I think this is the eol of this code, and it's time to move onto different visualization techniques for rendering point set
*/

void PrintTableContent(vtkSmartPointer<vtkTable> table);
float CenterLineRadius(float x, float y);

int main(int argc, char* argv[]){
    // std::cout << argv[2] << std::endl;

    // int fluid_var = std::stoi(argv[2]);

    // std::cout << fluid_var << std::endl;

    std::string simdata_location = "../../data/sim_data.csv"; /* THIS NEEDS TO BE SORTED IN ASCENDING ORDER */
    // std::string fluiddata_location = "../data/";

    vtkSmartPointer<vtkDelimitedTextReader> reader = vtkSmartPointer<vtkDelimitedTextReader>::New();
    reader->SetFileName(simdata_location.data());
    reader->SetHaveHeaders(true);
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

    vtkAbstractArray* x_a = simTable->GetColumn(0);
    vtkAbstractArray* y_a = simTable->GetColumn(1);
    vtkAbstractArray* z_a = simTable->GetColumn(2);
    vtkAbstractArray* fluiddata_a = simTable->GetColumn(7);
    // vtkAbstractArray* fluiddata_a = simTable->GetColumn(fluid_var);

    std::cout << x_a << std::endl;

    // std::cout << x << std::endl;
    // std::cout << x->GetNumberOfValues() << std::endl;

    if (!x_a || !y_a || !z_a || !fluiddata_a)
    {
        std::cout << "!x: " << !x_a << std::endl; 
        std::cout << "!y: " << !y_a << std::endl; 
        std::cout << "!z: " << !z_a << std::endl; 
        std::cout << "!rho: " << !fluiddata_a << std::endl; 
        std::cerr << "Error: One or more columns not found in the table." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Is x_a numeric? " << x_a->IsNumeric() << std::endl;
    std::cout << "Is y_a numeric? " << y_a->IsNumeric() << std::endl;
    std::cout << "Is z_a numeric? " << z_a->IsNumeric() << std::endl;
    std::cout << "Is rho_a numeric? " << fluiddata_a->IsNumeric() << std::endl;

    vtkSmartPointer<vtkPoints> grid_points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkFloatArray> fluid_vals = vtkSmartPointer<vtkFloatArray>::New();
    fluid_vals->SetNumberOfComponents(1);

    float x, y, z, fluid_val, r_temp, r_max;

    r_max = CenterLineRadius(x_a->GetVariantValue(x_a->GetNumberOfValues()-1).ToFloat(), y_a->GetVariantValue(y_a->GetNumberOfValues()-1).ToFloat()); // Assume simdata is sorted

    std::cout << "x_a first VariantValue " << x_a->GetVariantValue(0).ToFloat() << std::endl;
    std::cout << "y_a first VariantValue " << y_a->GetVariantValue(0) << std::endl;
    std::cout << "z_a first VariantValue " << z_a->GetVariantValue(0) << std::endl;
    std::cout << "fluid_a first VariantValue " << fluiddata_a->GetVariantValue(0) << std::endl;

    std::cout << "fluid_a number of values " << fluiddata_a->GetNumberOfValues() << std::endl;

    std::cout << "Inserting values " << std::endl;
    for (vtkIdType l = 0; l < fluiddata_a->GetNumberOfValues(); l++){
        if (l % 1000 == 0){
            std::cout << "Inserting l = " << l << std::endl;
        }
        x = x_a->GetVariantValue(l).ToFloat();
        y = y_a->GetVariantValue(l).ToFloat();
        z = z_a->GetVariantValue(l).ToFloat();
        fluid_val = fluiddata_a->GetVariantValue(l).ToFloat();

        r_temp = CenterLineRadius(x, y);
        if (r_temp < 0.25 * r_max) { // Want to only visualize the pinch
            grid_points->InsertNextPoint(x, y, z);
            fluid_vals->InsertNextValue(fluid_val);
        }
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

    vtkSmartPointer<vtkScalarBarActor> scalarBar = vtkSmartPointer<vtkScalarBarActor>::New();
    scalarBar->SetLookupTable(mapper->GetLookupTable());
    scalarBar->SetTitle("Rho v_z");
    scalarBar->SetNumberOfLabels(4);

    // Create a lookup table to share between the mapper and the scalarbar.
    vtkSmartPointer<vtkLookupTable> hueLut = vtkSmartPointer<vtkLookupTable>::New();
    hueLut->SetTableRange(0, 1);
    hueLut->SetHueRange(0, 1);
    hueLut->SetSaturationRange(1, 1);
    hueLut->SetValueRange(1, 1);
    hueLut->Build();

    mapper->SetLookupTable(hueLut);
    scalarBar->SetLookupTable(hueLut);

    // Create renderer
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->AddActor(actor);
    renderer->AddActor(scalarBar);
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

    return 0;
}

void PrintTableContent(vtkSmartPointer<vtkTable> table){
    int numRows = table->GetNumberOfRows();
    int numCols = table->GetNumberOfColumns();

    std::cout << "numRows: " << numRows << std::endl;
    std::cout << "numCols: " << numCols << std::endl;
}

float CenterLineRadius(float x, float y){
    return sqrt(pow(x,2) + pow(y,2));
}