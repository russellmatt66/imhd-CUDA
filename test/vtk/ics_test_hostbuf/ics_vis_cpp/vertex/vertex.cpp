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

#include <string>
#include <iostream>

void PrintTableContent(vtkSmartPointer<vtkTable> table);

int main(int argc, char* argv[]){
    std::string simdata_location = "../../data/sim_data.csv";
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

    // vtkFloatArray* x = vtkFloatArray::SafeDownCast(simTable->GetColumn(0));
    // vtkFloatArray* y = vtkFloatArray::SafeDownCast(simTable->GetColumn(1));
    // vtkFloatArray* z = vtkFloatArray::SafeDownCast(simTable->GetColumn(2));
    // vtkFloatArray* rho = vtkFloatArray::SafeDownCast(simTable->GetColumn(6));
    
    // vtkSmartPointer<vtkFloatArray> x = vtkSmartPointer<vtkFloatArray>::New();
    // vtkSmartPointer<vtkFloatArray> y = vtkSmartPointer<vtkFloatArray>::New();
    // vtkSmartPointer<vtkFloatArray> z = vtkSmartPointer<vtkFloatArray>::New();
    // vtkSmartPointer<vtkFloatArray> rho = vtkSmartPointer<vtkFloatArray>::New();

    // vtkSmartPointer<vtkFloatArray> x = vtkFloatArray::SafeDownCast(simTable->GetColumn(0));
    // vtkSmartPointer<vtkFloatArray> y = vtkFloatArray::SafeDownCast(simTable->GetColumn(1));
    // vtkSmartPointer<vtkFloatArray> z = vtkFloatArray::SafeDownCast(simTable->GetColumn(2));
    // vtkSmartPointer<vtkFloatArray> rho = vtkFloatArray::SafeDownCast(simTable->GetColumn(6));

    vtkAbstractArray* x_a = simTable->GetColumn(0);
    vtkAbstractArray* y_a = simTable->GetColumn(1);
    vtkAbstractArray* z_a = simTable->GetColumn(2);
    vtkAbstractArray* rho_a = simTable->GetColumn(6);

    std::cout << x_a << std::endl;

    // vtkFloatArray* x = vtkFloatArray::SafeDownCast(x_a);
    // vtkFloatArray* y = vtkFloatArray::SafeDownCast(y_a);
    // vtkFloatArray* z = vtkFloatArray::SafeDownCast(z_a);
    // vtkFloatArray* rho = vtkFloatArray::SafeDownCast(rho_a);

    // std::cout << x << std::endl;
    // std::cout << x->GetNumberOfValues() << std::endl;

    if (!x_a || !y_a || !z_a || !rho_a)
    {
        std::cout << "!x: " << !x_a << std::endl; 
        std::cout << "!y: " << !y_a << std::endl; 
        std::cout << "!z: " << !z_a << std::endl; 
        std::cout << "!rho: " << !rho_a << std::endl; 
        std::cerr << "Error: One or more columns not found in the table." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Is x_a numeric? " << x_a->IsNumeric() << std::endl;
    std::cout << "Is y_a numeric? " << y_a->IsNumeric() << std::endl;
    std::cout << "Is z_a numeric? " << z_a->IsNumeric() << std::endl;
    std::cout << "Is rho_a numeric? " << rho_a->IsNumeric() << std::endl;

    vtkSmartPointer<vtkPoints> grid_points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkFloatArray> fluid_vals = vtkSmartPointer<vtkFloatArray>::New();
    fluid_vals->SetNumberOfComponents(1);
    // fluid_vals->SetName("rho");

    // std::cout << "Inserting values " << std::endl;
    // for (vtkIdType l = 0; l < rho_a->GetNumberOfTuples(); l++){
    //     std::cout << "Inserting l = " << l << std::endl;
    //     grid_points->InsertNextPoint(x->GetValue(l), y->GetValue(l), z->GetValue(l));
    //     fluid_vals->SetValue(l, rho->GetValue(l));
    // }
    // std::cout << "Values inserted successfully" << std::endl;

    float x, y, z, rho;

    std::cout << "x_a first VariantValue " << x_a->GetVariantValue(0).ToFloat() << std::endl;
    std::cout << "y_a first VariantValue " << y_a->GetVariantValue(0) << std::endl;
    std::cout << "z_a first VariantValue " << z_a->GetVariantValue(0) << std::endl;
    std::cout << "rho_a first VariantValue " << rho_a->GetVariantValue(0) << std::endl;

    std::cout << "rho_a number of values " << rho_a->GetNumberOfValues() << std::endl;

    std::cout << "Inserting values " << std::endl;
    for (vtkIdType l = 0; l < rho_a->GetNumberOfValues(); l++){
        if (l % 1000 == 0){
            std::cout << "Inserting l = " << l << std::endl;
        }
        x = x_a->GetVariantValue(l).ToFloat();
        y = y_a->GetVariantValue(l).ToFloat();
        z = z_a->GetVariantValue(l).ToFloat();
        rho = rho_a->GetVariantValue(l).ToFloat();
        grid_points->InsertNextPoint(x, y, z);
        fluid_vals->InsertNextValue(rho);
        // grid_points->InsertNextPoint(x_a->GetVariantValue(l).ToFloat(), y_a->GetVariantValue(l).ToFloat(), z_a->GetVariantValue(l).ToFloat());
        // fluid_vals->SetValue(l, rho_a->GetVariantValue(l).ToFloat());
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

    free(x_a);
    free(y_a);
    free(z_a);
    free(rho_a);
    return 0;
}

void PrintTableContent(vtkSmartPointer<vtkTable> table){
    int numRows = table->GetNumberOfRows();
    int numCols = table->GetNumberOfColumns();

    std::cout << "numRows: " << numRows << std::endl;
    std::cout << "numCols: " << numCols << std::endl;
}
