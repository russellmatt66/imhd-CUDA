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
#include <vtkFFMPEGWriter.h>

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <filesystem>
namespace fs = std::filesystem;

int fillImageBuffer(const std::string data_file, std::vector<float>& image_data);
// void renderFrame(const std::string fluid_data_filename, const vtkSmartPointer<vtkPoints> grid_points, const int Nx, const int Ny, const int Nz);

/* CREATE AN ANIMATION OUT OF THE DATA FILES IN A GIVEN DIRECTORY */
int main(int argc, char* argv[]){
    int Nx = std::atoi(argv[1]); // Not too challenging of a requirement to provide data dimensionality - just look inside input file
    int Ny = std::atoi(argv[2]);
    int Nz = std::atoi(argv[3]);
    int which_fluid_data = std::atoi(argv[4]);

    std::string data_directory = "../data/";
    switch (which_fluid_data)
    {
    case 0:
        data_directory += "rho/";
        break;
    case 1:
        data_directory += "rhovx/";
        break;
    case 2:
        data_directory += "rhovy/";
        break;
    case 3:
        data_directory += "rhovz/";
        break;
    case 4:
        data_directory += "Bx/";
        break;
    case 5:
        data_directory += "By/";
        break;
    case 6:
        data_directory += "Bz/";
        break;
    case 7:
        data_directory += "e/";
        break;                                            
    default:
        break;
    }

    int Ng = 0;

    std::vector<float> x_grid, y_grid, z_grid, grid_data;

    Ng = fillImageBuffer("../../data/grid_basis.dat", grid_data); // returns number of points
    for (int l = 0; l < Ng; l++){
        if (l < Nx){ x_grid.push_back(grid_data[l]); } // Look inside "../include/gds.cu: `WriteGridBasisGDS()`" for structure of grid data
        else if (l < Nx + Ny) { y_grid.push_back(grid_data[l]); }
        else { z_grid.push_back(grid_data[l]); }
    }

    vtkSmartPointer<vtkPoints> grid_points = vtkSmartPointer<vtkPoints>::New();

    int Np = Nx * Ny * Nz;
    for (int k = 0; k < Nz; k++){
        for (int i = 0; i < Nx; i++){
            for (int j = 0; j < Ny; j++){
                grid_points->InsertNextPoint(x_grid[i], y_grid[j], z_grid[k]);
            }
        }
    }

    vtkSmartPointer<vtkFFMPEGWriter> avi_writer = vtkSmartPointer<vtkFFMPEGWriter>::New();
    avi_writer->SetFileName("imhd_video.avi");

    std::vector<std::string> imhd_datafiles;
    for (const auto & data_file : fs::directory_iterator(data_directory)){
        std::cout << data_file.path() << std::endl;
        imhd_datafiles.push_back(data_file.path());
    }

    return 0;
}

int fillImageBuffer(const std::string data_file, std::vector<float>& image_data){
    std::cout << "Data file is: " << data_file << std::endl;
    std::ifstream infile(data_file, std::ios::binary);
    if (!infile) {
        std::cerr << "Error opening file." << std::endl;
        return -1;
    }

    std::vector<char> buffer(std::istreambuf_iterator<char>(infile), {});
    size_t num_points = buffer.size() / sizeof(float);
    float temp;
    
    for (size_t l = 0; l < num_points; l++){
        std::memcpy(&temp, &buffer[l * sizeof(float)], sizeof(float));
        std::cout << "l = " << l << ", temp = " << std::endl;
        image_data.push_back(temp);
    }
    infile.close();

    std::cout << "Number of points read: " << num_points << std::endl;
    return num_points;
}

/* DONT NEED THIS, CAN USE vtkFFMPEGWRITER INSTEAD */
// void renderFrame(const std::string fluid_data_filename, const vtkSmartPointer<vtkPoints> grid_points, const int Nx, const int Ny, const int Nz){
//     /* GO THROUGH VISUALIZATION PIPELINE BUT RENDER TO AN IMAGE INSTEAD */
//     return;
// }