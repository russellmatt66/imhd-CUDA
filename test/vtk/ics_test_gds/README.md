# Overview
Part of the project to test writing the initial conditions out, and then visualize them

# Directory Structure
ics.cu
- Initializes simulation data, and writes it to `.dat` files
- Currently writes `rho_ics.dat`, and `xyz_grid.dat`

grid_vis/
- Visualizes grid data in `xyz_grid.dat`

ics_vis/
- Visualizes data in `rho_ics.dat`

# Current Tasks
(1) Complete `ics_vis/`
- 

(2) Investigate following related to pinned memory size limits `ics.cu`:
- ~/Desktop/imhd-CUDA/test/vtk/ics_test_gds$ ./ics_cu 256 256 384
    Beginning of program
    Total pinned memory: 5918 MB
    Free pinned memory: 5449 MB
    Right before writing data to storage
    Total pinned memory: 5918 MB
    Free pinned memory: 4391 MB
    Writing grid data out
    Size of grid data is 288 MB
    cuFile Buffer registration error: access beyond maximum pinned size
    Writing rho data out
    Fluid data successfully written out

- ~/Desktop/imhd-CUDA/test/vtk/ics_test_gds$ ./ics_cu 304 304 592
Beginning of program
Total pinned memory: 5918 MB
Free pinned memory: 5421 MB
Domain boundaries are: 
[x_min, x_max] = [-3.14159, 3.14159]
[y_min, y_max] = [-3.14159, 3.14159]
[z_min, z_max] = [-3.14159, 3.14159]
Spacing is: 
(dx = 0.0207366
dy = 0.0207366
dz = 0.0106314
Right before writing grid data to storage
Total pinned memory: 5918 MB
Free pinned memory: 3115 MB
Writing grid data out
Size of grid data is 626 MB
cuFile Buffer registration error: access beyond maximum pinned size
Right before writing fluid data to storage
Total pinned memory: 5918 MB
Free pinned memory: 3115 MB
Writing rho data out
Size of rho data is 208 MB
Fluid data successfully written out



(3) In `grid_vis/`
- Eliminate host code for reading in points
- Use `vtkImageData` reader
