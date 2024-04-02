# Overview
Part of the project to test writing the initial conditions out, and then visualize them

# Directory Structure
ics.cu
- Initializes simulation data, and writes it to `.dat` files
- Currently writes `rho_ics.dat`, and `xyz_grid.dat`
    - 

visualization/
- Visualizes data written out by `ics.cu`

# Current Tasks
(1) Investigate following related to `ics.cu`:
- ~/Desktop/imhd-CUDA/test/vtk/ics_test$ ./ics_cu 256 256 384
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
- Solution approaches: 
-- (a) Try and increase maximum pinned size
-- (b) Track memory usage with a tool to see what's going on in more depth

(2) Write visualization code


