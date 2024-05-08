# Overview
Visualizes the simulation output using VTK

# Current Tasks
(1) Complete `create_video_csv.cpp`
 

(2) Complete `create_video_dat.cpp`
- Refactor writing data into grid buffer after completing the code to write it out `../src/write_grid.cu`
- Problem writing 210 MB of data out with GDS (> max pinned size)