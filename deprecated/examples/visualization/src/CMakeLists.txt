cmake_minimum_required(VERSION 3.12)
project(imhd-CUDA_vis_example LANGUAGES CUDA CXX C)

find_package(HDF5 REQUIRED COMPONENTS C HL)
find_package(VTK REQUIRED)

include_directories(${HDF5_INCLUDE_DIRS})

add_executable(ics_vis vtk_icsvis_ser.cpp)

target_link_libraries(ics_vis ${VTK_LIBRARIES} ${HDF5_LIBRARIES})