cmake_minimum_required(VERSION 3.12)
project(imhd-visualize LANGUAGES CXX C)

find_package(HDF5 REQUIRED COMPONENTS C HL)
find_package(VTK REQUIRED)

include_directories(${HDF5_INCLUDE_DIRS})

set(VISUALIZATION_ROOT "..")
set(PROJECT_ROOT "../..")

add_executable(viewframe_driver view_frame/driver.cpp)
add_executable(view_frame view_frame/view_frame.cpp)

target_link_libraries(viewframe_driver PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(view_frame PRIVATE ${VTK_LIBRARIES})