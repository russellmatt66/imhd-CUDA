cmake_minimum_required(VERSION 3.12)
project(ICsVertex LANGUAGES C CXX)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

add_executable(vertex_vis vertex.cpp)

target_link_libraries(vertex_vis ${VTK_LIBRARIES})