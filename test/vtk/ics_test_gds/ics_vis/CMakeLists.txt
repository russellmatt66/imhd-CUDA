cmake_minimum_required(VERSION 3.0)
project(renderRho)

# Find VTK package
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

# Add executable
add_executable(render_rho vis_rho.cpp)

# Link VTK library
target_link_libraries(render_rho ${VTK_LIBRARIES})