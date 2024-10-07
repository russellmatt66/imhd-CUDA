# WIP
cmake_minimum_required(VERSION 3.12)
project(IMHD_AVI LANGUAGES CXX)

find_package(VTK COMPONENTS 
    IOFFMPEG
)

if (NOT VTK_FOUND)
  message(FATAL_ERROR "FFMPEG: Unable to find the VTK build folder.")
endif()

add_executable(imhd_avi create_video.cpp)
  
target_link_libraries(imhd_avi PRIVATE ${VTK_LIBRARIES})

# Not sure if this is needed
# vtk_module_autoinit(
#   TARGETS imhd_avi
#   MODULES ${VTK_LIBRARIES}
# )
