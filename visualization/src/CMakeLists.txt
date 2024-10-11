cmake_minimum_required(VERSION 3.12)
project(imhd-visualize LANGUAGES CXX C)

find_package(HDF5 REQUIRED COMPONENTS C HL)
find_package(VTK REQUIRED COMPONENTS 
    IOFFMPEG
    RenderingContextOpenGL2
    RenderingCore
    RenderingFreeType
    RenderingGL2PSOpenGL2
    RenderingOpenGL2
    RenderingVolume
    RenderingVolumeOpenGL2
    )

include_directories(${HDF5_INCLUDE_DIRS})

set(VISUALIZATION_ROOT "..")
set(PROJECT_ROOT "../..")

# View a given frame
add_executable(viewframe_driver view_frame/driver.cpp)
add_executable(view_frame view_frame/view_frame.cpp)

target_link_libraries(viewframe_driver PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(view_frame PRIVATE ${VTK_LIBRARIES})

# Make a movie out of all the data
add_executable(makemovie_driver make_movie/driver.cpp)
add_executable(read_fluid_data make_movie/read_fluid_data.cpp)
add_executable(read_grid_data make_movie/read_grid_data.cpp)

target_link_libraries(makemovie_driver PRIVATE ${VTK_LIBRARIES})
target_link_libraries(read_fluid_data PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(read_grid_data PRIVATE ${HDF5_LIBRARIES})

vtk_module_autoinit(
    TARGETS view_frame makemovie_driver
    MODULES ${VTK_LIBRARIES}
)