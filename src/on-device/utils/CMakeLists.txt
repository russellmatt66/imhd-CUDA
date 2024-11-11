cmake_minimum_required(VERSION 3.12)
project(imhd-CUDA_Utils LANGUAGES CUDA CXX)

include_directories(${HDF5_INCLUDE_DIRS})
include_directories(${MPI_INCLUDE_PATH})
include_directories(${PROJECT_ROOT}/include/)

add_executable(phdf5_write_all phdf5_write_all.cpp)
add_executable(add_attributes hdf5_write_attributes.cpp)
add_executable(write_grid hdf5_write_grid.cpp)
add_executable(fj_evs_compute compute_stability.cpp)

target_link_libraries(phdf5_write_all PRIVATE ${MPI_LIBRARIES} ${HDF5_LIBRARIES})
target_link_libraries(add_attributes PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(write_grid PRIVATE ${HDF5_LIBRARIES})