cmake_minimum_required(VERSION 3.12)
project(imhd-CUDA_h5write LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)

set(PROJECT_ROOT "../../..")
# message(PROJECT_ROOT IS "${PROJECT_ROOT}")
# include_directories()

include_directories(${HDF5_INCLUDE_DIRS})

add_library(initialize_od_lib ${PROJECT_ROOT}/include/initialize_od.cu)

add_executable(h5write_serial hdf5_write.cu)

target_link_libraries(h5write_serial PRIVATE initialize_od_lib ${HDF5_LIBRARIES})

set_target_properties(initialize_od_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(h5write_serial PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)