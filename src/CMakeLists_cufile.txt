cmake_minimum_required(VERSION 3.12)
project(imhd-CUDA LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)

find_library(CUFILE_LIBRARY NAMES cufile PATHS "/usr/local/cuda-12.3/targets/x86_64-linux/lib")
find_path(CUFILE_INCLUDE_DIR NAMES cufile.h PATH_SUFFIXES cuda)

add_library(kernels_od_lib ../include/kernels_od.cu)
add_library(initialize_od_lib ../include/initialize_od.cu)
add_library(gds_lib ../include/gds.cu)
add_library(utils_lib ../include/utils.cpp)

add_executable(imhd-cuda main.cu)
add_executable(write-grid write_grid.cu)

target_include_directories(imhd-cuda PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(write-grid PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(imhd-cuda PRIVATE kernels_od_lib)
target_link_libraries(imhd-cuda PRIVATE initialize_od_lib)
target_link_libraries(imhd-cuda PRIVATE gds_lib)
target_link_libraries(imhd-cuda PRIVATE utils_lib)
# target_link_libraries(imhd_cuda PRIVATE ${CUFILE_LIBRARY} ${CUDA_cudart_LIBRARY})
target_link_libraries(imhd-cuda PRIVATE ${CUFILE_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_cudart_LIBRARY})

target_link_libraries(write-grid PRIVATE kernels_od_lib)
target_link_libraries(write-grid PRIVATE initialize_od_lib)
target_link_libraries(write-grid PRIVATE gds_lib)
target_link_libraries(write-grid PRIVATE utils_lib)
target_link_libraries(write-grid PRIVATE ${CUFILE_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_cudart_LIBRARY})

set_target_properties(initialize_od_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(kernels_od_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(gds_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(imhd-cuda PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(write-grid PROPERTIES CUDA_ARCHITECTURES 75)
