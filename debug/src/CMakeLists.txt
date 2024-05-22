cmake_minimum_required(VERSION 3.12)
project(imhd-CUDA LANGUAGES CUDA CXX)

set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

find_library(CUFILE_LIBRARY NAMES cufile PATHS "/usr/local/cuda-12.3/targets/x86_64-linux/lib")
find_path(CUFILE_INCLUDE_DIR NAMES cufile.h PATH_SUFFIXES cuda)

add_library(kernels_od_lib ../../include/kernels_od.cu)
add_library(initialize_od_lib ../../include/initialize_od.cu)
add_library(utils_lib ../../include/utils.cpp)

add_executable(imhd-debug main.cu)

target_include_directories(imhd-debug PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(imhd-debug PRIVATE kernels_od_lib)
target_link_libraries(imhd-debug PRIVATE initialize_od_lib)
target_link_libraries(imhd-debug PRIVATE utils_lib)

target_compile_options(imhd-debug PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -v
                        -g
                        -lineinfo
                        >)

set_target_properties(initialize_od_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(kernels_od_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(imhd-debug PROPERTIES CUDA_ARCHITECTURES 75)
