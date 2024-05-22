cmake_minimum_required(VERSION 3.12)
project(imhd-CUDA LANGUAGES CUDA CXX)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_VERBOSE_MAKEFILE on)

find_package(CUDA REQUIRED)

add_library(kernels_od_lib ../../include/kernels_od.cu)
add_library(initialize_od_lib ../../include/initialize_od.cu)
add_library(utils_lib ../../include/utils.cpp)

add_executable(imhd-debug main.cu)

target_include_directories(imhd-debug PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(imhd-debug PRIVATE kernels_od_lib)
target_link_libraries(imhd-debug PRIVATE initialize_od_lib)
target_link_libraries(imhd-debug PRIVATE utils_lib)

target_compile_options(imhd-debug PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -lineinfo
                        >)
                        
target_compile_options(kernels_od_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -lineinfo
                        >)

# add_compile_options(-g -lineinfo -v)

set_target_properties(initialize_od_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(kernels_od_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(imhd-debug PROPERTIES CUDA_ARCHITECTURES 75)
