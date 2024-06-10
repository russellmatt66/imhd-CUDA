cmake_minimum_required(VERSION 3.12)
project(imhd-CUDA LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE on)

set(CMAKE_CXX_KEEP_INTERMEDIATE_OBJECTS 1)
set(CMAKE_CUDA_KEEP_INTERMEDIATE_OBJECTS 1)

add_library(helper_lib ../include/helper_functions.cu)
add_library(diffusion_lib ../include/diffusion.cu)
add_library(fluxes_lib ../include/kernels_od_fluxes.cu)
add_library(kernels_od_lib ../include/kernels_od.cu)
add_library(initialize_od_lib ../include/initialize_od.cu)
add_library(intvar_lib ../include/kernels_od_intvar.cu)
add_library(utils_lib ../include/utils.cpp)

add_executable(imhd-cuda main.cu)
add_executable(write-grid write_grid.cu)

target_link_libraries(fluxes_lib PRIVATE helper_lib)
target_link_libraries(intvar_lib PRIVATE fluxes_lib)
target_link_libraries(kernels_od_lib PRIVATE fluxes_lib diffusion_lib helper_lib)
target_link_libraries(imhd-cuda PRIVATE kernels_od_lib initialize_od_lib intvar_lib utils_lib)

target_link_libraries(write-grid PRIVATE kernels_od_lib)
target_link_libraries(write-grid PRIVATE initialize_od_lib)
target_link_libraries(write-grid PRIVATE utils_lib)

set_target_properties(kernels_od_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(initialize_od_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(fluxes_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(intvar_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(diffusion_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(helper_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(imhd-cuda PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(write-grid PROPERTIES CUDA_ARCHITECTURES 75)
