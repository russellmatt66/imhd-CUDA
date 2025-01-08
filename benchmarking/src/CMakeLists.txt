cmake_minimum_required(VERSION 3.12)
project(imhd-CUDA_bench LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

add_library(helper_lib ../../include/helper_functions.cu)
add_library(diffusion_lib ../../include/diffusion.cu)
add_library(fluxes_lib ../../include/kernels_od_fluxes.cu)
add_library(kernels_od_lib ../../include/kernels_od.cu)
add_library(initialize_od_lib ../../include/initialize_od.cu)
add_library(intvar_lib ../../include/kernels_od_intvar.cu)
add_library(utils_lib ../../include/utils.cpp)

add_executable(imhd-cuda_bench manual.cu)
add_executable(imhd-cuda_profile profile.cu)

target_link_libraries(fluxes_lib PRIVATE helper_lib)
target_link_libraries(intvar_lib PRIVATE fluxes_lib)
target_link_libraries(kernels_od_lib PRIVATE fluxes_lib diffusion_lib helper_lib)

target_link_libraries(imhd-cuda_bench PRIVATE kernels_od_lib initialize_od_lib intvar_lib utils_lib)
target_link_libraries(imhd-cuda_profile PRIVATE kernels_od_lib initialize_od_lib intvar_lib utils_lib)

set_target_properties(kernels_od_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(initialize_od_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(fluxes_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(intvar_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(diffusion_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(helper_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(imhd-cuda_bench PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(imhd-cuda_profile PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
