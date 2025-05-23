cmake_minimum_required(VERSION 3.12)
project(imhd-CUDA_bench LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

set(PROJECT_ROOT "../../")
message("PROJECT_ROOT: ${PROJECT_ROOT}")

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")

include_directories(${PROJECT_ROOT}/include/on-device)
include_directories(${PROJECT_ROOT}/include/on-device/utils)

# add_library(helper_lib ../../include/helper_functions.cu)
# add_library(diffusion_lib ../../include/diffusion.cu)
# add_library(fluxes_lib ../../include/kernels_od_fluxes.cu)
# add_library(kernels_od_lib ../../include/kernels_od.cu)
# add_library(initialize_od_lib ../../include/initialize_od.cu)
# add_library(intvar_lib ../../include/kernels_od_intvar.cu)
# add_library(utils_lib ../../include/utils.cpp)

add_library(helper_lib ${PROJECT_ROOT}/lib/on-device/helper_functions.cu)
add_library(diffusion_lib ${PROJECT_ROOT}/lib/on-device/diffusion.cu)
add_library(fluxes_lib ${PROJECT_ROOT}/lib/on-device/kernels_od_fluxes.cu)
add_library(kernels_od_lib ${PROJECT_ROOT}/lib/on-device/kernels_od.cu)
add_library(fluidbcs_lib ${PROJECT_ROOT}/lib/on-device/kernels_fluidbcs.cu)
add_library(initialize_od_lib ${PROJECT_ROOT}/lib/on-device/initialize_od.cu)
add_library(intvar_od_lib ${PROJECT_ROOT}/lib/on-device/kernels_od_intvar.cu)
add_library(intvarbcs_lib ${PROJECT_ROOT}/lib/on-device/kernels_intvarbcs.cu)
add_library(utils_lib ${PROJECT_ROOT}/lib/on-device/utils/utils.cpp)

add_executable(imhd-cuda_debug main.cu)
add_executable(imhd-cuda_debug-nodiff no_diffusion.cu)
add_executable(imhd-cuda_debug-nodiffmicro no_diffusion_micro.cu)

target_link_libraries(fluxes_lib PRIVATE helper_lib)
target_link_libraries(intvar_od_lib PRIVATE fluxes_lib)
target_link_libraries(intvarbcs_lib PRIVATE intvar_od_lib)
target_link_libraries(kernels_od_lib PRIVATE fluxes_lib diffusion_lib helper_lib)
target_link_libraries(fluidbcs_lib PRIVATE kernels_od_lib)
target_link_libraries(imhd-cuda_debug PRIVATE kernels_od_lib fluidbcs_lib initialize_od_lib intvar_od_lib intvarbcs_lib utils_lib)
target_link_libraries(imhd-cuda_debug-nodiff PRIVATE kernels_od_lib fluidbcs_lib initialize_od_lib intvar_od_lib intvarbcs_lib utils_lib)
target_link_libraries(imhd-cuda_debug-nodiffmicro PRIVATE kernels_od_lib fluidbcs_lib initialize_od_lib intvar_od_lib intvarbcs_lib utils_lib)

set_target_properties(kernels_od_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(fluidbcs_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(initialize_od_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(fluxes_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(intvar_od_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(intvarbcs_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(diffusion_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(helper_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(imhd-cuda_debug PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(imhd-cuda_debug-nodiff PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(imhd-cuda_debug-nodiffmicro PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(intvarbcs_lib PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)