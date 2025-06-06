cmake_minimum_required(VERSION 3.12)
project(imhd-CUDA_OnDevice LANGUAGES CUDA CXX)

# OLD CMAKELISTS.TXT
# find_package(CUDAToolkit REQUIRED) # This is for running on local RTX2060
# # find_package(CUDA REQUIRED) # This is for running on lightning.ai servers

# find_package(MPI REQUIRED)
# find_package(HDF5 REQUIRED COMPONENTS C HL)

message("PROJECT_ROOT: ${PROJECT_ROOT}")

include_directories(${PROJECT_ROOT}/include/on-device)
include_directories(${PROJECT_ROOT}/include/on-device/utils)
# include_directories(${PROJECT_ROOT}/lib/on-device)
# include_directories(${HDF5_INCLUDE_DIRS})
# include_directories(${MPI_INCLUDE_PATH})

# # set(CMAKE_VERBOSE_MAKEFILE on)

# # set(CMAKE_CXX_KEEP_INTERMEDIATE_OBJECTS 1)
# # set(CMAKE_CUDA_KEEP_INTERMEDIATE_OBJECTS 1)

add_library(helper_lib ${PROJECT_ROOT}/lib/on-device/helper_functions.cu)
add_library(diffusion_lib ${PROJECT_ROOT}/lib/on-device/diffusion.cu)
add_library(fluxes_lib ${PROJECT_ROOT}/lib/on-device/kernels_od_fluxes.cu)
add_library(kernels_od_lib ${PROJECT_ROOT}/lib/on-device/kernels_od.cu)
add_library(fluidbcs_lib ${PROJECT_ROOT}/lib/on-device/kernels_fluidbcs.cu)
add_library(initialize_od_lib ${PROJECT_ROOT}/lib/on-device/initialize_od.cu)
add_library(intvar_od_lib ${PROJECT_ROOT}/lib/on-device/kernels_od_intvar.cu)
add_library(intvarbcs_lib ${PROJECT_ROOT}/lib/on-device/kernels_intvarbcs.cu)
add_library(utils_lib ${PROJECT_ROOT}/lib/on-device/utils/utils.cpp)

add_executable(imhd-cuda main.cu)
add_executable(imhd-cuda_nodiff no_diffusion.cu)

# # CUDA libraries + binary
target_link_libraries(fluxes_lib PRIVATE helper_lib)
target_link_libraries(intvar_od_lib PRIVATE fluxes_lib)
target_link_libraries(kernels_od_lib PRIVATE fluxes_lib diffusion_lib helper_lib)
# target_link_libraries(kernels_od_lib PRIVATE fluxes_lib diffusion_lib) # THIS SHOULD WORK - figure out if it does at a later date
target_link_libraries(fluidbcs_lib PRIVATE kernels_od_lib)
target_link_libraries(intvarbcs_lib PRIVATE intvar_od_lib)
target_link_libraries(imhd-cuda PRIVATE kernels_od_lib fluidbcs_lib initialize_od_lib intvar_od_lib intvarbcs_lib utils_lib)
target_link_libraries(imhd-cuda_nodiff PRIVATE kernels_od_lib fluidbcs_lib initialize_od_lib intvar_od_lib intvarbcs_lib utils_lib)

set_target_properties(helper_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(diffusion_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(fluxes_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(kernels_od_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(fluidbcs_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(initialize_od_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(intvar_od_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(intvarbcs_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(imhd-cuda PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(imhd-cuda_nodiff PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)

# FIGURE OUT WHAT TO DO HERE - MORE BUILD FLAGS?
# Observe register pressure
# target_compile_options(initialize_od_lib PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
# )   

# target_compile_options(kernels_od_lib PRIVATE 
#     $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
# )

# target_compile_options(fluidbcs_lib PRIVATE 
#     $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
# )

# target_compile_options(intvar_od_lib PRIVATE 
#     $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
# )

# target_compile_options(intvarbcs_lib PRIVATE 
#     $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
# )