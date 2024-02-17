cmake_minimum_required(VERSION 3.12)
project(imhd-CUDA LANGUAGES CUDA CXX)
find_package(CUDA REQUIRED)

add_library(kernels_od_lib ../include/kernels_od.cu)

add_executable(imhd_cuda main.cu)

target_include_directories(imhd_cuda PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(imhd_cuda PRIVATE kernels_od_lib)

set_target_properties(kernels_od_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(imhd_cuda PROPERTIES CUDA_ARCHITECTURES 75)
