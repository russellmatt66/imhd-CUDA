cmake_minimum_required(VERSION 3.12)
project(output-TEST LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED) # This is for running on local RTX2060

add_library(initialize_od_lib ../../../include/initialize_od.cu)
add_library(utils_lib ../../../include/utils.cpp)

add_executable(output-test output.cu)

target_link_libraries(output-test PRIVATE initialize_od_lib utils_lib)

set_target_properties(initialize_od_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(output-test PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)