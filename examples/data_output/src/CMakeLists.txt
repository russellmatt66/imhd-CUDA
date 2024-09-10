cmake_minimum_required(VERSION 3.12)
project(imhd-CUDA_h5write LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

add_library(initialize_od_lib)