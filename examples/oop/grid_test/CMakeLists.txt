cmake_minimum_required(VERSION 3.12)
project(grid-test LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

add_executable(grid-test grid_test.cu)

set_target_properties(grid-test PROPERTIES CUDA_ARCHITECTURES 75)
