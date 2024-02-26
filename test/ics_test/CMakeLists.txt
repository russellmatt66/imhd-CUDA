cmake_minimum_required(VERSION 3.12)
project(ics-TEST LANGUAGES CUDA CXX)
find_package(CUDA REQUIRED)

add_library(initialize_od_lib ../include/initialize_od.cu)

add_executable(test_ics ics.cu)

target_include_directories(imhd_cuda PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(test_ics PRIVATE initialize_od_lib)

set_target_properties(initialize_od_lib PROPERTIES CUDA_ARCHITECTURES 75)