cmake_minimum_required(VERSION 3.12)
project(ICSTest LANGUAGES CUDA CXX)
find_package(CUDA REQUIRED)

# 
add_library(initialize_od_lib ../../../include/initialize_od.cu)
add_library(utils ../../../include/utils.cpp)

add_executable(ics_cu ics.cu)

target_include_directories(ics_cu PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(ics_cu PRIVATE initialize_od_lib)
target_link_libraries(ics_cu PRIVATE utils)
target_link_libraries(ics_cu PRIVATE "-L/usr/local/cuda/lib64 -lcufile -lcudart -lcuda")

set_target_properties(ics_cu PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(initialize_od_lib PROPERTIES CUDA_ARCHITECTURES 75)