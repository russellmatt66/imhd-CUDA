cmake_minimum_required(VERSION 3.12)
project(imhd-CUDA_bench LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)

add_library(kernels_od_lib ../../include/kernels_od.cu)
add_library(initialize_od_lib ../../include/initialize_od.cu)

add_executable(imhd-cuda_bench main.cu)

target_include_directories(imhd-cuda_bench PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(imhd-cuda_bench PRIVATE kernels_od_lib)
target_link_libraries(imhd-cuda_bench PRIVATE initialize_od_lib)
# target_link_libraries(imhd-cuda PRIVATE ${CUFILE_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_cudart_LIBRARY})

set_target_properties(initialize_od_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(kernels_od_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(imhd-cuda_bench PROPERTIES CUDA_ARCHITECTURES 75)
