cmake_minimum_required(VERSION 3.12)
project(imhd-CUDA_h5write LANGUAGES CUDA CXX C)

find_package(CUDAToolkit REQUIRED)

set(PROJECT_ROOT "../../..")

add_library(initialize_od_lib ${PROJECT_ROOT}/include/initialize_od.cu)

add_executable(fj_evs_compute flux_jacobian_evs.cpp)
add_executable(fj_evs_cu fjevs_compute_example.cu) # CUDA program that initializes grid, and forks `fj_evs`

target_link_libraries(fj_evs_cu PRIVATE initialize_od_lib)

set_target_properties(initialize_od_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(fj_evs_cu PROPERTIES CUDA_ARCHITECTURES 75)