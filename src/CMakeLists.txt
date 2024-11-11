cmake_minimum_required(VERSION 3.12)
project(imhd-CUDA LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED) # This is for running on local RTX2060
# find_package(CUDA REQUIRED) # This is for running on lightning.ai servers

find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)

get_filename_component(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/.." ABSOLUTE) # Referenced in sub-directory CMakeLists.txt
set(PROJECT_ROOT "${PROJECT_ROOT}" CACHE PATH "Root directory of project") 

option(ARCH_ONDEV "Enable build for on-device GPU solver" OFF)
option(ARCH_HOSTDEV "Enable build for host-device GPU solver" OFF)
option(ARCH_CPUSER "Enable build for serial host solver" OFF)
option(ARCH_CPUPAR "Enable build for parallel host solver" OFF)

if (ARCH_ONDEV)
    add_subdirectory(${PROJECT_ROOT}/src/on-device/utils ${PROJECT_ROOT}/build/on-device/utils)
    add_subdirectory(${PROJECT_ROOT}/src/on-device ${PROJECT_ROOT}/build/on-device)
endif()

if (ARCH_HOSTDEV)
    add_subdirectory(host-device)
endif()

if (ARCH_CPUSER)
    add_subdirectory(serial)
endif()

if (ARCH_CPUPAR)
    add_subdirectory(par)
endif()


# OLD ON-DEVICE CMAKELISTS
# cmake_minimum_required(VERSION 3.12)
# project(imhd-CUDA LANGUAGES CUDA CXX)

# find_package(CUDAToolkit REQUIRED) # This is for running on local RTX2060
# # find_package(CUDA REQUIRED) # This is for running on lightning.ai servers

# find_package(MPI REQUIRED)
# find_package(HDF5 REQUIRED COMPONENTS C HL)

# set(PROJECT_ROOT "..")

# include_directories(${PROJECT_ROOT}/include/)
# include_directories(${PROJECT_ROOT}/include/on-device)
# include_directories(${HDF5_INCLUDE_DIRS})
# include_directories(${MPI_INCLUDE_PATH})

# # set(CMAKE_VERBOSE_MAKEFILE on)

# # set(CMAKE_CXX_KEEP_INTERMEDIATE_OBJECTS 1)
# # set(CMAKE_CUDA_KEEP_INTERMEDIATE_OBJECTS 1)

# add_library(helper_lib ${PROJECT_ROOT}/lib/on-device/helper_functions.cu)
# add_library(diffusion_lib ${PROJECT_ROOT}/lib/on-device/diffusion.cu)
# add_library(fluxes_lib ${PROJECT_ROOT}/lib/on-device/kernels_od_fluxes.cu)
# add_library(kernels_od_lib ${PROJECT_ROOT}/lib/on-device/kernels_od.cu)
# add_library(initialize_od_lib ${PROJECT_ROOT}/lib/on-device/initialize_od.cu)
# add_library(intvar_lib ${PROJECT_ROOT}/lib/on-device/kernels_od_intvar.cu)
# add_library(utils_lib ${PROJECT_ROOT}/lib/utils.cpp)

# add_executable(imhd-cuda main.cu)
# add_executable(phdf5_write_all phdf5_write_all.cpp)
# add_executable(add_attributes hdf5_write_attributes.cpp)
# add_executable(write_grid hdf5_write_grid.cpp)
# add_executable(fj_evs_compute compute_stability.cpp)

# # CUDA libraries + binary
# target_link_libraries(fluxes_lib PRIVATE helper_lib)
# target_link_libraries(intvar_lib PRIVATE fluxes_lib)
# target_link_libraries(kernels_od_lib PRIVATE fluxes_lib diffusion_lib helper_lib)
# target_link_libraries(imhd-cuda PRIVATE kernels_od_lib initialize_od_lib intvar_lib utils_lib)

# # C++ binaries
# target_link_libraries(phdf5_write_all PRIVATE ${MPI_LIBRARIES} ${HDF5_LIBRARIES})
# target_link_libraries(add_attributes PRIVATE ${HDF5_LIBRARIES})
# target_link_libraries(write_grid PRIVATE ${HDF5_LIBRARIES})

# set_target_properties(kernels_od_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(initialize_od_lib PROPERTIES CUDA_ARCHITECTURES 75)
# set_target_properties(fluxes_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(intvar_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(diffusion_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(helper_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(imhd-cuda PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)