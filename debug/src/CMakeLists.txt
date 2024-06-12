cmake_minimum_required(VERSION 3.12)
project(imhd-CUDA LANGUAGES CUDA CXX)

set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_VERBOSE_MAKEFILE on)

find_package(CUDAToolkit REQUIRED)

# Define source files
set(KERNELS_OD_SRC ../../include/kernels_od.cu)
set(INITIALIZE_OD_SRC ../../include/initialize_od.cu)
set(FLUX_SRC ../../include/kernels_od_fluxes.cu)
set(DIFFUSION_SRC ../../include/diffusion.cu)
set(INTVAR_SRC ../../include/kernels_od_intvar.cu)
set(HELPER_SRC ../../include/helper_functions.cu)
set(UTILS_SRC ../../include/utils.cpp)
set(MAIN_SRC main.cu)

# Add libraries
add_library(helper_lib ${HELPER_SRC})
add_library(diffusion_lib ${DIFFUSION_SRC})
add_library(fluxes_lib ${FLUX_SRC})
add_library(kernels_od_lib ${KERNELS_OD_SRC})
add_library(initialize_od_lib ${INITIALIZE_OD_SRC})
add_library(intvar_lib ${INTVAR_SRC})
add_library(utils_lib ${UTILS_SRC})

# Add executable
add_executable(imhd-debug ${MAIN_SRC})

# Include directories
target_include_directories(imhd-debug PRIVATE ${CUDA_INCLUDE_DIRS})

# Link libraries
target_link_libraries(kernels_od_lib PRIVATE fluxes_lib diffusion_lib helper_lib)
target_link_libraries(imhd-debug PRIVATE kernels_od_lib initialize_od_lib intvar_lib utils_lib)

# Compile options for imhd-debug
target_compile_options(imhd-debug PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -g
                        -G
                        -Xptxas=-v
                        >)
                        
# Needed register requirements, and line info for debugging
target_compile_options(kernels_od_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -lineinfo
                        -Xptxas=-v
                        >)
# Need register requirements
target_compile_options(intvar_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xptxas=-v
                        >)   

target_compile_options(initialize_od_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xptxas=-v
                        >)                  
                        
# Set CUDA architectures
set_target_properties(initialize_od_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(helper_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(diffusion_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(fluxes_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(intvar_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(kernels_od_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(imhd-debug PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON)

# Custom commands to generate PTX files
foreach(source_file ${KERNELS_OD_SRC} ${INITIALIZE_OD_SRC} ${MAIN_SRC})
    get_filename_component(source_file_we ${source_file} NAME_WE)
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/${source_file_we}.ptx
        COMMAND ${CMAKE_CUDA_COMPILER} ${CMAKE_CUDA_FLAGS} -ptx ${CMAKE_CURRENT_SOURCE_DIR}/${source_file} -o ${CMAKE_BINARY_DIR}/${source_file_we}.ptx
        DEPENDS ${source_file}
        COMMENT "Generating PTX for ${source_file}"
        VERBATIM
    )
    list(APPEND PTX_FILES ${CMAKE_BINARY_DIR}/${source_file_we}.ptx)
endforeach()

# Add a custom target to build the PTX files
add_custom_target(build_ptx ALL DEPENDS ${PTX_FILES})

# Ensure the executable depends on the PTX generation
add_dependencies(imhd-debug build_ptx)
