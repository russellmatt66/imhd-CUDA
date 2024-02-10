cmake_minimum_required(VERSION 3.12)
project(LibraryTest LANGUAGES CUDA CXX)
find_package(CUDA REQUIRED)

# Just an effing mess

# Add library target for hellolauncher.cu
add_library(hellolauncher_lib library/hellolauncher.cu)

# set(SOURCE_FILES
#     main.cu
#     library/hellolauncher.cu
# )

add_executable(hello_launcher_cu main.cu)
# add_executable(hello_launcher_cpp main.cpp library/hellolauncher.cu)
# add_executable(hello_launcher_cu main.cu library/hellolauncher.cu)

# target_include_directories(hello_launcher_cpp PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(hello_launcher_cu PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(hello_launcher_cu PRIVATE hellolauncher_lib)

# Set CUDA_ARCHITECTURES
# set_target_properties(hello_launcher_cpp PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(hellolauncher_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(hello_launcher_cu PROPERTIES CUDA_ARCHITECTURES 75)


