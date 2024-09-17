cmake_minimum_required(VERSION 3.12)
project(LibraryTest LANGUAGES CUDA CXX)
find_package(CUDA REQUIRED)

# Add library target for hellolauncher.cu
add_library(hellolauncher_lib library/helloLauncher.cu)

set(SOURCE_FILES
    main.cpp
    library/callLauncher.cu
)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

add_executable(hello_launcher_cu main.cu)
add_executable(hello_launcher_cpp ${SOURCE_FILES})

set_source_files_properties(callLauncher.cu PROPERTIES LANGUAGE CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")

target_include_directories(hello_launcher_cu PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(hello_launcher_cpp PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(hello_launcher_cu PRIVATE hellolauncher_lib)
target_link_libraries(hello_launcher_cpp PRIVATE hellolauncher_lib)

# Set CUDA_ARCHITECTURES
set_target_properties(hellolauncher_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(hello_launcher_cu PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(hello_launcher_cpp PROPERTIES CUDA_ARCHITECTURES 75)

