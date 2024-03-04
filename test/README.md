# Directory Structure
problem_size/
- Data volume calculations

library_test/
- Demonstrates how to write a library of kernels, and then call this from inside a C or C++ file.

grid_test/
- Attempts to allocate a `Grid3D` structure to the device

simplest_struct/
- Attempts to allocate the simplest possible struct involving a `float*` to the device

cufile_test/
- RUN INSTRUCTIONS
    -`$ nvcc -o test cufile_test.cu -L/usr/local/cuda/lib64 -lcufile -lcudart -lcuda -arch=sm_75`
    - `lcufile`: Links against cuFile
    - `lcudart`: Links against CUDA runtime library
    - `lcuda`: Links against the CUDA driver library

write_cufilefunc_test/
- Tests function for writing data to storage with GDS

inverse_map/
- Python file to map from the linear index of the data arrays, l, to the tensor indices (i, j, k).

# Overview
Structs and classes don't seem to play successfully with CUDA C/C++. They must be allocated on the host, and then transferred to the device, introducing 