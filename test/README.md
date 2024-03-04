# Directory Structure
problem_size/
- Data volume calculations

library_test/
- Demonstrates how to write a library of kernels, and then call this from inside a C or C++ file.

grid_test/
- Attempts to allocate a `Grid3D` structure to the device

simplest_struct/
- Attempts to allocate the simplest possible struct involving a `float*` to the device

inverse_map/
- Python file to map from the linear index of the data arrays, l, to the tensor indices (i, j, k).

# Overview
Structs and classes don't seem to play successfully with CUDA C/C++. They must be allocated on the host, and then transferred to the device, introducing 