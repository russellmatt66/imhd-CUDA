# Overview
This is where the debugging happens

# Directory Structure
src/
- `main.cu` in here is stripped down version of code with no benchmarking, or data writing functionality implemented

build/
- Where `main.cu` is built

# Current Tasks
Figured out the problem with indexing! 

Problem was with indexing macro:
-`#define IDX3D(i, j, k, Nx, Ny, Nz) (k * (Nx * Ny) + i * Ny + j)`
Macro inserts arguments into expression literally. When argument for `k` is `k-1`, then what is calculated is `k - 1 * (Nx * Ny) + i * Ny + j`
This leads to negative indices for some grid locations, and even when not negative, it will lead to computation being incorrectly applied for `k+1`, `i+1`, and `i-1`. 

Solution is to add parentheses:
-`#define IDX3D(i, j, k, Nx, Ny, Nz) ((k) * (Nx * Ny) + (i) * Ny + j)`
