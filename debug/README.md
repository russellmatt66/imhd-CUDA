# Overview
This is where the debugging happens

# Directory Structure
src/
- `main.cu` in here is stripped down version of code with no benchmarking, or data writing functionality implemented

build/
- Where `main.cu` is built

grid-gds/
- For debugging why the grid values are read from the GDS data file as being all 0
    -- Culprit was in `InitializeGrid` device kernel
    -- Problem was related to specifying too many threads per block in the execution configuration 
- Didn't actually need this folder to find bug, so I deleted it, but kept this for posterity

# Current Tasks
(1) Debug out-of-bounds memory accesses in the kernel
- Fundamental issue determined, see (2)

(2) Refactor `FluidAdvance` and `BoundaryConditions` so that each state variable advance has its own kernel
- I think this is one of the core issues, the size of the kernel is probably demanding too many registers 
-- This is EXACTLY the problem, running in `cuda-GDB` gives `Cuda API error detected: cudaLaunchKernel returned (0x2bd)`
-`0x2bd` = 701, error code 701 in the [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038). is `cudaErrorLaunchOutOfResources`
- Compiling library with `-Xptxas=-v` yields:
```
~/Desktop/imhd-CUDA/debug/build$ make
[ 12%] Building CUDA object CMakeFiles/kernels_od_lib.dir/home/matt/Desktop/imhd-CUDA/include/kernels_od.cu.o
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z11SwapSimDataPfS_S_S_S_S_S_S_PKfS1_S1_S1_S1_S1_S1_S1_iii' for 'sm_75'
ptxas info    : Function properties for _Z11SwapSimDataPfS_S_S_S_S_S_S_PKfS1_S1_S1_S1_S1_S1_S1_iii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 20 registers, 492 bytes cmem[0]
ptxas info    : Compiling entry function '_Z18BoundaryConditionsPfS_S_S_S_S_S_S_PKfS1_S1_S1_S1_S1_S1_S1_S_S_S_S_S_S_S_S_fffffiii' for 'sm_75'
ptxas info    : Function properties for _Z18BoundaryConditionsPfS_S_S_S_S_S_S_PKfS1_S1_S1_S1_S1_S1_S1_S_S_S_S_S_S_S_S_fffffiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 162 registers, 576 bytes cmem[0], 192 bytes cmem[2]
ptxas info    : Function properties for __internal_accurate_pow
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Compiling entry function '_Z12FluidAdvancePfS_S_S_S_S_S_S_PKfS1_S1_S1_S1_S1_S1_S1_S_S_S_S_S_S_S_S_fffffiii' for 'sm_75'
ptxas info    : Function properties for _Z12FluidAdvancePfS_S_S_S_S_S_S_PKfS1_S1_S1_S1_S1_S1_S1_S_S_S_S_S_S_S_S_fffffiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 128 registers, 576 bytes cmem[0], 184 bytes cmem[2]
ptxas info    : Function properties for __internal_accurate_pow
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```
-- The above means that, for 1024 threads per block, `BoundaryConditions` requires 162k register memory per SM (RTX2060 has 64k), and `FluidAdvance` requires 128k. 

(3) Try launching `FluidAdvance` with execution configuration whose resource requirements can be satisfied.