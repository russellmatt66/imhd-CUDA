# Overview
This is the `build` folder for the benchmarking and profiling part of the project

# Build Instructions
Run the following commands in order to build the applications for manually profiling the kernels, and for working with NSight systems.
`make clean`
`cmake ../src`
`make`

# Run Instructions
Two applications are built following the completion of `make`: `imhd-cuda_bench`, and `imhd-cuda_fullprofile`.

`imhd-cuda_fullprofile` is intended for use inside NVIDIA NSight. 

If you just want to get a sense of the walltime of the kernels then use `python imhdLauncher.py` to run `imhd-cuda_bench`
See `../src/README.md` for explanation of what `imhd-cuda_bench` covers
