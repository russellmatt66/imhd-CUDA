# Overview
Header files

# Directory Structure
`on-device/`
- CUDA kernels that implement an "on-device" solver architecture

`host-device/`
- CUDA kernels that implement a "host-device" solver architecture

`serial/`
- C++ library that implements single-threaded CPU solver

`par/`
- C++ library that implements parallelized CPU solver

`utils/`
- Utilities for the various solver architectures

`cufile_sample_utils.h`
- Utilities for working with CUFile

`gds.cuh`
- Functions for using GDS to write device data directly to storage

`util.cuh`
- Utility functions for working with GPU