# Overview
Structs and classes don't seem to play successfully with CUDA C/C++. They must be allocated on the host, and then transferred to the device, introducing a need for significant host-device communication in applications with large data volumes.

grid_test/
- Attempts to allocate a `Grid3D` structure to the device

simplest_struct/
- Attempts to allocate the simplest possible struct involving a `float*` to the device