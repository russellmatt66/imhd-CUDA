### Summary
Code that is not part of the architecture, but is still useful

### Directory Structure
`hdf5_write_all.cu`
- CUDA C++ program that shows Proof Of Concept on how to write data to storage using PHDF5
- Deprecated because it implements a fundamentally flawed strategy
- Can't work because it requires the binary is launched with `mpirun`, meaning the CUDA kernels will be launched separately by each process
- Solution is to put the PHDF5 code into its own binary, copy device data into shared memory, and fork with a system call to `mpirun`