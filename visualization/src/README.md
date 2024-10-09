### Summary
Source code folder for visualization

The need for all the binaries, and shared memory, is based on VTK and HDF5 not mixing well in the same location. 
VTK expects .h5 files to be .vtkhdf5 files, and will intercept HDF5 calls to work with these files in order to replace them with calls that assume they are .vtkhdf5 files.
If they are not, a segfault will occur because where the computer is looking for some metadata (which doesn't exist) will be outside its page table. 