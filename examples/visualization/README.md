### Summary
Proof of Concept codes related to visualizing the output of a simulation using VTK

### Current Tasks
(1) Run `Debug` build for `ics_icsvis_ser.cpp` executable, and check why `H5Fopen()` is not working (it's just dying) 
- Result: `VTK` is intercepting the `H5Fopen()` call, and the `.h5` file is not `vtkhdf5` compatible