# Directory Structure
execution_config/
- Python snippets related to calculating execution configurations 

gds/ 
- Snippets related to GPU Direct Storage functionality

vtk/
- Code related to implementing VTK visualization codes

oop/
- Snippets related to implementing object-oriented kernels

problem_size/
- Data volume calculations

library_test/
- Demonstrates how to write a library of kernels, and then call this from inside a C or C++ file.
- Might need to be renamed

inverse_map/
- Python file to map from the linear index of the data arrays, l, to the tensor indices (i, j, k).

# Current Tasks
(1) Visualize initial conditions 
- `./vtk/ics_test_gds`
-- Writes data directly to storage from the device
- `./vtk/ics_test_hostbuf`
-- Writes data to storage via a bounce buffer through the host

