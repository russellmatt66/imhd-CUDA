# Summary
This is the build folder for the on-device architecture

# Input File
`Nt`
- This is the number of timesteps the solver will take
- INT32
- Greater than 0
- E.g., `40`

`Nx`
- This is the number of grid points in the x-direction
- INT32
- Greater than 0
- Be cognizant of data volume
- E.g., `64`

`Ny`
- This is the number of grid points in the y-direction
- INT32
- Greater than 0
- Be cognizant of data volume
- E.g., `64`

`Nz`
- This is the number of grid points in the z-direction
- INT32
- Greater than 0
- Be cognizant of data volume
- E.g., `64`

`J0`
- Amplitude of the current
- FP32 
- E.g., `1.0`

`D`
- Numerical diffusivity
- FP32
- E.g., `1.0`

`x_min`
- Lower bound of x-axis
- FP32
- E.g., `-3.14159`

`x_max`
- Upper bound of x-axis
- FP32
- E.g., `3.14159`

`y_min`
- Lower bound of y-axis
- FP32
- E.g., `-3.14159`

`y_max`
- Upper bound of y-axis
- FP32
- E.g., `3.14159`

`z_min`
- Lower bound of z-axis
- FP32
- E.g., `-3.14159`

`z_max`
- Upper bound of z-axis
- FP32
- E.g., `3.14159`

`dt`
- Timestep
- Choose one that satisfies CFL number
- E.g., `0.001`

`path_to_data`
- Where the data files are to be written
- E.g.,`../data/on-device/`

`phdf5_bin_name`
- Where the binary that writes data out with Parallel HDF5 lives
- Highly recommend NO CHANGE
- E.g., `on-device/utils/phdf5_write_all`

`attr_bin_name`
- Where the binary that writes attributes to data files lives
- Highly recommend NO CHANGE
- E.g., `on-device/utils/add_attributes`

`wgrid_bin_name`
- Where the binary that writes grid values to data files lives
- Highly recommend NO CHANGE
- E.g., `on-device/utils/write_grid`

`eigen_bin_name`
- Where the binary that computes CFL criterion on every gridpoint lives 
- Highly recommend NO CHANGE 
- E.g., `none` if you don't want to run it during the simulation
- E.g., `on-device/utils/fj_evs_compute` if you do

`num_proc`
- Number of processors used in the PHDF5 write
- E.g., `4`

`meshblockdims_xthreads`
- Number of threads in the x-dimension of the threadblocks that create the mesh
- Execution config. threadblock rules apply 
- E.g., `8`

`meshblockdims_ythreads`
- Number of threads in the y-dimension of the threadblocks that create the mesh 
- Execution config. threadblock rules apply 
- E.g., `8`

`meshblockdims_zthreads`
- Number of threads in the z-dimension of the threadblocks that create the mesh 
- Execution config. threadblock rules apply 
- E.g., `8`

`initblockdims_xthreads`
- Number of threads in the x-dimension of the threadblocks that initialize data
- Execution config. threadblock rules apply 
- E.g., `8`

`initblockdims_ythreads`
- Number of threads in the y-dimension of the threadblocks that initialize data
- Execution config. threadblock rules apply 
- E.g., `8`

`initblockdims_zthreads`
- Number of threads in the z-dimension of the threadblocks that initialize data 
- Execution config. threadblock rules apply 
- E.g., `8`

`intvarblockdims_xthreads`
- Number of threads in the x-dimension of the threadblocks that process Qint
- Execution config. threadblock rules apply 
- E.g., `8`

`intvarblockdims_ythreads`
- Number of threads in the y-dimension of the threadblocks that process Qint
- Execution config. threadblock rules apply 
- E.g., `8`

`intvarblockdims_zthreads`
- Number of threads in the z-dimension of the threadblocks that process Qint 
- Execution config. threadblock rules apply 
- E.g., `8`

`fluidvarblockdims_xthreads`
- Number of threads in the x-dimension of the threadblocks that process Q
- Execution config. threadblock rules apply 
- E.g., `8`

`fluidvarblockdims_ythreads`
- Number of threads in the y-dimension of the threadblocks that process Q
- Execution config. threadblock rules apply 
- E.g., `8`

`fluidvarblockdims_zthreads`
- Number of threads in the z-dimension of the threadblocks that process Q 
- Execution config. threadblock rules apply 
- E.g., `8`
