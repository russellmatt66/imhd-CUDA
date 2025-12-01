# Summary
This is the build folder for the on-device architecture

# Input File
`sim_init_keystring`
- This selects what initial conditions to populate the data with
- Key that hashes for a C++ function which launches a kernel to initialize the computational volume
- See `class SimulationInitializer` for more details 
- E.g., `screwpinch`, `screwpinch-stride`

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
- Where the binary that computes CFL criterion using on every gridpoint lives 
- Highly recommend NO CHANGE 
- E.g., `none` if you don't want to run it during the simulation
- E.g., `on-device/utils/fj_evs_compute` if you do

`num_proc`
- Number of processors used in the PHDF5 write
- E.g., `4`

`xgrid_threads`
- Number of threads in the x-dimension of the threadblocks that create the mesh
- Execution config. threadblock rules apply 
- E.g., `8`

`ygrid_threads`
- Number of threads in the y-dimension of the threadblocks that create the mesh 
- Execution config. threadblock rules apply 
- E.g., `8`

`zgrid_threads`
- Number of threads in the z-dimension of the threadblocks that create the mesh 
- Execution config. threadblock rules apply 
- E.g., `8`

`init_xthreads`
- Number of threads in the x-dimension of the threadblocks that initialize data
- Execution config. threadblock rules apply 
- E.g., `8`

`init_ythreads`
- Number of threads in the y-dimension of the threadblocks that initialize data
- Execution config. threadblock rules apply 
- E.g., `8`

`init_zthreads`
- Number of threads in the z-dimension of the threadblocks that initialize data 
- Execution config. threadblock rules apply 
- E.g., `8`

`FA_xthreads`
- Number of threads in the x-dimension of the threadblocks that launch the kernels for calculating Q and Qint on the interior of the computational domain
- Execution config. threadblock rules apply 
- E.g., `8`

`FA_ythreads`
- Number of threads in the y-dimension of the threadblocks that launch the kernels for calculating Q and Qint on the interior of the computational domain
- Execution config. threadblock rules apply 
- E.g., `8`

`FA_zthreads`
- Number of threads in the z-dimension of the threadblocks that launch the kernels for calculating Q and Qint on the interior of the computational domain 
- Execution config. threadblock rules apply 
- E.g., `8`

`BCLeftRight_xthreads`
- Number of threads in the x-dimension of the threadblocks that process Q and Qint on the Left and Right faces of the computational domain
- Left: {i,j,k} \in {[0, Nx-2], Ny-1, [1, Nz-2]}
- Right: {i,j,k} \in {[0, Nx-2], 0, [1, Nz-2]}
- Execution config. threadblock rules apply 
- E.g., `8`

`BCLeftRight_zthreads`
- Number of threads in the z-dimension of the threadblocks that process Q and Qint on the Left and Right faces of the computational domain
- Left: {i,j,k} \in {[0, Nx-2], Ny-1, [1, Nz-2]}
- Right: {i,j,k} \in {[0, Nx-2], 0, [1, Nz-2]}
- Execution config. threadblock rules apply 
- E.g., `8`

`BCTopBottom_ythreads`
- Number of threads in the y-dimension of the threadblocks that process Q and Qint on the Top and Bottom faces of the computational domain 
- Execution config. threadblock rules apply 
- Top: (i,j,k) \in {0, [0, Ny-2], [1, Nz-2]}
- Bottom: (i,j,k) \in {Nx-1, [0, Ny-2], [1, Nz-2]}
- E.g., `8`

`PBC_xthreads`
- Number of threads in the first dimension of the threadblocks that process periodic boundary conditions across a particular pair of faces of the computational domain

`PBC_ythreads`
- Number of threads in the second dimension of the threadblocks that process periodic boundary conditions across a particular pair of faces of the computational domain

`QintBC_FrontLeft_xthreads`
- Number of threads in the x-dimension of the threadblocks that process Qint along the FrontLeft edge of the computational domain
- FrontLeft: (i,j,k) \in {[0, Nx-2], Ny-1, 0} 

`QintBC_FrontBottom_ythreads`
- Number of threads in the y-dimension of the threadblocks that process Qint along the FrontBottom edge of the computational domain
- FrontBottom: (i,j,k) \in {Nx-1, [0, Ny-2], 0}

`QintBC_BottomLeft_zthreads`
- Number of threads in the z-dimension of the threadblocks that process Qint
along the BottomLeft edge of the computational domain
- BottomLeft: (i,j,k) \in {Nx-1, Ny-1, [0, Nz-2]}

`SM_mult_grid_x`
- Multiplies the number of device SMs to get the number of blocks in the execution grid of the kernels which initialize the grid x-data

`SM_mult_grid_y`
- Multiplies the number of device SMs to get the number of blocks in the execution grid of the kernels which initialize the grid y-data

`SM_mult_grid_z`
- Multiplies the number of device SMs to get the number of blocks in the execution grid of the kernels which initialize the grid z-data

`SM_mult_init_x`
- Multiplies the number of device SMs to get the number of x-blocks in the execution grid of the kernels which initialize the plasma state

`SM_mult_init_y`
- Multiplies the number of device SMs to get the number of y-blocks in the execution grid of the kernels which initialize the plasma state

`SM_mult_init_z`
- Multiplies the number of device SMs to get the number of z-blocks in the execution grid of the kernels which initialize the plasma state

`SM_mult_FA_x`
- Multiplies the number of device SMs to get the number of x-blocks in the execution grid of the kernels which advance the Q and Qint variables

`SM_mult_FA_y`
- Multiplies the number of device SMs to get the number of y-blocks in the execution grid of the kernels which advance the Q and Qint variables

`SM_mult_FA_z`
- Multiplies the number of device SMs to get the number of z-blocks in the execution grid of the kernels which advance the Q and Qint variables

`fvbc_init_keystring`
- Keystring for hashing which set of initialization kernels to run  

`ivk_keystring`
- Keystring for hashing which set of kernels to advance the state of Qint on the interior of the computational domain  
 
`ivbc_keystring`
- Keystring for hashing which set of kernels to run for calculating the state of Qint on the boundaries

`fvk_keystring`
- Keystring for hashing which set of kernels to run for advancing the state of Q on the interior of the computational domain 

`fvbc_loop_keystring`
- Keystring for hashing which set of kernels to run for calculating the state of Q on the boundaries of the computational domain during the main loop of the simulation   