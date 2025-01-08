### Overview
Project to implement the Lax-Wendroff scheme with CUDA in order to solve the Ideal MHD system using GPUs.

### Build Instructions
See `build/README.md`

### Directory Structure
`build/`
- Build folder

`src/`
- Source code folder for the drivers

`lib/`
- Library files

`include/`
- Header files

`visualization/`
- VTK lives here

`tests/`
- Source code for the unit tests

### Current Tasks
[O] = "Occomplished"
[] = "(Not Occomplished)"

(1) Clean up project and QoL Refactor
- Go through and update all READMEs
- Go through and clean up stray comments
- QoL:
* `lib/on-device/`: Move definition of `IDX3D` to a single location
* `src/on-device/main.cu`: Add additional arguments for threadblock execution configurations
* `src/on-device/main.cu`: Stream standard output to a log file
* `visualization/src/make_movie/`: Add additional arguments for camera settings, and other parameters
* `visualization/src/view_frame/`: Add additional arguments for camera settings, and other parameters
* `data`: Separate the output data from the different solver versions into their own folders

(2) Stabilize
- Integrate PoC functionality to scan for numerical instability points using `Eigen`, and write data out with `PHDF5`
* P: Above has been done, timestep altered, solver is still unstable. 
* Q: Is CFL being violated?
- Boundary Conditions or Initial Conditions to blame?
- Incorrect implementation of numerical diffusion to blame?
* Noticed that a factor of `\delta t` is missing

(3) Debugging and Tests
- Unit tests of functional components, see `tests/README.md`
- Numerical test problems

(4) Profile simulation
- Compare with CPU code
- Figure out how to profile `304x304x592` case
- Compare GPU and CPU code to directive-based methods, i.e., OpenACC, and OpenMP

(5) Optimize
- Kernels need to be refactored again for global memory coalescing
- Execution configurations should (?) go to 1B1T (test this)
- Shared memory access needs to be implemented in kernels

(6) CPU Code
- Implement single-, and multi-threaded CPU versions of the solver

(7) Build Instructions
- Develop document that describes how to build the project from a fresh install of Ubuntu 22.04
- Enumerate dependencies
* CUDA
* MPI
* HDF5
* VTK

### VCS
- `v1.1`: (10/11/24) End-to-end pipeline is complete. Data is written to .h5 files, and a high-performance visualization pipeline renders both individual frames, as well as the totality as a `.avi` file. 

- `v1.0`: (6/16/24) The project has reached a milestone where it passes the functional correctness testing of `compute-sanitizer` with all end-to-end components. This working draft needs further work to done to visualize the output, improve the performance, clean up the repo, and profile the production case of `304x304x592`. It is stored with a `git tag` of `v1.0`.

### Design
## The Basic Problem
Limited DRAM on GPU (RTX 2060) means storage is at a premium, and thread asynchronicity means that it will not work to launch a binary with an algorithm naively implemented to work directly on the variables in memory, because there is no guarantee for when the work will happen. Meaning, as a consequence of thread asynchronicity, this approach can in general lead to a race condition or a data hazard. The good news is that a race condition should **not** exist for the Lax-Wendroff scheme. Fundamentally, the equation being solved is,

![The Ideal MHD System](https://latex.codecogs.com/svg.image?\frac{\partial\vec{Q}}{\partial&space;t}&plus;\nabla\cdot\overline{\overline{T}}=D\nabla^{2}\vec{Q})

What all these terms are exactly will not be explained here. In the future I will upload a paper explaining things. The numerical method being used to solve this equation here is a *predictor-corrector* method, given by,

![Lax-Wendroff](./.svgs/predictor_corrector.svg)

This is of course absent the diffusion, which is discretized with a second-order central difference approach, and is only present in the first place to add a numerical means of controlling numerical instabilities which may arise. 

## On-Device vs. Host-Device
One solution in this arena of limited VRAM is to update the fluid variables one-by-one, and only have what fluid variables are needed for this task on the device at any given moment. While this sounds simple, in practice implementing it correctly involves significant complexity due to the substantial communication and synchronization involved. The advantage of this `host-device` approach is that, by utilizing host RAM, it will enable simulations that are larger than a purely `on-device` approach where all the necessary data to update fluid variables is stored directly in device memory, at all times. However, the required memory migration, and synchronization barriers, all but ensure that this will be much slower than the `on-device` implementation would be. There is also the question of what *exactly* the architecture, and runtime, of the `host-device` approach would look like. 

## Megakernel vs. Microkernel
This open-source project is all about performance. Explicit GPU, like this code, is where the CFD industry is headed, but the Lax-Wendroff scheme is, to put things colorfully, a bit long in the tooth. It may have been cutting-edge decades ago, but in the present moment the frontier of CFD is the Lattice Boltzmann Method (LBM), and the frontier of Computational Magnetohydrodynamics (CMHD) is the Discontinuous Galerkin Finite Element Method (DG-FEM). Therefore, this project is all about performance. 

Performant CPU code is much more straightforward than performant GPU code when it comes to implementing a Lax-Wendroff solver. Performant Finite-Difference, Time-Domain (FDTD) on a CPU is all about contiguous memory access. When it comes to brass tacks, you *have* to calculate all the values in the two equations described above, for every location on the grid, for every fluid variable. Loop through it, loop through them, and let the compiler optimize it all. When it comes to parallelizing this, too much communication is what will kill you.

Performant GPU code, on the other hand, is when we start getting into tradeoffs, and needing to do some **real** engineering. The fundamental tradeoff here is that of register pressure vs. cache-thrashing. Essentially, what we can do in each kernel is either:

(1) Read all the necessary data that we need from memory into local variables one time, i.e., at the very beginning of the kernel, and pass these values into the flux functions
OR
(2) Pass a handle to memory into the flux functions, and let them read it themselves

These options create a fundamental tradeoff. Either we will be storing a ton of local variables, and therefore introducing a high degree of "register pressure", or we will be "thrashing" memory as redundant access upon redundant access is performed by the flux functions. The problem with register pressure is that it reduces the number of threads that you can specify, and the problem with memory-thrashing is that it is a costly, and wasteful operation that kills performance.

## Data Volume
# RTX2060 - 6GB
The goal is to design the simulation according to a data volume of 5 GB. For the RTX2060 architecture which this application is targeted at, because it has been the platform upon which it was predominantly developed, upon this gives a nice headroom of ~1 GB to handle overhead tasks like display while the algorithm is running, as well as store small amounts of additional data, like the cartesian basis of the simulation mesh, which is `O(max(Nx,Ny,Nz))`, instead of `O(NxNyNz)`. 

This splits nicely down the middle into 2.5 GB for the fluid variables, and 2.5 GB for the intermediate variables. Each of these constitutes 8 different variables, and each variable requires 4 bytes (B) of memory. 1 GB = 1024^3 B which gives a side length of ~437 elements in the largest case.  

## Dimensionless variables
Numerical simulations of physical systems are best implemented in `non-dimensional` form. Essentially, this kind of way of writing a system of equations normalizes them in such a way that a set of dimensionless numbers, e.g., Mach, Euler, etc., which represent ratios between various forces, naturally fall out, and attach themselves to terms after some algebra.

Fortunately, the Ideal MHD system is already written in such a way, with the only physical constants left behind being the adiabatic index, which in general is related to the number of degrees of freedom that the particles of a gaseous system possess, and the vacuum permeability, which can easily be set to 1 since Ideal MHD presumes that the speed of light is infinite. Plasmas, being an electrically-charged gas which on macroscopic scales is roughly electrically-neutral as the abundance of free charge in the system acts in a way to shield electric potentials from the bulk, can be readily analyzed by considering the particles to have only translational degrees of freedom in three dimensions, which yields a value of 5 / 3 for the adiabatic index.    

### Misc
`git ls-files '*.cpp' '*.hpp' '*.cu' '*.cuh' '*.py' | xargs wc -l`
- Check # of relevant LoC