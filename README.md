# Overview
Project to implement the Lax-Wendroff scheme in order to solve the Ideal MHD system.

# Current Tasks
(1) Run kernel and visualize results

(2) Figure out how to change the maximum amount of device pinned memory
- `test/gds/max_pinned` 


# Design
## On-Device or Host-Device?
Limited VRAM on GPU (RTX 2060) means storage is at a premium, and thread asynchronicity means that it will not work to launch a binary with the algorithm naively implemented to work directly on the variables in memory, because there is no guarantee for when a given fluid variable will be updated. Meaning, as a consequence of thread asynchronicity, this approach will lead to a race condition where variables whose threads have updated them to the future timestep will be used in the update for another variable - who should only be updated using data that represents fluid variables at the current timestep.

One solution in this arena of limited VRAM, is to store the fluid variables for the current timestep on the device, and then load in the data from host representing variables at a future timestep. This `host-device` approach of loading the fluid variables in one-by-one, updating them, and then migrating the data out to cycle in the next fluid variable, updating it, so on and so forth, etc., will enable simulations that are larger than a purely `on-device` approach where all the necessary data to update fluid variables is stored directly in device memory.

One disadvanage of the `host-device` approach is that it requires a significant amount of memory migration, and synchronization, but with the addition of host RAM, it also allows for much larger problems to be simulated as the amount of data that must be stored on device is more than halved. The RTX 2060 has a bandwidth of 336 GB/s, and a performance of ~6.5 TFlops, meaning the speed of a `host-device` computation will be limited by the significantly-slower memory bandwidth. Orchestrating the memory migration also requires a significant amount of synchronization barriers which introduces further points of slow-down. 

Why not pick a problem size that can be handled without all of this memory migration, i.e., an `on-device` approach? An `on-device` approach would store all the required data in the GPUs VRAM, and will consequently be much faster as no migrations will be required during the bulk of the computation, and the amount of synchronization required will be decimated as well. The tradeoff is not being able to simulate as large of a domain. Also, exactly *what* data is being stored on the device should be specified as well. See `./test/problem_size/` for data volume calculations based on these specifics.

## Data Volume
The goal is to design the simulation according to a data volume of 5 GB. For the RTX2060 architecture which this application is targeted at, this gives a nice headroom of ~1 GB to handle tasks like display while the algorithm is running, as well as store small amounts of additional data, like the cartesian basis of the simulation mesh, which is `O(max(Nx,Ny,Nz))`, instead of `O(NxNyNz)`. 

## Dimensionless variables
Numerical simulations of physical systems are best implemented in `non-dimensional` form. Essentially, this kind of way of writing a system of equations normalizes them in such a way that a set of numbers, e.g., Mach, Euler, Frobenius, etc., representing physical constants, are left behind.

Fortunately, the Ideal MHD system is already written in such a way, with the only physical constant left behind being the adiabatic index, which is related to the number of degrees of freedom that the plasma particles possess. 

## Initial and Boundary Conditions
WIP

# Directory Structure
WIP