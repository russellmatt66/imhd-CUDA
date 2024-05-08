# Overview
Project to implement the Lax-Wendroff scheme in order to solve the Ideal MHD system.

# Current Tasks
(1) Run kernel and visualize results
- Write data out
-- Running into problems with GDS (max pinned size)

(2) Figure out how to change the maximum amount of device pinned memory
- `test/gds/max_pinned` 
-- Not really sure why it's not changing, guess project will have to go with device2host transfers

(3) Benchmark
- Kernel
- BCs
- Swap buffer

# Design
## On-Device or Host-Device?
Limited DRAM on GPU (RTX 2060) means storage is at a premium, and thread asynchronicity means that it will not work to launch a binary with the algorithm naively implemented to work directly on the variables in memory, because there is no guarantee for when a given fluid variable will be updated. Meaning, as a consequence of thread asynchronicity, this approach will lead to a race condition where variables whose threads have updated them to the future timestep will be used in the update for another variable - who should only be updated using data that represents fluid variables at the current timestep. 

One solution in this arena of limited VRAM, is to store the fluid variables for the current timestep on the device, and then load in the data from host representing variables at a future timestep. This `host-device` approach of loading the fluid variables in one-by-one, updating them, and then migrating the data out to cycle in the next fluid variable, updating it, so on and so forth, etc., will enable simulations that are larger than a purely `on-device` approach where all the necessary data to update fluid variables is stored directly in device memory.

One disadvantage of the `host-device` approach is that it requires a significant amount of memory migration, and synchronization, but with the addition of host RAM, it also allows for much larger problems to be simulated as the amount of data that must be stored on device is more than halved. The RTX 2060 has a bandwidth of 336 GB/s, and a performance of ~6.5 TFlops, meaning the speed of a `host-device` computation will be limited by the significantly-slower memory bandwidth. Orchestrating the memory migration also requires a significant amount of synchronization barriers which introduces further points of slow-down. In this regime, the GPU can be thought of as a banquet table of ravenous diners, and the host as the kitchen and wait-staff who are unable to provide them with food as fast as they'd like.   

Why not pick a problem size that can be handled without all of this memory migration, i.e., an `on-device` approach? An `on-device` approach would store all the required data in the GPUs DRAM, and will consequently be much faster as no migrations will be required during the bulk of the computation, and the amount of synchronization required will be decimated as well. The tradeoff is not being able to simulate as large of a domain. Also, because the data representing the future timestep is not being migrated to the device from the host every timestep, and instead is located on the device for the duration of the simulation, a swap buffer must be implemented that moves this data into the array representing the variable values at the current timestep.   

Finally, the Lax-Wendroff method requires that a set of "intermediate" variables be calculated every time when the state variables are being updated. Declaring these in the form of local variables would be a foolish approach due to how many there are, so the entire set is instead stored in device memory.   

## Data Volume
The goal is to design the simulation according to a data volume of 5 GB. For the RTX2060 architecture which this application is targeted at, this gives a nice headroom of ~1 GB to handle overhead tasks like display while the algorithm is running, as well as store small amounts of additional data, like the cartesian basis of the simulation mesh, which is `O(max(Nx,Ny,Nz))`, instead of `O(NxNyNz)`. 

Calculations regarding the problem size which can be simulated can be found in `./test/problem_size/`.

## Dimensionless variables
Numerical simulations of physical systems are best implemented in `non-dimensional` form. Essentially, this kind of way of writing a system of equations normalizes them in such a way that a set of numbers, e.g., Mach, Euler, Frobenius, etc., representing physical constants, are left behind.

Fortunately, the Ideal MHD system is already written in such a way, with the only physical constant left behind being the adiabatic index, &gamma, which in general is related to the number of degrees of freedom that the particles of a gaseous system possess. Plasmas, being an electrically-charged gas which on macroscopic scales is roughly electrically-neutral as the abundance of free charge in the system acts in a way to shield electric potentials from the bulk, can be analyzed by considering the particles to have only translational degrees of freedom in three dimensions, which yields a &gamma = 5 / 3.    

## Initial and Boundary Conditions
The system that is currently implemented for simulation is a screw-pinch fusion plasma. The screw-pinch is a kind of magnetic confinement fusion (MCF) configuration which uses external magnets along with the `pinch` effect in order to heat, and confine, a plasma to fusion temperatures for long enough, and at high enough density, that breakeven is achieved. Currently, no screw-pinch, or MCF configuration for that matter, has achieved breakeven, a.k.a ignition. 

# Directory Structure
WIP