# Overview
Project to implement the Lax-Wendroff scheme in order to solve the Ideal MHD system.

# Design
## On-Device or Host-Device?
Limited VRAM on GPU (RTX 2060) means storage is at a premium, and thread asynchronity means that the computation cannot just launched with the algorithm naively implemented. 

That will not work, because there is no guarantee for when a given fluid variable will be updated. Meaning, as a consequence of thread asynchronity, eventually variables whose threads have updated them to the future timestep will be used when computing the flux function in order to update another variable - who should only be updated using data that represents fluid variables at the current timestep.

The solution in this arena of limited VRAM, is to store the fluid variables for the current timestep on the device, and then load in the data from host representing variables at a future timestep. Load these in one by one.

This `host-device` approach requires a significant amount of memory migration, and synchronization, but with the additional 12 GB of host RAM, it also allows 
for problems as large as `N = 512` side elements to be simulated. The RTX 2060 has a bandwidth of 336 GB/s, and a performance of ~6.5 TFlops, meaning the speed
of the computation will be limited by the significantly-slower memory bandwidth. Orchestrating the memory migrations also requires a significant amount of 
synchronization

Why not pick a problem size that can be handled without all of this memory migration, i.e., an on-device approach?
Data volume = 4 * (8 + 24 + 8 + 24 + 8)* N^3 (Cur. fluid vars + fluxes + int. vars + int. fluxes + Fut. fluid vars b/c of thread asychronicity problem)
\therefore N <= ((6 * 1024^3) / 288)^(1/3) ~= 288, this is the largest side of a data cube that could be simulated for a completely on-device approach.

The on-device approach would store all the required data in the GPUs VRAM, and will consequently be much faster as no migrations will be required during the 
bulk of the computation, and the amount of synchronization required will be decimated as well. The tradeoff is not being able to simulate as large of a domain. 

## 

# Directory Structure
