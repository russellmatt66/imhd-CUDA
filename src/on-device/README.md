# Summary
This is where the source code for the on-device runtime goes.

# Directory Structure
`main.cu` (WIP)
- Standard runtime for the simulation
- Includes diffusion (UNTESTED)
- Highly recommended not to use right now, honestly probably won't work b/c the argument list is out of phase with the launcher

`no_diffusion.cu` 
- Runtime for the simulation that uses megakernels for Fluid/Qint advance, and microkernels for Fluid/Qint BCs.
- This is the optimal form for the runtime. See below for explanation. 
- Recommended to use this runtime.

`no_diffusion_mega`
- Runtime for the simulation that uses megakernels for everything.
- Highly recommended not to use - ever: BC megakernels are ~100x slower than microkernels.
- Here for comparison (making social media posts).

`no_diffusion_micro` (WIP)
- Runtime for the simulation that uses microkernels for everything.
- Resource limits (register pressure) means that the execution config. is best explicitly specified, and additional synchronization is required.
- Microkernels for Fluid/Qint advance are slower in aggregate than the megakernels due to software overhead.

`no_diffusion_micro_ec`
- Runtime for the simulation that uses microkernels for everything
- Execution configuration can be freely specified from input file
- Highly recommended not to use unless you like `cudaErrorInvalidConfiguration`
- Resource limits still apply.
- Here for comparison (making social media posts).

`utils/`
- Source code for the various utilities