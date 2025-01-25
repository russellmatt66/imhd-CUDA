### Summary
The build folder.

### Build Instructions
## GPU Microarchitecture Caveat
This project has currently only been tested on an RTX2060 (Turing architecture - TU106, CC 7.5) running with the Ubuntu 22.04 operating system, and using CUDA v12.3. 

Cards with different compute capability, meaning non-7.5, will have trouble building the project as-is. I have not been able to test any cards besides the RTX2060, so I cannot state with exact certainty the modifications that need to be done to get it to work, but the first thing I'd try is modifying the `CMakeLists.txt` with your respective card's architecture.

For example, if you had a card based on the Ampere architecture, like the RTX 3060 (GA106, CC 8.6), and wanted to build, and run, the `on-device` solver, then you could try changing the lines in `src/on-device/CMakeLists.txt` which specify the `CUDA_ARCHITECTURES` property for the various libraries from `75` to `86`, like so,

```
set_target_properties(kernels_od_lib PROPERTIES CUDA_ARCHITECTURES 75 CUDA_SEPARABLE_COMPILATION ON) # CC 7.5
set_target_properties(kernels_od_lib PROPERTIES CUDA_ARCHITECTURES 86 CUDA_SEPARABLE_COMPILATION ON) # CC 8.6
```

Without having tested anything beyond my own workstation, I cannot confirm if this will work for any other particular combination of OS, card, and CUDA version.

## Dependencies
Building the project requires:

- CUDA
- CMake
- HDF5 (Parallel)
- MPI

Again, it has only been tested to work on the following configuration:

- GPU: `RTX2060 (TU106, CC 7.5)`
- OS: `Ubuntu 22.04`
- CUDA: `12.3`
- CMake: `3.29.3` (`3.12` minimum)
- HDF5: `1.14.3`
- MPI: `5.0.3` 

## Obtaining Dependencies
**WIP** 

Each of the dependencies listed above has instructions on their websites for how to obtain the software. 

Currently, data is written out with Parallel HDF5 (PHDF5), and means for doing a serial write has not yet been implemented. 

That means at present you will need PHDF5 installed in order to build, and run the software. 

## How to Build
After you have all the dependencies installed

1. Navigate to the `build/` folder (which you are currently in if you are reading this). 
2. Run `cmake ../src -DARCH_NAME=ON`, with `ARCH_NAME` corresponding to the solver architecture that you want to build. You can build more than one. 
- `ARCH_NAME`:
    - `ARCH_ONDEV` = The `on-device` solver architecture
    - `ARCH_HOSTDEV` = The `host-device` solver architecture
    - `ARCH_CPUSER` = The serial host (CPU) solver
    - `ARCH_CPUPAR` = The parallel host (CPU) solver
3. Run `make` 

Currently, only `ARCH_ONDEV` is end-to-end implemented. `make` will cause a set of binaries to be built in the appropriate directories. 

### Run Instructions
`python simulation_launcher.py -`
- Since only the `on-device` architecture is currently implemented, this does not need any arguments to be accepted, however following the principle of YAGNI (You Are Gonna Need It) I have added a `mode` argument. It currently does nothing so I just input the `-` character as a placeholder.

The above command reads the file `input.inp` in the specified build folder (which is currently `on-device/`), and launches the simulation accordingly. 

See the **input file specification** for the appropriate solver to understand the list of all inputs and their meaning:
- [ARCH_ONDEV Input File](./on-device/README.md#input-file) 

### Directory Structure
`simulation_launcher.py`
- Python code that launches the simulation

`on-device/`
- On-device build files go in here

`host-device/`
- Host-device build files go in here

`serial/`
- Serial host build files go in here

`par/`
- Parallel host build files go in here

# Useful Commands
` make 2>&1 | tee make.log`
- Pipe stderr and stdout produced by `make` into `make.log`
- Point is to be able to study register pressure of library kernels 