# Overview
This is the source folder for the benchmarking and profiling part of the project

# Directory Structure
`manual.cu`
- This is the source code for the manual benchmarking of the various kernels
- Currently only benchmarks the `on-device` megakernels for the Screw-pinch equilibrium

`profile_full.cu`
- This is the source code for the application which is used with NSight tools to profile the simulation fully. 
- Currently only benchmarks the `on-device` megakernels for the Screw-pinch equilibrium 