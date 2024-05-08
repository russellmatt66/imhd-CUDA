# Overview
Source directory

# Directory Structure
`main.cu`
- Main executable for the simulation
- Runs it

`write_grid.cu`
- Writes the grid out so that the data can be consumed by post-processing
- Don't have the space to do this during the simulation

# Current Tasks
(1) Figure out why all Jz for it > 0 are 0
(2) Add GDS to `write_grid.cu`
- Resolve max pinned size GDS error
```
~/Desktop/imhd-CUDA/build$ python3 gridLauncher.py 
cuFile Buffer registration error: access beyond maximum pinned size
```
