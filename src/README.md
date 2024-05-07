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
(1) Complete `write_grid.cu`

(2) Write the data out with GDS
- Implement feature to write out data every given number of timesteps or so