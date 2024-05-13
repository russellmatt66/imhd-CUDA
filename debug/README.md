# Overview
This is where the debugging happens

# Directory Structure
src/
- `main.cu` in here is stripped down version of code with no benchmarking, or data writing functionality implemented

build/
- Where `main.cu` is built

grid-gds/
- For debugging why the grid values are read from the GDS data file as being all 0
    -- Culprit was in `InitializeGrid` device kernel
    -- Problem was related to specifying too many threads per block in the execution configuration 
- Didn't actually need this folder to find bug, so I deleted it, but kept this for posterity

# Current Tasks
