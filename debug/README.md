# Overview
This is where the debugging happens

# Directory Structure
grid-gds/
- For debugging why the grid values are read from the GDS data file as being all 0
    -- Culprit was in `InitializeGrid` device kernel
    -- Problem was related to specifying too many threads per block in the execution configuration 

# Current Tasks
