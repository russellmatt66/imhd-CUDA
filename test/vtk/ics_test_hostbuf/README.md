# Summary 
Write Initial Conditions out to `.csv`'s using host bounce-buffer, and then visualize point cloud data with VTK

# Current Tasks


# Directory Structure 
ics.cu
- WIP 

condense_csvs.py
- Must run after `./ics_cu Nx Ny Nz`
- Combines data files from above into one
- Tested for up to 256 x 256 x 256 (RTX 2060 6 GB)

condense_fluid.py
- WIP

condense_grid.py
- WIP

condense_maxdata.py
- WIP

condense_maxdata_cpu.py
- WIP