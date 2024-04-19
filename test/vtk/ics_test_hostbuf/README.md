# Summary 
Write Initial Conditions out to `.csv`'s using host bounce-buffer, and then visualize with VTK

# Current Tasks
(1) Work on VTK scripts 
- Python: `ics_vis/rho_vis.py`
-- Point Cloud
--- Spheres
--- Vertices
-- Structured Grid

- C++
-- Ape Python

# Directory Structure
condense_csvs.py
- Must run after `./ics_cu Nx Ny Nz`
- Combines data files from above into one 