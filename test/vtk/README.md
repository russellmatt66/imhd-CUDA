# Directory Structure
animate_test/

ics_test_hostbuf/
- Visualizes initial conditions and grid using data written to storage via the CPU (bounce buffer)

ics_test_gds/
- Visualizes initial conditions and grid using data written out with GPU Direct Storage (GDS)

# Current Tasks
(1) Complete `ics_test_gds`
- Speed up visualization by eliminating host code to read in the data 
- Figure out max pinned memory problem
- Visualize `rho_ics.dat`

(2) Complete `ics_test_hostbuf`
-  
