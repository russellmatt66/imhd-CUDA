# Directory Structure
animate_test/

ics_test_hostbuf/
- Visualizes initial conditions and grid using data written to storage via the CPU (bounce buffer)

ics_test_gds/
- Visualizes initial conditions and grid using data written out with GPU Direct Storage (GDS)

# Current Tasks
(1) Complete `ics_test_hostbuf`
(2) Complete `ics_test_gds`
- DEBUG: Why are grid values reading as being 0?
-- Culprit is almost certainly in `InitializeGrid` device kernel

