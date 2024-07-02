# Overview
- fluid_vis_spheres.py
High-cost rendering of plasma as a point cloud of spheres using sphere_source 

Data Volume:
64 x 64 x 64 = 795 MiB
96 x 96 x 96 = 1194 MiB
128 x 128 x 128 = Killed

- fluid_vis_structgrid.py
Low-cost rendering of plasma and vacuum as a structured grid

Data Volume:
64 x 64 x 64 = 19 MiB
128 x 128 x 128 = 22 MiB
256 x 256 x 256 = 34 MiB
304 x 304 x 592 = 192 MiB

- fluid_vis_vertex.py
Low-cost rendering of fluid variable as a point cloud using VertexGlyphFilter

Data Volume:
64 x 64 x 64 = 24 MiB
128 x 128 x 128 = 197 MiB
256 x 256 x 256 = 339 MiB
304 x 304 x 592 = Killed