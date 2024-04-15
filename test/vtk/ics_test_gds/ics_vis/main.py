'''
Doesn't work
'''
import vtk
import numpy as np
import sys

Nx = int(sys.argv[1])
Ny = int(sys.argv[2])
Nz = int(sys.argv[3])

with open("../rho_ics.dat", "rb") as f:
    raw_data = f.read()

rho_float_values = np.frombuffer(raw_data, dtype=np.float32)

grid_dimensions = (Nx, Ny, Nz)

grid = vtk.vtkStructuredGrid()
grid.SetDimensions(grid_dimensions)

vtk_float_array = vtk.vtkFloatArray()
vtk_float_array.SetNumberOfComponents(1)
vtk_float_array.SetArray(rho_float_values, len(rho_float_values), 1)
grid.GetPointData().SetScalars(vtk_float_array)

mapper = vtk.vtkDataSetMapper()
mapper.SetInputData(grid)

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

renderer.ResetCamera()
renderer.SetBackground(1.0, 1.0, 1.0)

render_window.Render()
render_window_interactor.Start()

