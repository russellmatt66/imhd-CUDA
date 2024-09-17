import pandas as pd
import vtk
import os
import sys

'''
Low-cost rendering of plasma and vacuum as a structured grid

Data Volume:
64 x 64 x 64 = 19 MiB
128 x 128 x 128 = 22 MiB
256 x 256 x 256 = 34 MiB
304 x 304 x 592 = 192 MiB

Next steps:
(1) Enable rendering of arbitrary plasma variable
'''

gridfile_location = "../data/grid/"
fluidvar_location = "../data/fluidvars/"

def extract_number(filename): # need to sort the files according to number at the end
    data_num_string = filename.split('_')[-1].split('.')[0] # every file looks like *_dataN.csv
    return int(data_num_string[4]) 

gridfile_list = [gridfile_location + f for f in os.listdir(gridfile_location) if os.path.isfile(os.path.join(gridfile_location, f))]
# print(gridfile_list)
gridfile_list_sorted = sorted(gridfile_list, key=extract_number)
print(gridfile_list_sorted)

fluidvar_list = [fluidvar_location + f for f in os.listdir(fluidvar_location) if os.path.isfile(os.path.join(fluidvar_location, f))]
# print(fluidvar_list)
fluidvar_list_sorted = sorted(fluidvar_list, key=extract_number)
print(fluidvar_list_sorted)

grid_dfs = [pd.read_csv(f) for f in gridfile_list_sorted]
fluid_dfs = [pd.read_csv(f) for f in fluidvar_list_sorted]

grid_df = pd.concat(grid_dfs, ignore_index=True)
print(grid_df.shape)
print(grid_df.head)

fluid_df = pd.concat(fluid_dfs, ignore_index=True)
print(fluid_df.shape)
print(fluid_df.head)

x = grid_df['x']
y = grid_df['y']
z = grid_df['z']
rho = fluid_df['rho']

# Store the raw data in VTK data objects 
points = vtk.vtkPoints()
vtk_float_array = vtk.vtkFloatArray()
vtk_float_array.SetNumberOfComponents(1)
grid = vtk.vtkStructuredGrid()
grid.SetDimensions(len(set(x)), len(set(y)), len(set(z))) 

for l in range(len(rho)):
    points.InsertNextPoint(x[l], y[l], z[l])
    vtk_float_array.InsertNextValue(rho[l])

grid.SetPoints(points)
grid.GetPointData().SetScalars(vtk_float_array)

# Create a lookup table (LUT)
lut = vtk.vtkLookupTable()
lut.SetHueRange(0.0, 0.7)  # Set the range of colors (blue to red)
lut.SetTableRange(min(rho), max(rho))
print(lut.GetTableRange())
lut.Build()

# Create mapper
mapper = vtk.vtkDataSetMapper()
mapper.SetInputData(grid)

# Create actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)


# Create renderer
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(1.0, 1.0, 1.0)  # white background

# Create render window and interactor
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

# Adjust camera position
renderer.ResetCamera()

# Add a color bar
scalar_bar = vtk.vtkScalarBarActor()
scalar_bar.SetLookupTable(mapper.GetLookupTable())
scalar_bar.SetTitle("Rho")
scalar_bar.SetNumberOfLabels(5)  # Adjust as needed
renderer.AddActor2D(scalar_bar)

# Render the Data
render_window.Render()

# Display or Save the Visualization
render_window_interactor.Start()