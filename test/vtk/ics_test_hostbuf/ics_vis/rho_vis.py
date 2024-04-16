import pandas as pd
import vtk
import os
import sys

gridfile_location = "../data/grid/"
fluidvar_location = "../data/fluidvars/"

gridfile_list = [gridfile_location + f for f in os.listdir(gridfile_location) if os.path.isfile(os.path.join(gridfile_location, f))]
print(gridfile_list)

fluidvar_list = [fluidvar_location + f for f in os.listdir(fluidvar_location) if os.path.isfile(os.path.join(fluidvar_location, f))]
print(fluidvar_list)

grid_dfs = [pd.read_csv(f) for f in gridfile_list]
fluid_dfs = [pd.read_csv(f) for f in fluidvar_list]

grid_df = pd.concat(grid_dfs, ignore_index=True)
print(grid_df.shape)

fluid_df = pd.concat(fluid_dfs, ignore_index=True)
print(fluid_df.shape)
