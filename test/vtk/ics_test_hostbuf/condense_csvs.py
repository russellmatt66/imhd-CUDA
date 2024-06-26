# import pandas as pd
import cudf
import os
import pynvml
import time

'''
Combine data files into a single one
DOESN'T WORK FOR 304 x 304 x 592 (on RTX 2060 6 GB)

Works for up to 256 x 256 x 256 (on RTX 2060 6 GB)
'''
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Need to specify GPU

gridfile_location = "./data/grid/"
fluidvar_location = "./data/fluidvars/"

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

grid_dfs = [cudf.read_csv(f) for f in gridfile_list_sorted]
fluid_dfs = [cudf.read_csv(f) for f in fluidvar_list_sorted]

grid_df = cudf.concat(grid_dfs, ignore_index=True)
print(grid_df.shape)
print(grid_df.head)

fluid_df = cudf.concat(fluid_dfs, ignore_index=True)
print(fluid_df.shape)
print(fluid_df.head)

mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

print("Memory stats after concating fluid and grid: free, used, total\n")
print(mem.free / 1024**3, mem.used / 1024**3, mem.total / 1024**3)

merged_df = cudf.merge(grid_df, fluid_df, on=['i', 'j', 'k'])

merged_usage = merged_df.memory_usage(deep=True)
print(merged_usage / 1024 / 1024 / 1024)
merged_df = merged_df.sort_values(by=['j', 'i', 'k'])

print("Memory stats before writing merged_df to storage\n")
print(mem.free / 1024**3, mem.used / 1024**3, mem.total / 1024**3)

h_df = merged_df.to_pandas() # RTX 2060 6 GB doesn't have enough memory 

h_df.to_csv('./data/sim_data.csv', index=False)
