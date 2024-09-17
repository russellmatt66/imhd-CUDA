import pandas as pd
import cudf
import os
import pynvml
import time

'''
CAN PROBABLY COMPRESS THIS AND OTHER `condense` SCRIPTS into one with RMM

Use with 304 x 304 x 592 (max simulation size on RTX 2060 6 GB)
Combine data files into a single one

GPU doesn't have enough memory to handle this all at once
Need to use RMM to manage memory explicitly: https://github.com/rapidsai/rmm
'''
start = time.time()
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Need to specify GPU

gridfile_location = "./data/grid_data.csv"
fluidvar_location = "./data/fluid_data.csv"

grid_gdf = cudf.read_csv(gridfile_location)

mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
print("Memory stats after loading grid: free, used, total\n")
print(mem.free / 1024**3, mem.used / 1024**3, mem.total / 1024**3)

grid_df = grid_gdf.to_pandas()

mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
print("Memory stats after writing grid to host, and before loading fluid to device: free, used, total\n")
print(mem.free / 1024**3, mem.used / 1024**3, mem.total / 1024**3)

fluid_gdf = cudf.read_csv(fluidvar_location)

mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
print("Memory stats after loading fluid: free, used, total\n")
print(mem.free / 1024**3, mem.used / 1024**3, mem.total / 1024**3)

fluid_df = fluid_gdf.to_pandas()

mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
print("Memory stats after writing fluid to host, and before merging on host: free, used, total\n")
print(mem.free / 1024**3, mem.used / 1024**3, mem.total / 1024**3)

merged_df = pd.merge(grid_df, fluid_df, on=['i', 'j', 'k'])

merged_df.to_csv('./data/sim_data.csv', index=False)

end = time.time()

print("Took " + str(end-start) + " seconds\n")