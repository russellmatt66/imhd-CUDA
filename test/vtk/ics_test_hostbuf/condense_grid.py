# import pandas as pd
import cudf
import os
import pynvml
import time

'''
CAN PROBABLY COMPRESS THIS AND `condense_fluid.py` AND `condense_maxdata.py` INTO ONE FILE WITH RMM

Use with 304 x 304 x 592
Combine grid datafiles into a single one
'''
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Need to specify GPU

gridfile_location = "./data/grid/"

def extract_number(filename): # need to sort the files according to number at the end
    data_num_string = filename.split('_')[-1].split('.')[0] # every file looks like *_dataN.csv
    return int(data_num_string[4]) 

gridfile_list = [gridfile_location + f for f in os.listdir(gridfile_location) if os.path.isfile(os.path.join(gridfile_location, f))]
# print(gridfile_list)
gridfile_list_sorted = sorted(gridfile_list, key=extract_number)
print(gridfile_list_sorted)

grid_dfs = [cudf.read_csv(f) for f in gridfile_list_sorted]

grid_df = cudf.concat(grid_dfs, ignore_index=True)
print(grid_df.shape)
print(grid_df.head)

mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
print("Memory stats after concating grid data together: free, used, total")
print(mem.free / 1024**3, mem.used / 1024**3, mem.total / 1024**3)

h_df = grid_df.to_pandas()

mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
print("Memory stats after sending grid data to host: free, used, total")
print(mem.free / 1024**3, mem.used / 1024**3, mem.total / 1024**3)

h_df.to_csv('./data/grid_data.csv')