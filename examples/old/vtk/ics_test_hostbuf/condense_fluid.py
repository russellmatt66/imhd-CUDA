# import pandas as pd
import cudf
import cupy
import os
import pynvml
import time

'''
CAN PROBABLY COMPRESS THIS AND `condense_grid.py` AND `condense_maxdata.py` INTO ONE FILE WITH RMM

Use with 304 x 304 x 592
Combine fluid datafiles into a single one
'''
start = time.time()

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Need to specify GPU

fluidvar_location = "./data/fluidvars/"

def extract_number(filename): # need to sort the files according to number at the end
    data_num_string = filename.split('_')[-1].split('.')[0] # every file looks like *_dataN.csv
    return int(data_num_string[4]) 

fluidvar_list = [fluidvar_location + f for f in os.listdir(fluidvar_location) if os.path.isfile(os.path.join(fluidvar_location, f))]
# print(fluidvar_list)
fluidvar_list_sorted = sorted(fluidvar_list, key=extract_number)
print(fluidvar_list_sorted)

fluid_gdfs = [cudf.read_csv(f) for f in fluidvar_list_sorted]

fluid_gdf = cudf.concat(fluid_gdfs, ignore_index=True)
print(fluid_gdf.shape)
print(fluid_gdf.head)

mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
print("Memory stats after concating fluid: free, used, total")
print(mem.free / 1024**3, mem.used / 1024**3, mem.total / 1024**3)

h_df = fluid_gdf.to_pandas()

mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
print("Memory stats after sending fluid data to host: free, used, total")
print(mem.free / 1024**3, mem.used / 1024**3, mem.total / 1024**3)

h_df.to_csv('./data/fluid_data.csv', index=False)

end = time.time()
print("Total execution time = " + str(end - start) + " seconds")