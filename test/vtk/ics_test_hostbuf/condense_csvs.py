# import pandas as pd
import cudf
import os

'''
Implement this with cuDF
'''

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

# for partial_datafile in gridfile_list_sorted:
#     os.remove(partial_datafile)

# for partial_datafile in fluidvar_list_sorted:
#     os.remove(partial_datafile)

# grid_df.to_csv('./data/grid_data.csv')
# fluid_df.to_csv('./data/fluid_data.csv')
merged_df = cudf.merge(grid_df, fluid_df, on=['i', 'j', 'k'])
sorted_df = merged_df.sort_values(by=['j', 'i', 'k'])
merged_df.to_csv('./data/sim_data.csv', index=False)