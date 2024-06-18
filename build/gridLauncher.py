import subprocess
from os import remove
from os import listdir
from os.path import isfile, join

# Delete all datafiles in the grid data directory
data_files_location = "../data/grid/"

all_data_files = [f for f in listdir(data_files_location) if isfile(join(data_files_location, f))]

print(f"Deleting grid data files: {all_data_files}")
for data_file in all_data_files:
    remove(data_files_location + data_file)

# Read `grid.inp`, and parse it.
arg_list = []
arg_list.append('./write-grid')

with open('grid.inp', 'r') as input_file:
    for line in input_file:
        cmd_arg = line.split('=')[1]
        arg_list.append(cmd_arg)

# Run `write-grid` binary
print("Running `write-grid`")
subprocess.run(arg_list)

# Concatenate the grid files together
import pandas as pd

all_data_files = [data_files_location + f for f in listdir(data_files_location) if isfile(join(data_files_location, f))] # grid fragments

df_list = []

for data_file in all_data_files:
    df = pd.read_csv(data_file)
    df_list.append(df)

print(f"Concating data files: {all_data_files}")
grid_df = pd.concat(df_list)

print("Ascending sort by j, i, k")
grid_df = grid_df.sort_values(by=['k', 'i', 'j'])
grid_df.to_csv(data_files_location + 'grid.csv', index=False)

print(f"Deleting data files: {all_data_files}")
for data_file in all_data_files: # Delete the fragments
    remove(data_file)