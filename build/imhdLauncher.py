import subprocess
from os import remove
from os import listdir
from os.path import isfile, join

# Delete all datafiles in data directory
data_root = "../data/"
all_data_dirs = [dir for dir in listdir(data_root) if not isfile(join(data_root, dir))]
print(all_data_dirs)
for data_dir in all_data_dirs:
    if data_dir == "grid": # leave grid data to `gridLauncher.py`
        continue
    data_path = data_root + data_dir + "/"
    all_data_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    print(all_data_files)
    for data_file in all_data_files:
        remove(data_path + data_file)

# Read `imhd-CUDA.inp`, and parse it.
arg_list = []
arg_list.append('./imhd-cuda')

with open('imhd-cuda.inp', 'r') as input_file:
    for line in input_file:
        cmd_arg = line.split('=')[1]
        arg_list.append(cmd_arg)

# Run `imhd_cuda` binary
subprocess.run(arg_list)

# Concat the datafiles from each timestep together
import pandas as pd

def ConcatFragments(data_files_location: str, write_var: str, number_of_timesteps: int) -> None: 
    all_data_files = [data_files_location + f for f in listdir(data_files_location) if isfile(join(data_files_location, f))] # grid fragments    
    it = 0
    while it < number_of_timesteps - 1: # initial timestep is it = 0
        df_list = []
        file_list = []

        for data_file in all_data_files:
            # print(data_file.split(write_var)[2].split('_')[0])
            if data_file.split(write_var)[2].split('_')[0] == str(it):
                df = pd.read_csv(data_file)
                df_list.append(df)
                file_list.append(data_file)

        print(f"Concating data files: {file_list}")
        var_df = pd.concat(df_list)

        print("Ascending sort by j, i, k")
        var_df = var_df.sort_values(by=['k', 'i', 'j'])
        var_df.to_csv(data_files_location + 'var_' + str(it) + '.csv', index=False)



        it += 1

    # Delete the fragments
    print(f"Deleting data files: {all_data_files}")
    for data_file in all_data_files: 
        remove(data_file)
    
    return

with open('imhd-cuda.inp', 'r') as input_file:
    for line in input_file:
        arg_name = line.split('=')[0]
        if (arg_name == 'Nt'):
            Nt = int(line.split('=')[1])
        try: 
            write_var = arg_name.split('write_')[1]
            # print(write_var)
            write_flag = int(line.split('=')[1])
            # print(write_flag)
            if (write_flag):
                data_files_location = "../data/" + write_var + "/"
                ConcatFragments(data_files_location, write_var, Nt) 
        except IndexError:
            continue 
