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

