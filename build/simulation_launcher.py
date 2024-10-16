import subprocess
from os import remove
from os import listdir
from os.path import isfile, join

# Delete all datafiles in data directory
data_root = "../data/"
all_data_files = [file for file in listdir(data_root) if isfile(join(data_root, file))]
print(all_data_files)

for file in all_data_files:
    if file == "README.md":
        continue
    else:
        remove(data_root + file)

# Read input file, and parse it.
arg_list = []
arg_list.append('./imhd-cuda')

with open('input.inp', 'r') as input_file:
    for line in input_file:
        cmd_arg = line.split('=')[1].strip('\n')
        arg_list.append(cmd_arg)

# Run `imhd_cuda` binary
subprocess.run(arg_list)