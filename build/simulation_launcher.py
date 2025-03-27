import subprocess
import sys 

from os import remove
from os import listdir
from os.path import isfile, join

'''
This file launches the simulation once the specified architecture has been built
Dev Notes:
(1) Currently hard-coded to only run the `on-device` binaries 
'''

architecture = sys.argv[1]

# Delete all datafiles in data directory
data_root = "../data/on-device/" # 
all_data_files = [file for file in listdir(data_root) if isfile(join(data_root, file))]
print(all_data_files)

for file in all_data_files:
    if file == "README.md":
        continue
    else:
        remove(data_root + file)

# Read input file, and parse it.
arg_list = []

mode = sys.argv[2]

runtime = ''
input_file = './on-device/input.inp'

if mode == 'nodiff':
    runtime = './on-device/imhd-cuda_nodiff'
else:
    runtime = './on-device/imhd-cuda'

arg_list.append(runtime)

with open(input_file, 'r') as input_file:
    for line in input_file:
        cmd_arg = line.split('=')[1].strip('\n')
        arg_list.append(cmd_arg)

# Run `imhd_cuda` binary
subprocess.run(arg_list)