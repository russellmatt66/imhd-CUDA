import subprocess
import sys 

from os import remove
from os import listdir
from os.path import isfile, join

# Delete all datafiles in data directory
data_root = "../data/on-device/"
all_data_files = [file for file in listdir(data_root) if isfile(join(data_root, file))]
print(all_data_files)

for file in all_data_files:
    if file == "README.md":
        continue
    else:
        remove(data_root + file)

# Read input file, and parse it.
arg_list = []

mode = sys.argv[1]

driver = ''
input_file = './on-device/input.inp'

if mode == "nodiff-mega": # launch w/out diffusion, using megakernels for qint bdry
    driver = './on-device/imhd-cuda_nodiff_qintmega'
    # input_file = './on-device/input_nodiff.inp'
elif mode == "nodiff-micro": # launch w/out diffusion, using microkernels for qint bdry
    driver = './on-device/imhd-cuda_nodiff_qintmicro'
    # input_file = './on-device/input_nodiff.inp'
else:
    driver = './on-device/imhd-cuda'
    # input_file = ''

arg_list.append(driver)

with open(input_file, 'r') as input_file:
    for line in input_file:
        cmd_arg = line.split('=')[1].strip('\n')
        arg_list.append(cmd_arg)

# Run `imhd_cuda` binary
subprocess.run(arg_list)