# Launch the `make_movie` binary for rendering contents of data folder as a movie
import os
import sys
import subprocess

path_to_data = sys.argv[1]
dset_name = sys.argv[2]

arg_list = []
arg_list.append("./makemovie_driver")
arg_list.append(path_to_data)
arg_list.append(dset_name)
arg_list.append(path_to_data + "grid.h5")

# Read data folder to get a measure of how many frames are in there
def numberOfFrames(path_to_data: str) -> int:
    Nt = 0
    all_data_files = [file for file in os.listdir(path_to_data) if os.path.isfile(os.path.join(path_to_data, file))]

    for file in all_data_files:
        if file == "README.md": 
            continue
        file_name = file.split('.h5')[0]
        if file_name == "grid":
            continue
        frame_num = file_name.split('_')[1]
        if int(frame_num) > Nt: 
            Nt = int(frame_num)
         
    return Nt

Nt = numberOfFrames(path_to_data)

arg_list.append(str(Nt))

# Specify rendering parameters
# Append them to arg list
input_file = './makemovie.inp'
with open(input_file, 'r') as input_file:
    for line in input_file:
        cmd_arg = line.split('=')[1].strip('\n')
        arg_list.append(cmd_arg)

def getdt(sim_input_file: str) -> str:
    dt = ''
    with open(sim_input_file, 'r') as input_file:
        for line in input_file:
            input_param = line.split('=')[0]
            if input_param == 'dt':
                dt = line.split('=')[1].strip('\n')
    return dt

def getD(sim_input_file: str) -> str:
    D = ''
    with open(sim_input_file, 'r') as input_file:
        for line in input_file:
            input_param = line.split('=')[0]
            if input_param == 'D':
                D = line.split('=')[1].strip('\n')
    return D

'''
WILL NEED TO REFACTOR THIS IN THE FUTURE SO THAT IT WORKS FOR MULTIPLE ARCHITECTURES
EASY SOLUTION IS JUST TO ACCEPT AS AN INPUT, OR YOU CAN SPECIFY DIRECTLY SINCE IT'S PYTHON
'''
# Append dt and D to arg list 
sim_input_file = '../../build/on-device/input.inp' # run from inside ${PROJECT_ROOT}/visualization/build

dt = getdt(sim_input_file)
arg_list.append(dt)

D = getD(sim_input_file)
arg_list.append(D)

print(f"Running command: {' '.join(arg for arg in arg_list)}")
subprocess.run(arg_list)