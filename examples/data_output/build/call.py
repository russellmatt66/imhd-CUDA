'''
Python driver for Proof-of-Concept .h5 file writing 
Run Instructions:
python call.py binary_name num_processes
'''
import subprocess
import sys
import os

path = '../data/'
print(os.access(path, os.W_OK))  # Should print True if writable

# Clean up data folders

# Call binaries
# Read `imhd-CUDA.inp`, and parse it.
binary_name = sys.argv[1]

if len(sys.argv) > 2:
    num_processes = sys.argv[2]

arg_list = []

# Run the h5write binary
# Wrap this in a function
if binary_name == "h5write_serial":
    arg_list.append('./' + binary_name)
    with open('input.inp', 'r') as input_file:
        for line in input_file:
            cmd_arg = line.split('=')[1].strip('\n')
            arg_list.append(cmd_arg)
    arg_list = arg_list[:(len(arg_list) - 2)] # serial version doesn't need same number of args as parallel
    print("Launching h5write_serial from Python")
    print(f"List of arguments is {arg_list}".format(arg_list))
    subprocess.run(arg_list)

elif binary_name == "h5write_par":
    arg_list.append('./' + binary_name)
    with open('input.inp', 'r') as input_file:
        for line in input_file:
            cmd_arg = line.split('=')[1].strip('\n')
            arg_list.append(cmd_arg)
    # mpirun_command = ["mpirun", "-np", num_processes, binary_name] + arg_list
    print("Launching h5write_par from Python")
    print(f"List of arguments is {arg_list}".format(arg_list))
    result = subprocess.run(arg_list)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    # print(f"Command is {mpirun_command}".format(mpirun_command))
    # try: 
        # result = subprocess.run(mpirun_command, check=True, stderr=subprocess.PIPE)
    # except subprocess.CalledProcessError as e: 
        # print("Error with running mpirun:")
        # print(e.stderr.decode())