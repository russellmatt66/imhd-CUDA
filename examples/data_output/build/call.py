'''
Python driver for Proof-of-Concept .h5 file writing 
'''
import subprocess
# Clean up data folders

# Call binaries
# Read `imhd-CUDA.inp`, and parse it.
arg_list = []
arg_list.append('./h5write_serial')

with open('input.inp', 'r') as input_file:
    for line in input_file:
        cmd_arg = line.split('=')[1].strip('\n')
        arg_list.append(cmd_arg)

# Run `imhd_cuda` binary
print("Launching h5write_serial from Python")
print(f"List of arguments is {arg_list}".format(arg_list))
subprocess.run(arg_list)