'''
Python driver for Proof-of-Concept Flux Jacobian eigenvalue calculation
Run Instructions:
python call.py 
'''
import subprocess
import sys
import os

# Call binary
# Read `input.inp`, and parse it.
binary_name = sys.argv[1]

# if len(sys.argv) > 2:
#     num_processes = sys.argv[2]

arg_list = []
arg_list.append('./' + binary_name)

with open('input.inp', 'r') as input_file:
    for line in input_file:
        cmd_arg = line.split('=')[1].strip('\n')
        arg_list.append(cmd_arg)

print(arg_list)

# Run `fj_evs_cu` binary
subprocess.run(arg_list)