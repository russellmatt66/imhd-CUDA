'''
Call `ics_vis` binary to open .h5 file representing output from a single simulation timestep, and visualize the data
'''
import subprocess
import sys
import os

# Parse input file to get file_name, and data file to get attribute
bin_name = sys.argv[1]

arg_list = ['./' + bin_name]

with open('input.inp', 'r') as input_file:
        for line in input_file:
            cmd_arg = line.split('=')[1].strip('\n')
            arg_list.append(cmd_arg)

print(arg_list)

# file_name = arg_list[0]

# Question if this is necessary
# result = subprocess.run(['h5dump', '-n 1', '--contents=1', file_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
# print(result.stdout)

# Call binary
subprocess.run(arg_list)