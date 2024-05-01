import subprocess

# Read `imhd-CUDA.inp`, and parse it.
arg_list = []
arg_list.append('./imhd_cuda')

with open('imhd-CUDA.inp', 'r') as input_file:
    for line in input_file:
        cmd_arg = line.split('=')[1]
        arg_list.append(cmd_arg)

# Run `imhd_cuda` binary
subprocess.run(arg_list)