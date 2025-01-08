import subprocess

# Read `imhd-CUDA.inp`, and parse it.
arg_list = []
arg_list.append('./imhd-cuda_bench')

with open('imhd-cuda_bench.inp', 'r') as input_file:
    for line in input_file:
        cmd_arg = line.split('=')[1].strip('\n')
        arg_list.append(cmd_arg)

print(f"Passing {arg_list} to subprocess.run()")

# Run binary
subprocess.run(arg_list)