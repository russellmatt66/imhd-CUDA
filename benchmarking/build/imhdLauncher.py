import sys 
import subprocess

mode = sys.argv[1]

driver = ''
if mode == "mega":
    driver = './imhd-cuda_bench-mega'
elif mode == "micro":
    driver = './imhd-cuda_bench-micro'

# Read `imhd-CUDA.inp`, and parse it.
arg_list = []
arg_list.append(driver)

with open('imhd-cuda_bench.inp', 'r') as input_file:
    for line in input_file:
        cmd_arg = line.split('=')[1].strip('\n')
        arg_list.append(cmd_arg)

print(f"Passing {arg_list} to subprocess.run()")

# Run binary
subprocess.run(arg_list)