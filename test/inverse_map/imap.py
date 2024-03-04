# Maps from the linear index, l, to the corresponding indices, (i, j, k), in the rank-3 tensor represented by the data cube 
import sys
import math

def f_l(i: int, j: int, k: int, Nx: int, Ny: int, Nz: int)-> int:
    return (Nx*Ny*k + Ny*i + j)

# ./imap.py l Nx Ny Nz
l = int(sys.argv[1])
Nx = int(sys.argv[2])
Ny = int(sys.argv[3])
Nz = int(sys.argv[4])

k = math.floor(l / (Nx * Ny)) 
print(f"k = {k}")
i = math.floor((l - Nx*Ny*k) / Ny)
j = l - Nx*Ny*k - Ny*i 

print(f"(i, j, k) = ({i}, {j}, {k})")
print(f"l = {f_l(i, j, k, Nx, Ny, Nz)}")
