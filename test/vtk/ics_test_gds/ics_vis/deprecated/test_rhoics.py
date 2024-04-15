# Test `../rho_ics.dat` to make sure it's correct
import numpy as np
import sys

Nx = int(sys.argv[1])
Ny = int(sys.argv[2])
Nz = int(sys.argv[3])

with open("../rho_ics.dat", "rb") as f:
    raw_data = f.read()

rho_float_values = np.frombuffer(raw_data, dtype=np.float32)

print(type(rho_float_values))
print(rho_float_values.shape) # Needs to match with Nx * Ny * Nz
print(Nx * Ny * Nz)

print(rho_float_values[:5])
print(rho_float_values[:-5])

print(np.nonzero(rho_float_values))
print(rho_float_values.max())
print(rho_float_values.min())
print(np.unique(rho_float_values))