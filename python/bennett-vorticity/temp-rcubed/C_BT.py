import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

'''
Plot values of C_BT for different plasma states
'''
mu0 = 4*np.pi*1e-7 # permeability of free space in Tm/A
e = 1.602176634e-19 # elementary charge in C
k_B = 1.380649e-23 # Boltzmann constant in J/K

num_points = 1000

u_z0 = np.logspace(4, 7, num_points) # plasma velocity in m/s

# rp = np.logspace(-4, 4, num_points) # pinch radius in meters
# rp_mm = rp*1e3 # pinch radius in mm
rp_mm = np.logspace(0, 2, num_points) # pinch radius in mm
rp = rp_mm*1e-3 # pinch radius in meters

n0 = 1e23 # plasma density in m^-3
Tp = np.array([1e3, 2e3, 5e3, 10e3, 20e3]) # edge plasma temperature in eV
# Tp = 1e3 # edge plasma temperature in eV
# Tp = 2e3 # edge plasma temperature in eV
Tp_degK = Tp*11604.52 # plasma temperature in Kelvin

C_BT = np.zeros((num_points, num_points, len(Tp_degK)))

for i in range(num_points):
    for j in range(num_points):
        for k in range(len(Tp_degK)):
            C_BT[i, j, k] = (mu0 * e**2 * u_z0[i]**2 * rp[j]**3 *n0) / (16 * k_B * Tp_degK[k])

for k in range(len(Tp_degK)):
    plt.figure(figsize=(8, 6))
    plt.contourf(rp_mm, u_z0, C_BT[:, :, k], levels=20, cmap='viridis', norm=LogNorm())
    plt.colorbar(label='C_BT')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Pinch Radius (mm)')
    plt.ylabel('Plasma Velocity (m/s)')
    plt.title(f'$C_{{B,T}}$ for Tp = {Tp[k] * 1e-3} keV, and $n_{0}$ = {n0} $m^{{-3}}$')

# plt.contourf(rp_mm, u_z0, C_BT, levels=20, cmap='viridis', norm=LogNorm())
# plt.colorbar(label='C_BT')
# plt.xscale('log')
# plt.yscale('log')

# plt.xlabel('Pinch Radius (mm)')
# plt.ylabel('Plasma Velocity (m/s)')
# plt.title('$C_{B,T}$ for FuZE-like conditions')

plt.show()