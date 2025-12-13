import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

'''
Calculate the minimum length threshold for an r^3 Bennett Vortex under FuZE-like conditions.
'''

# Physical Constants
mu0 = 4 * np.pi * 1e-7 # N/A^{2}
e = 1.6e-19 # C
mH = 1.67e-27 # kg
kB = 1.38e-23 # J/K

# eV to Kelvin conversion
eV_to_K = 1.1605e4 # K/eV

# FuZE-like Parameters
Tp = 1e3 # Edge Plasma Temperature [eV]
Tp_degK = Tp * eV_to_K # Convert to Kelvin
n0 = 1e23 # m^-3

u0_array = np.array([1e4, 2e4, 5e4, 1e5, 2e5]) # m/s

rp_max = 50e-3 # m

# rp_array = np.linspace(5e-3, rp_max, 5) # m
rp_array_mm = np.array([5, 10, 20, 30, 50]) # mm
rp_array = rp_array_mm * 1e-3 # m

phi = np.logspace(-3, 0, 1000) # Dimensionless coordinate
# phi = np.linspace(0.01, 10, 1000) # Dimensionless coordinate


L_min = np.zeros((len(u0_array), len(rp_array), phi.shape[0])) # m
for i, u0 in enumerate(u0_array):
    for j, rp in enumerate(rp_array):
        # Calculate the minimum length threshold
        C_BT = (mu0 * e**2 * u0**2 * rp**3 * n0) / (16 * kB * Tp_degK)
        # Full Expression
        L_min[i, j, :] = (np.pi / 20) * (mu0 * e * n0) / (np.sqrt(mH * n0 * mu0)) * C_BT**2 * (phi + 1)**2 / phi**2 * (
                        phi**3 - 3*phi**2 - 6*phi + 6*phi*np.log(phi + 1) + 6*np.log(phi + 1))
        # Second Order Approximation

# Need to plot C_BT for each rp and u0
C_BT_array = np.zeros((len(u0_array), len(rp_array))) # [m]
phi_p = np.zeros((len(u0_array), len(rp_array))) # dimensionless pinch radius
for i, u0 in enumerate(u0_array):
    for j, rp in enumerate(rp_array):
        C_BT_array[i, j] = (mu0 * e**2 * u0**2 * rp**3 * n0) / (16 * kB * Tp_degK)
        phi_p[i, j] = rp / C_BT_array[i, j]

colors = ['blue', 'orange', 'green', 'red', 'purple']

# Plot the results
for i, u0 in enumerate(u0_array):
    plt.figure(figsize=(10, 6))
    
    for j, rp in enumerate(rp_array):
        plt.plot(phi, L_min[i, j, :], label=f'rp={rp * 1e3:.1f} mm', color=colors[j])
        plt.axvline(x=phi_p[i, j], color=colors[j], linestyle='--')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$\\phi = \\frac{r}{C_{B,T}}$')
    plt.ylabel('$L_{min}$ (m)')
    plt.title(f'Minimum Length Threshold for an $r^{{3}}$ Bennett Vortex\nu0 = {u0*1e-3:.0f} km/s (FuZE-like Conditions)')
    plt.legend()

# plt.figure(figsize=(10, 6))
# for i, u0 in enumerate(u0_array):
#     for j, rp in enumerate(rp_array):
#         plt.plot(phi, L_min[i, j, :], label=f'u0={u0*1e-3} km/s, rp={rp * 1e3} mm')

# plt.xlabel('phi')
# plt.ylabel('L_min (m)')
# plt.title('Minimum Length Threshold for an $r^{3}$ Bennett Vortex under FuZE-like Conditions')
# plt.legend()

plt.figure(figsize=(10, 6))
cbt_contour = plt.contourf(rp_array_mm, u0_array * 1e-3, C_BT_array, levels=50, cmap='viridis', norm=LogNorm())
plt.colorbar(cbt_contour, label='$C_{B,T}$ (m)')
plt.xlabel('rp (mm)')
plt.ylabel('u0 (km/s)')
plt.title('$C_{B,T}$ for Different $r_{p}$ and $u_{0}$ under FuZE-like Conditions')


plt.show()