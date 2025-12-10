import matplotlib.pyplot as plt
import numpy as np

'''
Plot the flow and shear of an r^3 Bennett Vortex FuZE-like conditions.
'''

u_z0 = 1.0e4 # Edge Plasma Flow [m/s]
r_p = 30e-3 # m
T_p= 1e3 # eV
n0 = 1e23 # m^-3

mu0 = 4 * np.pi * 1e-7 # N/A^{2}
e = 1.6e-19 # C
kB = 1.38e-23 # J/K 

T_p_degK = T_p * 1.1605e4 # K

C_BT = (mu0 * n0 * e**2 * u_z0**2 * r_p**3) / (16 * kB * T_p_degK) # m
C_BT_mm = C_BT * 1e3 # mm

# r = np.logspace(-6, -1, 500) # m
r_mm = np.linspace(0.0, 20, 500) # mm
# r_mm = r # mm

u_z = u_z0 * (r_mm**2) / (C_BT_mm + r_mm)**2 # m/s

fuze_fig, fuze_ax = plt.subplots()
fuze_ax.plot(r_mm, u_z * 1e-3, label='$u_{z}$') # km/s
# fuze_ax.set_xscale('log')
fuze_ax.set_xlabel('r (mm)')
fuze_ax.set_ylabel('$u_{z}$ (km/s)')
fuze_ax.set_title('$r^{3}$ Bennett Vortex FuZE-like Flow')

fuze_ax.set_xlim(r_mm[0], r_mm[-1])
fuze_ax.set_ylim(0.0, 2*u_z0 * 1e-3)

fuze_ax.axhline(u_z0 * 1e-3, color='red', linestyle='--', label='$u_{z,0}$')

plt.legend()

plt.show()