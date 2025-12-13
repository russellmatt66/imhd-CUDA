import matplotlib.pyplot as plt
import numpy as np

'''
Calculate the enclosed current profile of an r^3 Bennett Vortex for FuZE-like conditions.
'''
u_z0 = 1e5 # Edge Plasma Flow [m/s]
u_z0_array = np.array([1e4, 2e4, 5e4, 1e5, 2e5]) # m/s

r_p_max = 50e-3 # m

T_p = 1e3 # Edge Plasma Temperature [eV]
n0 = 1e23 # m^-3

mu0 = 4 * np.pi * 1e-7 # N/A^{2}
e = 1.6e-19 # C
kB = 1.38e-23 # J/K
T_p_degK = T_p * 1.1605e4 # K

# C_BT_mm = C_BT * 1e3 # mm

# rp_mm = np.linspace(0.0, r_p_max * 1e3, 1000) # mm
rp_mm = np.logspace(-2, np.log10(r_p_max * 1e3), 500) # mm
rp = rp_mm * 1e-3 # m

current_fig, current_ax = plt.subplots()

Iencl = np.zeros((len(u_z0_array), len(rp_mm)))

for i, u_z0 in enumerate(u_z0_array):
    C_BT = (mu0 * n0 * e**2 * u_z0**2 * rp**3) / (16 * kB * T_p_degK) # m
    Iencl[i, :] = (2 * np.pi * e * n0* u_z0) / (2 * (rp + C_BT)) * (
        rp**3 
        - 3 * rp**2 * C_BT 
        - 6 * rp * C_BT**2 * (1 + np.log(C_BT / (rp + C_BT))) 
        - 6 * C_BT**3 * np.log(C_BT / ((rp + C_BT)))
    )

for i in range(len(u_z0_array)):
    current_ax.plot(rp_mm, np.abs(Iencl[i]), label=f'$u_{{z,0}}$ = {u_z0_array[i]*1e-3:.1f} km/s') # There's a negative sign because of direction

# plt.plot(rp_mm, np.abs(Iencl)) # There's a negative sign because of direction 
current_ax.set_xscale('log')
current_ax.set_yscale('log')
current_ax.set_xlabel('r_p (mm)')
current_ax.set_ylabel('$I_{encl}$ (A)')
current_ax.set_title('Enclosed Current of an $r^{3}$ FuZE-like Bennett Vortex')
current_ax.legend()

plt.show()