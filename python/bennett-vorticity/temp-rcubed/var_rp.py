import matplotlib.pyplot as plt
import numpy as np

'''
Plot the:

- flow  
- shear 
- Btheta

of an r^3 Bennett Vortex FuZE-like conditions.
'''

# u_z0 = 1.0e4 # Edge Plasma Flow [m/s]
u_z0 = 1.0e5 # Edge Plasma Flow [m/s]
r_p = 10e-3 # m
T_p = 1e3 # Edge Plasma Pressure [eV]
n0 = 1e23 # m^-3

r_p_array = np.array([5e-3, 10e-3, 15e-3, 20e-3, 30e-3]) # m

mu0 = 4 * np.pi * 1e-7 # N/A^{2}
e = 1.6e-19 # C
kB = 1.38e-23 # J/K 

T_p_degK = T_p * 1.1605e4 # K

# C_BT = (mu0 * n0 * e**2 * u_z0**2 * r_p**3) / (16 * kB * T_p_degK) # m
C_BT = (mu0 * n0 * e**2 * u_z0**2 * r_p_array**3) / (16 * kB * T_p_degK) # m
C_BT_mm = C_BT * 1e3 # mm


# r_mm = 4*np.logspace(-2, 2, 1000) # mm
r_mm = np.linspace(1e-3, max(r_p_array) * 1e3, 1000) # mm
# r_mm = r # mm
r = r_mm * 1e-3 # m

u_z = np.zeros((len(r_p_array), len(r_mm))) # m/s
for i, r_p_val in enumerate(r_p_array):
    C_BT_val = (mu0 * n0 * e**2 * u_z0**2 * r_p_val**3) / (16 * kB * T_p_degK) # m
    C_BT_mm_val = C_BT_val * 1e3 # mm
    u_z[i, :] = u_z0 * (r_mm**2) / (C_BT_mm_val + r_mm)**2 # m/s

duz_dr = np.zeros((len(r_p_array), len(r_mm)))

for i, r_p_val in enumerate(r_p_array):
    C_BT_val = (mu0 * n0 * e**2 * u_z0**2 * r_p_val**3) / (16 * kB * T_p_degK) # m
    C_BT_mm_val = C_BT_val * 1e3 # mm
    duz_dr[i, :] = u_z0 * ((2*r_mm) / ((r_mm + C_BT_mm_val)**2) - (2*r_mm**2) / ((r_mm + C_BT_mm_val)**3))
    duz_dr[i, :] = duz_dr[i, :] * 1e3 # convert to 1/s from m s^{-1} mm^{-1}

fuze_fig, fuze_ax = plt.subplots()

for i in range(len(r_p_array)):
    fuze_ax.plot(r_mm, u_z[i] * 1e-3, label='$r_p$ = {:.1f} mm'.format(r_p_array[i]*1e3)) # km/s

# fuze_ax.set_xscale('log')
fuze_ax.set_xlabel('r (mm)')
fuze_ax.set_yscale('log')
fuze_ax.set_ylabel('$u_{z}$ (km/s)')
fuze_ax.set_title('$r^{3}$ Bennett Vortex FuZE-like Flow')

fuze_ax.set_xlim(r_mm[0], r_mm[-1])
fuze_ax.set_ylim(1e-3, 2*u_z0 * 1e-3)
# fuze_ax.set_ylim(0.0, 2*u_z0 * 1e-3)

fuze_ax.axhline(u_z0 * 1e-3, color='red', linestyle='--', label=f'$u_{{z,0}}$ = {u_z0*1e-3:.1f} km/s')

fuze_ax.legend()

fuze_nonlog_fig, fuze_nonlog_ax = plt.subplots()
for i in range(len(r_p_array)):
    fuze_nonlog_ax.plot(r_mm, u_z[i] * 1e-3, label='$r_p$ = {:.1f} mm'.format(r_p_array[i]*1e3)) # km/s

fuze_nonlog_ax.set_xlabel('r (mm)')
fuze_nonlog_ax.set_ylabel('$u_{z}$ (km/s)')
fuze_nonlog_ax.set_title('$r^{3}$ Bennett Vortex FuZE-like Flow')
fuze_nonlog_ax.set_xlim(r_mm[0], r_mm[-1])
fuze_nonlog_ax.set_ylim(0.0, 2*u_z0 * 1e-3)
fuze_nonlog_ax.axhline(u_z0 * 1e-3, color='red', linestyle='--', label=f'$u_{{z,0}}$ = {u_z0*1e-3:.1f} km/s')
fuze_nonlog_ax.legend()

fuzeshear_fig, fuzeshear_ax = plt.subplots()
for i in range(len(r_p_array)):
    fuzeshear_ax.plot(r_mm, duz_dr[i], label=f'$r_{{p}}$ = {r_p_array[i]*1e3:.1f} mm') # 1/s
    print(f'Max duz_dr for r_p = {r_p_array[i]*1e3:.1f} mm: {np.max(duz_dr[i])} 1/s')

# fuzeshear_ax.set_xscale('log')
fuzeshear_ax.set_xlabel('r (mm)')

fuzeshear_ax.set_yscale('log')
fuzeshear_ax.set_ylabel('$\\frac{du_{z}}{dr}$ (1/s)')
fuzeshear_ax.set_ylim(1e1, 1e8)

fuzeshear_ax.set_title('$r^{3}$ Bennett Vortex FuZE-like Shear Flow')

fuzeshear_ax.legend()

btheta = np.zeros((len(r_p_array), len(r_mm)))
for i, r_p_val in enumerate(r_p_array):
    C_BT_val = (mu0 * n0 * e**2 * u_z0**2 * r_p_val**3) / (16 * kB * T_p_degK) # m
    C_BT_mm_val = C_BT_val * 1e3 # mm
    phi = r_mm / C_BT_mm_val
    btheta[i, :] = (mu0 * e * n0 * u_z0 * C_BT_val) / (phi * (phi + 1)) * (
        phi**3 - 3*phi**2 - 6*phi + 6*phi* np.log(1 + phi) + 6*np.log(phi + 1))  # B_theta in Tesla
    # btheta[i, :] = (mu0 * e * n0 * u_z0) / (2 * r_mm * (r_mm + C_BT_mm_val)) * (
    #     r_mm**3 - 3*r_mm**2 * C_BT_mm_val - 6*r_mm * C_BT_mm_val**2 *(1 - np.log(1 + r_mm / C_BT_mm_val)) 
    #     + 6 * C_BT_mm_val**3 * np.log(1 + r_mm / C_BT_mm_val))  # B_theta in Tesla

btheta_fig, btheta_ax = plt.subplots()
for i in range(len(r_p_array)):
    btheta_ax.plot(r_mm, btheta[i], label=f'$r_{{p}}$ = {r_p_array[i]*1e3:.1f} mm') # Tesla

btheta_ax.set_xlabel('r (mm)')
# btheta_ax.set_yscale('log')
btheta_ax.set_ylabel('$B_{\\theta}$ (T)')
btheta_ax.set_title('$r^{3}$ Bennett Vortex FuZE-like Azimuthal Magnetic Field')
btheta_ax.set_xlim(r_mm[0], r_mm[-1])
btheta_ax.set_ylim(1e-4, 1e1)
btheta_ax.legend()

plt.show()