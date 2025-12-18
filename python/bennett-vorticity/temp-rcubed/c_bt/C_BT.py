import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

'''
Plot values of C_BT for different plasma states
'''
plt.close('all')  

mu0 = 4*np.pi*1e-7 # permeability of free space in Tm/A
e = 1.602176634e-19 # elementary charge in C
k_B = 1.380649e-23 # Boltzmann constant in J/K

num_points = 500

u_z0 = np.logspace(4, 7, num_points) # plasma velocity in m/s

# rp = np.logspace(-4, 4, num_points) # pinch radius in meters
# rp_mm = rp*1e3 # pinch radius in mm
rp_mm = np.logspace(-6, 27, num_points) # pinch radius in mm
rp = rp_mm*1e-3 # pinch radius in meters

n0 = 1e26 # plasma density in m^-3
Tp = np.array([1e3, 2e3, 5e3, 10e3, 20e3, 50e3, 100e3]) # edge plasma temperature in eV
# Tp = 1e3 # edge plasma temperature in eV
# Tp = 2e3 # edge plasma temperature in eV
Tp_degK = Tp*11604.52 # plasma temperature in Kelvin

C_BT = np.zeros((num_points, num_points, len(Tp_degK)))

for i in range(num_points):
    for j in range(num_points):
        for k in range(len(Tp_degK)):
            C_BT[i, j, k] = (mu0 * e**2 * u_z0[i]**2 * rp[j]**3 * n0) / (16 * k_B * Tp_degK[k])

# Contour plot of C_BT in ur-space
# for k in range(len(Tp_degK)):
#     plot_fig = plt.figure(figsize=(8, 6))
#     plot_fig.contourf(rp_mm, u_z0, C_BT[:, :, k], levels=20, cmap='viridis', norm=LogNorm())
#     plot_fig.colorbar(label='C_BT')
#     plot_fig.xscale('log')
#     plot_fig.yscale('log')
#     plot_fig.xlabel('Pinch Radius (mm)')
#     plot_fig.ylabel('Plasma Velocity (m/s)')
#     plot_fig.title(f'$C_{{B,T}}$ for Tp = {Tp[k] * 1e-3} keV, and $n_{0}$ = {n0} $m^{{-3}}$')

rp_mesh, uz0_mesh = np.meshgrid(rp, u_z0)
rp_mesh_log = np.log10(rp_mesh)
uz0_mesh_log = np.log10(uz0_mesh)


# k = 0
for k in range(len(Tp_degK)):
    fig = plt.figure(figsize=(8, 6)) 
    # ax = fig.add_subplot(121, projection='3d')
    # cbt_surf = ax.plot_surface(rp_mesh, uz0_mesh, C_BT[:, :, k], cmap='viridis', norm=LogNorm())
    # fig.colorbar(cbt_surf, label='$C_{B,T}$ (m)')
    # # ax.set_xscale('log')
    # # ax.set_yscale('log')
    # ax.set_xlabel('$r_{p}$ (m)')
    # ax.set_ylabel('$u_{z0}$ (m/s)')
    # ax.set_zlabel('$C_{B,T}$ (m)')
    # ax.set_title(f'$C_{{B,T}}$ Surface for Tp = {Tp[k] * 1e-3} keV, and $n_{{0}}$ = {n0} $m^{{-3}}$')

    ax_log = fig.add_subplot(111, projection='3d')
    cbt_surf_log = ax_log.plot_surface(rp_mesh_log, uz0_mesh_log, np.log10(C_BT[:, :, k]), cmap='viridis')
    fig.colorbar(cbt_surf_log, label='log10($C_{B,T} (m)$)')
    ax_log.set_xlabel('log10($r_{p}$) (m)')
    ax_log.set_ylabel('log10($u_{z0}$) (m/s)')
    ax_log.set_zlabel('log10($C_{B,T}$) (m)')
    ax_log.set_title(f'log10($C_{{B,T}}$) Surface for Tp = {Tp[k] * 1e-3} keV, and $n_{{0}}$ = {n0} $m^{{-3}}$')

plt.show()