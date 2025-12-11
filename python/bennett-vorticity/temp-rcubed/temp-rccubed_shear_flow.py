import numpy as np 
import matplotlib.pyplot as plt

'''
Plot the normalized flow profile of an r^3 Bennett Vortex.
Plot the normalized shear flow of an r^3 Bennett Vortex.
'''

phi = np.logspace(-6, 1, 500) # r / C_{B,T}
# phi = np.linspace(0.0, 10.0, 200) # r / C_{B,T}

uz_tilde = phi**2 / (1 + phi)**2 # normalized flow profile - uz_tilde = uz / (u_z0 * C_BT)

shear_tilde = ((2 * phi) / (phi + 1)**2) - ((2 * phi**2) / (phi + 1)**3) # normalized shear flow - shear_tilde = d(uz_tilde)/d(phi)

flow_fig, flow_ax = plt.subplots()
flow_ax.plot(phi, uz_tilde)
flow_ax.set_xlabel('$r / C_{B,T}$')
flow_ax.set_ylabel('$\\tilde{u}_{z}$')

flow_ax.set_xlim(phi[0], phi[-1])
# flow_ax.set_ylim(uz_tilde[0], np.max(uz_tilde))
flow_ax.set_ylim(uz_tilde[0], 1.0)

flow_ax.set_title('Normalized Flow Profile $\\tilde{u}_{z} = \\frac{u_z}{u_{z0}}$ of an $r^{3}$ Bennett Vortex')

shear_fig, shear_ax = plt.subplots()
shear_ax.plot(phi, shear_tilde)
shear_ax.set_xlabel('$r / C_{B,T}$')
shear_ax.set_ylabel('$\\frac{d\\tilde{u}_{z}}{d\\phi)}$')

shear_ax.set_xlim(phi[0], phi[-1])
# shear_ax.set_ylim(shear_tilde[0], np.max(shear_tilde))
shear_ax.set_ylim(shear_tilde[0], 1.0)
shear_ax.set_title('Normalized Shear Flow Profile $\\frac{d\\tilde{u}_{z}}{d\phi}$ of an $r^{3}$ Bennett Vortex')

plt.show()