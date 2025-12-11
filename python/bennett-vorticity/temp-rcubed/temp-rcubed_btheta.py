import matplotlib.pyplot as plt
import numpy as np

'''
Plot the normalized azimuthal magnetic field profile of an r^3 Bennett Vortex.
'''

phi = np.logspace(-6, 1, 500) # r / C_{B,T}

btheta_tilde = (1.0) / (phi * (phi + 1)) * (phi**3 - 3*phi**2 - 6*phi + 6*phi* np.log(1 + phi) + 6*np.log(phi + 1))  # normalized B_theta profile - btheta_tilde = B_theta / B_theta_max

plt.plot(phi, btheta_tilde)
plt.xlabel('$\\phi = r / C_{B,T}$')
plt.ylabel('$\\tilde{B}_{\\theta} = \\frac{B_{\\theta}}{\\mu_{0}e n_{0}u_{z,0}C_{B,T}}$')
plt.title('Normalized Azimuthal Magnetic Field Profile of an $r^{3}$ Bennett Vortex')
plt.xscale('log')
# plt.grid(True)
plt.show()