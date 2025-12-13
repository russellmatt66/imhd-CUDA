import numpy as np
import matplotlib.pyplot as plt

'''
Plot the normalized enclosed current profile of an r^3 Bennett Vortex.
''' 

phi_p = np.logspace(-6, 1, 500) # r_{p} / C_{B,T}

Iencl_tilde = (1.0) / ((phi_p + 1)) * (phi_p**3 - 3*phi_p**2 
                                     - 6*phi_p + 6*phi_p* np.log(1 + phi_p) 
                                     + 6*np.log(phi_p + 1))  # normalized enclosed current profile - Iencl_tilde = Iencl / (pi * e * n0 * u_z0 * C_{B,T}^2)
plt.plot(phi_p, np.abs(Iencl_tilde))
plt.xlabel('$\\phi_{p} = r_{p} / C_{B,T}$')
plt.ylabel('$\\tilde{I}_{encl} = \\frac{I_{encl}}{\\pi e n_{0} u_{z,0} C_{B,T}^{2}}$')
plt.title('Normalized Enclosed Current Profile of an $r^{3}$ Bennett Vortex')
# plt.xscale('log')
# plt.yscale('log')
# plt.grid(True)
plt.show()