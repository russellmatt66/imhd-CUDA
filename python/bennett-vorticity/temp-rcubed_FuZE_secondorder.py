import numpy as np
import matplotlib.pyplot as plt

'''
Predict the length threshold for shear-flow stabilization using a second-order series expansion for the logarithmic terms of the magnetic field.
'''

mu0 = 4*np.pi*1e-7  # Permeability of free space in H/m
e = 1.602e-19     # Elementary charge in C
kB = 1.381e-23    # Boltzmann constant in J/K
mH = 1.673e-27   # Mass of hydrogen atom in kg

eV_to_K = 11604.5  # Conversion factor from eV to Kelvin

# FuZE parameters
# L = 0.5 # [m]
Tp = 1 # [keV]
Tp_degK = Tp * eV_to_K * 1e3 # Convert temperature to Kelvin
n0 = 1e23 # [m^-3]
u0 = 1e4 # [m/s]
rp = 1e-3 # [m]

def phi_C(n0: float, u0: float, rp: float, Tp: float) -> float:
    term1 = n0 * u0**2 * rp**3 / Tp # assumed [Tp] = [degK]
    term2 = (e**5 * np.pi / 20) * np.sqrt(n0 * mu0 / mH) * (mu0 / (16.0 * kB))**2 
    return term1**2 * term2

def C_B(u0: float) -> float:
    return (mu0 * e**2 * u0**2) / (16.0 * kB)

def C_T(rp: float, Tp: float) -> float:
    return Tp / rp**3

def C_BT(n0: float, rp: float, Tp: float, u0: float) -> float:
    return C_B(u0) / C_T(rp, Tp) * n0

def error_sorder_coeff(C_BT: float, n0: float) -> float:
    return 0.3 * C_BT**2 * e * np.pi * np.sqrt(mu0 * n0 / mH) 

nphi = 200
phi = np.logspace(-6, 1, nphi) # r / C_{B,T}

threshold_val = phi_C(n0, u0, rp, Tp_degK) * (phi + 1)**2 * (phi + 3) # second-order expansion

plt.semilogy(phi, threshold_val, label='Threshold Value for Shear-Flow Stabilization')
plt.ylim(threshold_val.min(), threshold_val.max()*10)
plt.xlim(phi.min(), phi.max())
plt.ylabel('L [m]')
plt.xlabel('$r / C_{B,T}$')

plt.title('Shear-Flow Stabilization Length Threshold of an $r^{3}$, FuZE-like, Bennett Vortex')

ax = plt.gca()
y_min, y_max = ax.get_ylim() 
plt.fill_between(phi, threshold_val, y_max, color='gray', alpha=0.5, label='Stable Region')

error_term = error_sorder_coeff(C_BT(n0, rp, Tp_degK, u0), n0) * (1 + phi)**3
plt.semilogy(phi, error_term, '--', color='red', label='Error Estimate (Second-Order)')

plt.legend()
plt.show()