'''
This module is for calculating Bennett equilibria,

    n(r) = n0 / (1 + b*n0*r**2)**2

where,

    b = (mu0 * e^2 * u0^2) / (16 * kB * T)
'''
import constants as cnst
import numpy as np

# Bennett Profile
def n(r: float, b: float, n0: float):
    return n0 / (1 + b*n0*r**2)**2

def Btheta(r: float, u0: float, b: float, n0: float):
    return (cnst.mu0 * cnst.e * u0 * n0) / (2.0 + 2.0*b*n0*r**2)

# Get the "Bennett parameter"
def b_get(u0: float, T: float):
    '''
    [T] - [degK]
    [u0] - [m/s]
    '''
    return (cnst.mu0 * cnst.e**2 * u0**2) / (16.0 * cnst.kB * T)

# Net flux of electromagnetic power into the plasma through one of the end-caps
def Phi_I(u0: float, b: float, n0: float, R: float):
    l_coeff = -(2.0 * np.pi * u0) / (8.0 * b**2 * n0**2)
    r_coeff = (np.log(1.0 + b*n0*R**2) + 1.0 / (1.0 + b*n0*R**2) - 1)
    return l_coeff * r_coeff