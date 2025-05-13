import constants as cnst
import conversions as cnv

from math import sqrt
import numpy as np

'''
The module which holds all the functionality for the plasma-parameter calculations
Don't expect performance - very much written in the style of a C or C++ library
Shouldn't need that much of it anyways
'''
# Magnetic Reynold's Number
def Rm(mu: float, u: float, sigma: float, L: float) -> float:
    return mu * u * sigma * L

# (Species) Debye Length
def lambda_D(T: float, n: float, Z: float) -> float: # [m]
    '''
    [T] = [degK]
    [n] = [m^{-3}]
    '''
    return sqrt((cnst.eps0 * cnst.kB * T) / (n * (Z*cnst.q_e)**2))

# Maximum impact parameter (normalized to r_{0} - which is the radius of the collision cross-section)
def Lambda(T: float, n: float, Z: float):
    return 12.0 * cnst.pi * n * lambda_D(T, n, Z)**3  

def spitzer_resistivity(T: float, n: float, Z: float, m: float):
    # debye_L = lambda_D(T, n, Z)
    coulomb_log = np.log(Lambda(T, n, Z)) 
    coeff = (np.pi * (Z*cnst.q_e)**2 * sqrt(n)) / ((4.0 * np.pi * cnst.eps0**2) * (cnst.kB * T)**(3/2))
    return coeff * coulomb_log

