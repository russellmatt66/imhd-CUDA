'''
This module is for calculating vortical Bennett pinch equilibria,

    u_{z}(r) = u0 / (1 + b*n0*r**2)**2

where,

    b = (mu0 * e^2 * u0^2) / (16 * kB * T)
'''
import constants as cnst 


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