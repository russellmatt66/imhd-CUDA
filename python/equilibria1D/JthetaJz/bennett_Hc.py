'''
The cross-helicity is computed in this file.

This is not a z-pinch so applying a Bennett profile to it is artificial, but 
this file does calculations for an axisymmetric equilibrium where, 

J_{\theta}(r) = en(r)u_{0,theta}
J_{z}(r) = en(r)u_{0,z}

with,

n(r) = n0 / (1 + b n0 r^2)^2

The current now has a vortical nature to it, and the total plasma current is no 
longer the same thing as the axial plasma current

The magnetic field can be analytically obtained. 
'''
import numpy as np

import sys
sys.path.insert(0, '../lib/')

import bennett
import constants as cnst
import conversions as cnv

# Bulk axial plasma current
def Iz(R: float, n0: float, u0: float, u0_z: float, T: float) -> float:
    '''
    u0 - magnitude of the flow speed
    u0_z - magnitude of the axial flow speed

    '''
    b = bennett.b_get(u0, T)
    nonint_portion = n0 * R**2 / (1 )
    return np.pi * cnst.e * u0_z * nonint_portion

# Bulk azimuthal plasma current
def Itheta(R: float, L: float, n0: float, u0: float, u0_theta: float, T: float) -> float:
    b = bennett.b_get(u0, T)
    xi = np.sqrt(b * n0) # This makes things more readable
    
    integral_portion = 0.5 * n0 * ((R / (1 + xi**2 * R**2)) + np.arctan(R * xi) / xi)
    nonint_portion = n0 / (1 + b*n0*R**2)**2

    res = (cnst.e * u0_theta * L) * (2.0 * integral_portion + nonint_portion) 
    return res

L = 1e0 # [m]
R = 1e0 # [cm]
R_meters = R *cnv.cm_to_meter

n0 = 1e21 # [m^-3]
T = 1e3 # [eV]
T_K = T * cnv.eV_to_degK

u0 = 1e3 # [m / s]
alpha = 0.1 # 

u0_theta = alpha * u0
u0_z = np.sqrt(1 - alpha**2) * u0
print(f"The azimuthal flow speed is {u0_theta}, and the axial flow speed is {u0_z} [m / s]")

b = bennett.b_get(u0, T_K)

Iz_mag = Iz(R_meters, n0, u0, u0_z, T_K)
Itheta_mag = Itheta(R_meters, L, n0, u0, u0_theta, T_K)

print(f"The bulk azimuthal plasma current is {Itheta_mag} [A]")
print(f"The bulk axial plasma current is {Iz_mag} [A]")

B0_z = 1.0 # [T]

A = cnst.mu0 * cnst.e * u0_theta * u0_z * n0 # problem-specific bundle of constants that shows up everywhere

xi = np.sqrt(b * n0) 

Hc = np.pi * L * (R_meters**2 * u0_z * B0_z - A / xi * R_meters * np.arctan(R_meters * xi) + 0.5*A / xi**2 *np.log(1 + R_meters**2*xi**2))

print(f"The cross-helicity is {Hc}")