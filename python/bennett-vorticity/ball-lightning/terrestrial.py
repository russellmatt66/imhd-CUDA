import numpy as np
import matplotlib.pyplot as plt

'''
Calculating properties of a terrestrial cubic Bennett Vortex
'''
mu0 = 4e-7 * np.pi  # Permeability of free space [N / A^2]
eps0 = 8.85e-12  # Permittivity of free space [F / m]
e = 1.6e-19  # Elementary charge [C]
kB = 1.38e-23  # Boltzmann constant [J / K]
m_e = 9.11e-31  # Electron mass [kg]

rp = 1 # pinch radius [m]
uz0 = 1e-3 # root of the axial velocity [m/s]
Tp = 1e3 # edge plasma temperature [degK]
n0 = 1e25 # plasma number density [m^-3]

L = 0.5 # pinch length [m]

Zeff = 1
lnCoulomb = 10  # Coulomb logarithm, typical value

def C_BT():
    Cbt_1 = (mu0 * e**2) / (16 * kB) 
    Cbt_2 = n0 * uz0**2 * rp**3 / Tp
    return Cbt_1 * Cbt_2

def p0_CUBIC(Cbt, rp):
    '''
    Calculate the pressure at the center of the cubic, $\chi = 2$ Bennett vortex 
    '''
    p0 = 1/(2*(Cbt + rp)**2) * (rp * (2*Cbt + rp)*(-15*Cbt**2 - 12*Cbt*rp + rp**2) 
                           + 6 * Cbt**2 * (Cbt + rp) * np.log(Cbt/(Cbt + rp)) 
                           * (-5 * Cbt - 3 * rp + (Cbt + rp) * np.log(Cbt/(Cbt + rp))))
    return mu0 * (e*n0*uz0)**2 * 0.5 * p0 

def eta_SPITZER(T_e):
    '''
    Calculate the Spitzer resistivity for a given electron temperature T_e (in eV)
    '''
    eta_1 = 4.0 * np.sqrt(2 * np.pi) / 3
    eta_2 = e**2 * np.sqrt(m_e) * lnCoulomb / (4 * np.pi * eps0)**2
    eta_3 = (kB * T_e)**(-1.5)
    return eta_1 * eta_2 * eta_3 * Zeff

def f(rp, Cbt):
    '''
    Part of the azimuthal magnetic field form
    '''
    return rp**3 - 3*rp**2 * Cbt - 6*rp*Cbt**2 * (1 + np.log(Cbt/(Cbt + rp))) - 6*Cbt**3 * np.log(Cbt/(Cbt + rp))

def BMAX(Cbt, rp):
    '''
    Calculate the maximum azimuthal magnetic field
    '''
    return mu0 * e * n0 * uz0 * f(rp, Cbt) / (2 * rp * (rp + Cbt))

def TAU_E():
    '''
    Calculate the electron collision time
    '''
    coeff1 = 6 * np.sqrt(2) * np.pi**1.5 * eps0**2
    coeff2 = np.sqrt(m_e) * (Tp)**1.5
    denom = e**4 * n0 * lnCoulomb 
    return coeff1 * coeff2 / denom

def kappa_PERP(Cbt, rp):
    '''
    Calculate the perpendicular thermal conductivity
    '''
    Bmax = BMAX(Cbt, rp)
    omega_c = e * Bmax / m_e
    tau_e = TAU_E()
    return 4.7 * (n0 * Tp) / (m_e * omega_c**2 * tau_e)

def TAU_CONFINEMENT(rp, p0, Bmax):
    '''
    Calculate the energy confinement time
    '''
    coeff1 = p0 * rp**3 / 12.0 
    coeff2 = 6 * np.sqrt(2) * np.pi**1.5 * eps0**2 / (4.7 * lnCoulomb)
    coeff3 = Bmax**2 / np.sqrt(Tp) 
    coeff4 = 1.0 / (n0**2 * m_e**1.5 * e**2)
    return coeff1 * coeff2 * coeff3 * coeff4

def P_ELECTRICAL():
    '''
    Calculate the electrical power associated with the vortex
    '''
    coeff1 = e**2 * n0**2 * uz0**2 * np.pi
    coeff2 = L / rp**4
    coeff3 = f(rp, Cbt)**2 / (Cbt + rp)**2
    return coeff1 * coeff2 * coeff3 * eta_SPITZER(Tp)

Cbt = C_BT()
p0 = p0_CUBIC(Cbt, rp)
eta_e = eta_SPITZER(Tp)
Bmax = BMAX(Cbt, rp)
kappa_perp = kappa_PERP(Cbt, rp)
tau_conf = TAU_CONFINEMENT(rp, p0, Bmax)
p_electrical = P_ELECTRICAL()

print(f"The cubic Bennett parameter Cbt is {Cbt:.2e} meters")
print(f"The axial pressure p0 is {p0:.2e} Pascals")
print(f"The energy confinement time is {tau_conf:.2e} seconds")
print(f"The electrical power is {p_electrical:.2e} Watts")