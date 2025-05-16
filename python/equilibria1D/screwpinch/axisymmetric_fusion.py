# Calculates based on an axisymmetric configuration: 
# (/del_{/theta} = /del_{z} = 0)
# \vec{J} = J_{z}\hat{z}
# \vec{B} = B_{\theta}\hat{\theta} + B_{0}\hat{z}

import numpy as np
import constants as cnst
import conversions as cnv

import sys 

# PROBLEM VARIABLES
r_p = 1e1 # pinch radius [cm]
rp_meters = r_p*cnv.cm_to_meter # Used so much

A_p = np.pi * (rp_meters)**2 # pinch area [m^2]

L = 1 # pinch length [m]

I0 = 1e6 # total plasma current  [A]
J0 = 2 * I0 / (A_p) # peak current density [A m^-2]
print(f"The peak current density (at the axis) is {J0} [A m^-2]")

m_D = 2.0 * cnst.m_H # mass of deuterium [kg]
n0 = 1e23 # plasma density [m^-3]

rho = m_D * n0 # plasma mass density [kg * m^-3]

B_0 = 1 # uniform, axial magnetic field [T]

rp_coeff = float(sys.argv[1])
# rp_coeff = np.sqrt(2.0 / 3.0) # this is the coefficient for where the magnetic field is greatest  
r = rp_coeff * rp_meters # [rp] = [cm]
print(f"The region under consideration is {rp_coeff} times the pinch radius of {r_p} [cm]")

def uz(r): # axial flow velocity - Bennett profile
    if r > rp_meters:
        return 0
    return (m_D * J0) / (cnst.q_e * rho) * (1.0 - r**2 / rp_meters**2)

print(f"The flow velocity here is {uz(r)} [m s^-1]")

def Btheta(r): # azimuthal magnetic field
    if r > rp_meters:
        return 0
    return cnst.mu0 * J0 * r * 0.5 * (1.0 - 0.5 * r**2 / rp_meters**2)

print(f"The azimuthal magnetic field here is {Btheta(r)} [T]")

# B_r = B_{0,r} / r = 0
# B_z = B_0 

u_sq = uz(r)**2
u_dot_b = uz(r) * B_0
B_sq = Btheta(r)**2 + B_0**2

P_sq = B_sq**2 * u_sq - B_sq * u_dot_b**2
print(f"The electromagnetic power flow in that region has an amplitude of {np.sqrt(P_sq)} [W m^-2]")

def T(r: float):  # Really `Delta T = T(r) - T(0)`
    temp = 0
    if r > rp_meters:
        temp = 2.0 * rp_meters**6 - 9.0 * rp_meters**4 * rp_meters**2 + 12 * rp_meters**2 * rp_meters**4
    else:
        temp = 2.0 * r**6 - 9.0 * r**4 * rp_meters**2 + 12 * r**2 * rp_meters**4
    return -(temp) / (n0 * cnst.kB) * (J0**2 * cnst.mu0) / (48.0 * (r_p*cnv.cm_to_meter)**4)

print(f"The temperature drop in that region is {T(r)} [K]")
print(f"The temperature drop in that region is {T(r) * cnv.degK_to_eV} [eV]")

# def p_drop(r: float): 

print(f"The pressure drop is {T(r) / (cnst.kB * n0)} [Pa]")

def Jz(r): 
    return n0 * cnst.q_e * uz(r) 
print(f"The axial current density at this point is {Jz(r)} [A m^-2]")

def Q_heat(r: float) -> float:
    return (cnst.kappa_D / (n0 * cnst.kB)) * Jz(r) * Btheta(r)

radial_heat_flux = Q_heat(r)
print(f"The amplitude of the heat flux at the point {rp_coeff}R is {radial_heat_flux} [W / m^-2]")

print(f"The ratio of electromagnetic power flux to thermal power flux is {np.sqrt(P_sq) / radial_heat_flux}")