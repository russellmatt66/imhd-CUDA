# Calculates based on an axisymmetric configuration: 
# (/del_{/theta} = /del_{z} = 0)
# \vec{J} = J_{z}\hat{z}
# \vec{B} = B_{\theta}\hat{\theta} + B_{0}\hat{z}

import numpy as np
import constants as cnst
import conversions as cnv

import sys 

# PROBLEM VARIABLES
L = 1e3 # length of plasma column [ly]
R = 1e0 # radius of plasma column [ly]

R_meters = R * cnv.ly_to_m
# R_meters = R * cnv.km_to_m

A = np.pi * R_meters**2 # Cross-sectional area of plasma column [pc^2] -> [m^2]

I0 = 5e17 # total plasma current  [A]
J0 = 2 * I0 / (A) # peak current density [A m^-2]
print(f"The peak current density (at the axis) is {J0} [A m^-2]")

m_D = 2.0 * cnst.m_H # mass of deuterium [kg]
n0 = 1e14 # plasma density [m^-3]

rho = m_D * n0 # plasma mass density [kg * m^-3]

B_0 = 1e3 # uniform, axial magnetic field [T]

rp_coeff = float(sys.argv[1])
r = R_meters * rp_coeff  # [R] = [ly]
print(f"The region under consideration is {rp_coeff} times the plasma column radius {R} [pc]")

def uz(r): # axial flow velocity - Bennett profile
    if r > R_meters:
        return 0
    return (m_D * J0) / (cnst.q_e * rho) * (1.0 - r**2 / R_meters**2)

print(f"The flow velocity here is {uz(r)} [m s^-1]")

def Btheta(r): # azimuthal magnetic field
    if r > R_meters:
        return 0
    return cnst.mu0 * J0 * r * 0.5 * (1.0 - 0.5 * r**2 / R_meters**2)

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
    if r > R_meters:
        temp = 2.0 * R_meters**6 - 9.0 * R_meters**4 * R_meters**2 + 12 * R_meters**2 * R_meters**4
    else:
        temp = 2.0 * r**6 - 9.0 * r**4 * R_meters**2 + 12 * r**2 * R_meters**4
    return -(temp) / (n0 * cnst.kB) * (J0**2 * cnst.mu0) / (48.0 * (R_meters)**4)

print(f"The temperature drop in that region is {T(r)} [K]")
print(f"The pressure drop is {T(r) / (cnst.kB * n0)} [Pa]")

p_B2 = (0.5 / cnst.mu0) * (B_0**2 + Btheta(r)**2)
p_B1 = (0.5 / cnst.mu0) * (B_0**2)
print(f"The magnetic pressure of the uniform case is {p_B1} [Pa]")
print(f"The magnetic pressure of the Bennett column is {p_B2} [Pa]")
print(f"The difference in magnetic pressure at this point is {p_B2 - p_B1}")
print(f"The ratio of magnetic pressure at this point is {p_B2 / p_B1}")

Phi_B = 0.5 * (np.pi * R_meters**2 * L * J0 * cnst.mu0)
print(f"The total magnetic flux is {Phi_B} [T m^2]")