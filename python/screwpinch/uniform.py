'''
This code is for astrophysical plasmas that we can model with Ideal MHD
Meaning, L is going to be very large. 
'''
# Calculates based on a uniform, magnetized plasma: 
# (/del_{r} = /del_{/theta} = /del_{z} = 0)
# \vec{J} = 0
# \vec{B} = B_{0}\hat{z}

import numpy as np
import constants as cnst
import conversions as cnv

import sys 

# PROBLEM VARIABLES
R = 1e0 # [pc]
L = 1e3 # [pc]
n0 = 1e10 # [m^-3]
gamma = 5.0 / 3.0

B0 = 0.5 # [T]

T = 1e0 # [eV]
def e(T: float): # [J]
    return T*cnv.eV_to_degK * cnst.kB * n0 / (gamma - 1.0) + 0.5 * (1.0 / cnst.mu0) * B0**2 

E = np.pi * R**2 * L * e(T) * cnv.pc_to_m**3 
print(f"The total energy of the plasma is {E} [J m^-3]")

magT = B0**2 / (2.0 * cnst.mu0)
beta = n0 * cnst.kB * T * cnv.eV_to_degK / (magT)
print(f"The beta of the plasma is {beta}")

p = n0 * cnst.kB * T * cnv.eV_to_degK
print(f"The plasma pressure is {p} [Pa]")
# n_crit = (cnst.eps0 * cnst.kB * T) / (cnst.q_e * L**2 * cnv.pc_to_m**2)
# print(f"The critical density for this plasma is {n_crit} [m^-3]")
