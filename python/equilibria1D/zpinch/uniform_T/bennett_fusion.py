'''
Calculates axisymmetric zpinch equilibria for a uniform 
temperature plasma that has a Bennett profile.  

Meaning,

(/del_{/theta} = /del_{z} = 0)
T = T_{0}
n = n(r) = n_{B}(r)
\vec{J} = J_{z}(r) \hat{z} = eu_{0}n(r)\hat{z}
\vec{B} = B_{\theta}(r) \hat{\theta}
e = e(\rho, p, \vec{J}, \vec{B})
'''

# TODO:
# - Refactor to send output to a log instead of printing to stdout
import sys 

sys.path.append("../../lib")

import numpy as np

import constants as cnst
import conversions as cnv
import plasma_library as pl
import bennett

# PROBLEM VARIABLES
L = 1 # length of plasma column [m]
R = 1e0 # radius of plasma column [cm]

R_meters = R * cnv.cm_to_meter
print(f"The radius is {R_meters} [m]")
# rp_coeff = float(sys.argv[1])
rp_coeff = np.sqrt(2.0 / 3.0) # r = rp_coeff * R is where the magnetic field has a maximum

r = R_meters * rp_coeff  # [R] = [cm]
print(f"The region under consideration is {rp_coeff} times the plasma column radius {R} [cm]")

A = np.pi * R_meters**2 # Cross-sectional area of plasma column [m^2]

# 
T = 1e3 # [eV]
m_D = 2.0 * cnst.m_H # mass of deuterium [kg]

# Axisymmetric current
# Entirely new relations for the density gradient case
I0 = 1e9 # total plasma current  [A]
u0 = 1e3 # uniform flow speed [m / s]

# Bennett Relation for Ideal MHD w/uniform temperature
N = (cnst.mu0 * I0**2) / (16.0 * np.pi * cnst.kB * T * cnv.eV_to_degK)

print(f"The linear density of plasma particles is {N} [m^{-1}]")

n0 = 1e21 # core plasma density [m^{-3}]

# "Bennett parameter"
b = bennett.b_get(u0, T * cnv.eV_to_degK)
print(f"The Bennett parameter is {b}")

Phi_I = bennett.Phi_I(u0, b, n0, R)
print(f"The net flux of electromagnetic power through an end-cap is {Phi_I} [W]")