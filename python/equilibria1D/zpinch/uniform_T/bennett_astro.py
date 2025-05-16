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
L = 5e0 # length of plasma column [ly]
R = 1e-3 # radius of plasma column [ly]

R_meters = R * cnv.ly_to_m
print(f"The radius is {R_meters} [m]")
rp_coeff = float(sys.argv[1])
# rp_coeff = np.sqrt(2.0 / 3.0) # r = rp_coeff * R is where the magnetic field has a maximum

r = R_meters * rp_coeff  # [R] = [ly]
print(f"The region under consideration is {rp_coeff} times the plasma column radius {R} [l.y.]")

A = np.pi * R_meters**2 # Cross-sectional area of plasma column [pc^2] -> [m^2]

# 
T = 1e0 # [eV]
m_D = 2.0 * cnst.m_H # mass of deuterium [kg]

# Axisymmetric current
# Entirely new relations for the density gradient case
I0 = 1e9 # total plasma current  [A]
u0 = 1e2 # uniform flow speed [m / s]

n0 = 1e14 # core plasma density [m^{-3}]

# Bennett Relation
N = (cnst.mu0 * I0**2) / (16.0 * np.pi * cnst.kB * T * cnv.eV_to_degK)

print(f"The linear density of plasma particles is {N} [m^{-1}]")