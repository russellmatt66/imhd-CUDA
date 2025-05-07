from math import pi
import conversions as cnv

# MASS
m_H = 1.00784 * cnv.amu_to_kg # [kg] atomic mass of hydrogen

# E&M
e = 1.602e-19 # [C]
q_e = 1.6e-19 # [C] elementary electric charge

mu0 = pi * 4.0e-7 # [H m^-1] vacuum permeability
eps0 = 8.854e-12 # [F m^-1] vacuum permitivitty

# THERMO
kB = 1.380649e-23 # [J K^-1] Boltzmann constant

kappa_D = 1.3007e2 # [W K^{-1}] thermal conductivity of deuterium