import numpy as np
import matplotlib.pyplot as plt

'''
Obtains plots for Bennett Vortex profiles
- 
'''
r_star = np.linspace(0.0, 5.0, num=1000)

# Normalized shear flow profile
dudr = (r_star) / (1 + r_star**2)**3 # normalized by 4\xi

shear_fig, shear_ax = plt.subplots()
shear_ax.plot(r_star, dudr)
shear_ax.set_xlabel('r*')
shear_ax.set_ylabel('$\\tilde{S_{f}}$')

shear_ax.set_xlim(r_star[0], r_star[-1])
shear_ax.set_ylim(dudr[0], np.max(dudr))

shear_ax.set_title('Normalized flow shear of the Bennett Vortex')

# Normalized current density
jz_norm = -1.0 / (1.0 + r_star**2)**2

jz_fig, jz_ax = plt.subplots()
jz_ax.plot(r_star, jz_norm)
jz_ax.set_xlabel('r*')
jz_ax.set_ylabel('$\\tilde{J}_{z}$')

jz_ax.set_xlim(r_star[0], r_star[-1])
jz_ax.set_ylim(0.0, np.min(jz_norm))

jz_ax.set_title('Normalized current density of the Bennett Vortex')

# Normalized magnetic field and tension
Btheta_norm = -0.5 * r_star / (1 + r_star**2)
Btens_norm = 0.25 * r_star / (1 + r_star**2)**2

Btheta_fig, Btheta_ax = plt.subplots()
Btheta_ax.plot(r_star, Btheta_norm)
Btheta_ax.set_xlabel('r*')
Btheta_ax.set_ylabel('$\\tilde{B}_{\\theta}$')

Btheta_ax.set_xlim(r_star[0], r_star[-1])
Btheta_ax.set_ylim(0.0, np.min(Btheta_norm))

Btheta_ax.set_title('Normalized magnetic field of the Bennett Vortex')

Btens_fig, Btens_ax = plt.subplots()
Btens_ax.plot(r_star, Btens_norm)
Btens_ax.set_xlabel('r*')
Btens_ax.set_ylabel('$\\tilde{M}_{T}$')

Btens_ax.set_xlim(r_star[0], r_star[-1])
Btens_ax.set_ylim(0.0, np.max(Btens_norm))

Btens_ax.set_title('Normalized magnetic tension of the Bennett Vortex')

# Normalized plasma pressure
p_norm = 1 - r_star**2 * (2.0 + r_star**2) / (1 + r_star**2)**2

p_fig, p_ax = plt.subplots()
p_ax.plot(r_star, p_norm)
p_ax.set_xlabel('r*')
p_ax.set_ylabel('$\\tilde{p}$')

p_ax.set_xlim(r_star[0], r_star[-1])
p_ax.set_ylim(0.0, np.max(p_norm))

p_ax.set_title('Normalized plasma pressure of the Bennett Vortex')

# Normalized electric field
E_norm = r_star / (1 + r_star**2)**3

E_fig, E_ax = plt.subplots()
E_ax.plot(r_star, E_norm)
E_ax.set_xlabel('r*')
E_ax.set_ylabel('$\\tilde{E}_{r}$')

E_ax.set_xlim(r_star[0], r_star[-1])
E_ax.set_ylim(0.0, np.max(E_norm))

E_ax.set_title('Normalized electric field')

plt.show()