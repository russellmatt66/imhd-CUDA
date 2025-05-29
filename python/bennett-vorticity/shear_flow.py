import numpy as np
import matplotlib.pyplot as plt

# import constants as cnst
# import conversions as cnv
# import bvortex_module as bvm
'''
'''
r_star = np.linspace(0.0, 5.0, num=1000)

dudr = -(r_star) / (1 + r_star**2)**3

plt.plot(r_star, dudr)
plt.xlabel('r*')
plt.ylabel('$\\tilde{S_{f}}$')

plt.xlim(r_star[0], r_star[-1])
plt.ylim(dudr[0], np.min(dudr))

plt.title('Normalized flow shear of the Bennett Vortex')

plt.show()