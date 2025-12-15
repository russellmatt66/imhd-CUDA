import matplotlib.pyplot as plt
import numpy as np

phi = np.logspace(-2, 2, 100)

duzdr_tilde = 1 / (1 + phi)**2

plt.plot(phi, duzdr_tilde)
plt.xlabel('$\\phi$')
plt.ylabel('$\\tilde{\\frac{{du_z}}{{dr}}}$')
plt.xscale('log')

plt.title('Shear of a Uniform Cubic Bennett Vortex')

plt.show()