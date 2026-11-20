import numpy as np
import matplotlib.pyplot as plt

phi = np.linspace(0.0, 10.0, 100)

uztilde_chi1 = phi / (phi + 1)
uztilde_chi1_inverse = (phi + 1) / phi

plt.plot(phi, uztilde_chi1, label=r'$\tilde{u}_z(\chi=1)$')
plt.plot(phi, uztilde_chi1_inverse, label=r'$\tilde{u}_z^{-1}(\chi=1)$', linestyle='--')
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\tilde{u}_z$')
plt.title(r'Normalized Axial Velocity Profile of a $\chi=1$ Bennett vortex')

plt.legend()
plt.show()