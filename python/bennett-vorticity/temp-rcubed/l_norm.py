import matplotlib.pyplot as plt
import numpy as np

phi = np.logspace(-8, 8, int(10**5))
# phi = np.linspace(0.00000001, 10000000, int(10**6))

L_tilde = (1 + phi)**2 / phi**2 * (phi**3 - 3*phi**2 -6*phi + 6*phi*np.log(phi + 1) + 6*np.log(phi + 1))

plt.plot(phi, L_tilde)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\\phi$')
plt.ylabel('$\\tilde{L}$')
plt.title('Plot of $\\tilde{L}$ vs. $\\phi$')
plt.show()