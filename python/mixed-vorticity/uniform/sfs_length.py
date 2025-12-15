import matplotlib.pyplot as plt
import numpy as np

phi = np.logspace(-4, 2, 1000)

L_tilde = ((phi + 1) / phi)**(1.5)*(phi**3 + 3*phi**2 -6*phi*(1 + np.log(phi+1)) + 6*np.log(phi+1))

plt.plot(phi, L_tilde)
plt.xlabel('$\\phi$')
plt.ylabel('$\\tilde{L}(\\phi)$')
plt.title('Normalized length threshold for a uniform cubic vortex')
# plt.xscale('log')
# plt.yscale('log')

# plt.ylim((1e-6, 10**6))

plt.show()