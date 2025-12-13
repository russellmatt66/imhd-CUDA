import numpy as np
import matplotlib.pyplot as plt

# phi = np.logspace(-6, 6, 1000)  # Dimensionless plasma radius
phi = np.linspace(0.0, 2.0, 1000)  # Dimensionless plasma radius

chi = 1 / (np.log(phi + 1) - np.log(phi)) # optimal shear parameter

plt.figure(figsize=(8, 6))
plt.plot(phi, chi, label='Optimal Shear Parameter ($\\chi$)')
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('Dimensionless Plasma Radius ($\\phi$)')
plt.ylabel('Optimal Shear Parameter ($\\chi$)')
plt.title('Optimal Shear of a Cubic Bennett Vortex')
plt.axhline(y=2, color='red', linestyle='--', linewidth=0.5, label='Pure-flow vortex')

plt.legend()

plt.show()