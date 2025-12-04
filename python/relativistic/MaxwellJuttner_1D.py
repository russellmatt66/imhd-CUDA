import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from scipy.special import kn # Need Bessel functions of the second kind

num_T = 1000
T = np.linspace(0.025, 10.0, num=num_T) # [MeV]

theta = 0.511 / T # relativistic inverse temperature

# Macroscopic flow velocity (normalized by c) 
u = np.exp(-theta) / (theta * kn(1, theta))

# Save to CSV
df = pd.DataFrame({
    'Theta' : theta,
    'T' : T,
    'u / c' : u
}
)

df.to_csv("MJ1D_u.csv", index=True)

# PLOT
plt.loglog(T, u)
plt.xlim([T[0], T[-1]])
plt.ylim([u[0], u[-1]])

selection_freq = int(num_T / 5) # every `selection_freq`

# xticks = T[::int(selection_freq / 5)]
# yticks = u[::selection_freq]

# plt.xticks(xticks)
# plt.yticks(yticks)

plt.xlabel('Temperature (MeV)')
plt.ylabel('$\\frac{u}{c}$')

plt.show()