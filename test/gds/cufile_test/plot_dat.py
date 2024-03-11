import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = np.fromfile('ics.dat', dtype=np.float32)

plt.plot(data)
plt.show()