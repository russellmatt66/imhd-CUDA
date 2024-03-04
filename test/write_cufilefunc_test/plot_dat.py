import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

buf_1 = np.fromfile('buf1.dat', dtype=np.float32)
buf_2 = np.fromfile('buf2.dat', dtype=np.float32)
buf_3 = np.fromfile('buf3.dat', dtype=np.float32)

plt.plot(buf_1, label='buf1')
plt.plot(buf_2, label='buf2')
plt.plot(buf_3, label='buf3')
# plt.legend() # slow with lots of data

plt.show()