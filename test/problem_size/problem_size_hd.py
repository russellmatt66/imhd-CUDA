# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Side lengths of simulation domain
Nx = np.linspace(256,1024,49)
Ny = np.linspace(256,1024,49)
Nz = np.linspace(256,1024,49)
# These are exploratory
# Nx = [2**i for i in range(5,11)]
# Ny = [2**j for j in range(5,11)]
# Nz = [2**k for k in range(5,11)]

def DV(nx: int, ny: int, nz: int) -> np.float64: # Assumes intermediate variables are also being stored in device memory
    # 4.0 = `number of bytes per float`
    # 10.0 = `number of rank-3 tensors (8 - od fluid variables, 1 - future fluid var, 1 - intermediate vars for future fluid var)`
    # nx = `x-dimension of the rank-3 tensor (number of floats)`
    # ny = `y-dimension of the rank-3 tensor (number of floats)`
    # nz = `z-dimension of the rank-3 tensor (number of floats)`
    return (4.0 * 10.0 * nx * ny * nz) / (1024.0**3) # GB

problem_sizes = []
data_volumes = []
for nx in Nx:
    for ny in Ny:
        for nz in Nz:
            problem_sizes.append((nx, ny, nz))
            data_volumes.append(DV(nx, ny, nz))

# print(data_volumes)

ps_df = pd.DataFrame(problem_sizes,columns=['Nx','Ny','Nz'])
print(ps_df.shape)
# print(ps_df.head())

dv_df = pd.DataFrame(data_volumes,columns=['GB'])
print(dv_df.shape)
# print(dv_df.head())

# df = pd.concat([ps_df, dv_df])
df = pd.merge(ps_df, dv_df, left_index=True, right_index=True)
print(df.shape)
# print(df.head())

too_large = 5.0 # GB
tl_df = df[df['GB'] >= too_large]
print(tl_df.shape)
# print(tl_df.head())
impossible_sizes = tl_df.index.values.tolist()

possible_df = df.drop(index=impossible_sizes)
print(possible_df.shape)
print(possible_df.head())

''' TODO: Find the largest possible problem sizes from the above '''
possible_df = possible_df.sort_values(by=['Nz'], ascending=False)
possible_df[(possible_df['GB'] >= 4.5) & (possible_df['Nx'] == possible_df['Ny'])].to_csv('./possible_hd.csv',index=False)

# tl_df = dv_df.loc['']

# Visualization might be nice for writeup
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# i = 0
# for p in problem_sizes:
#     print(p[0])
#     print(p[1])
#     problem_size = np.array([p[0], p[1]])
#     # ax.scatter(problem_size, data_volumes[i] / p[2])
#     i += 1

# xx, yy = np.meshgrid(Nx, Ny)
# print(xx)
# print()
# print(yy)

# z = (1.0 / 96.0) * (1.0 / (xx * yy)) * (5.0 * 1024.0**3)

# ax.plot_surface(xx, yy, z, alpha=0.5, cmap='viridis')

# ax.set_xlabel('Nx')
# ax.set_ylabel('Ny')
# ax.set_zlabel('Nz')

# plt.show()