import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uz_module as uz_mod

from sklearn.metrics import mean_squared_error

uz_data = pd.read_csv('../../../experimental_data/zap_2001/zap2001_uz_fig5.csv')

uz_data.columns = ['Radius (mm)', 'uz (10^{4} m / s)']

# print(uz_data)

fullpinch = uz_data.copy()
fullpinch['Radius (mm)'] = fullpinch['Radius (mm)'] - fullpinch['Radius (mm)'].iloc[0]  # center at 0

print(fullpinch)

pinch_neghalf = uz_data[uz_data['Radius (mm)'] < 0]
pinch_poshalf = uz_data[uz_data['Radius (mm)'] >= 0]

pinch_neghalf = pinch_neghalf.iloc[::-1].reset_index(drop=True) # radius was really x
pinch_neghalf['Radius (mm)'] = -pinch_neghalf['Radius (mm)']

pinch_poshalf = pinch_poshalf.reset_index(drop=True)

# print(pinch_neghalf)
# print(pinch_poshalf)

rp = 10e-3  # m
n0 = 9e22  # m^-3
Tp = 2.0e4 * 11604.5 # keV to K
# uzrp = 10e4  # m/s
uz0 = 8.0e4  # m/s

# uz0 = 1e5 # m/s
# uz0 = uz_mod.uz0(Tp, n0, rp, uzrp)

Cbt = uz_mod.C_BT(n0, uz0, rp, Tp)
print(f"C_BT = {Cbt} m")

# print(f"uz0 = {uz0} m/s")

# r = np.linspace(0, pinch_poshalf['Radius (mm)'].values.max() * 1e-3, 1000)
r = np.linspace(0, fullpinch['Radius (mm)'].values.max() * 1e-3, 1000)
uz = uz_mod.uz_pureflow(r, n0, uz0, rp, Tp) + 4.5e4  # m/s offset
# uz = uz_mod.uz_pureflow(r, n0, uz0, rp, Tp) # no offset

"""
Half-chord plots
"""
plt.plot(r, uz * 1e-4, label='Pure-flow uz', color='blue')
plt.plot(pinch_poshalf['Radius (mm)'].values * 1e-3, pinch_poshalf['uz (10^{4} m / s)'].values, 'o', label='Zap 2001 data', color='red')
plt.xlabel('Radius (mm)')
plt.ylabel('uz ($10^{4}$ m/s)')
plt.title(f'Pure-flow cubic vortex fit of Zap 2001 positive half-chord $u_{{z}}$ \n $n_{{0}}$ = {n0} $m^{-3}$, $T_{{p}}$ = {Tp/11604.5/1e3} keV, $u_{{z0}} = {uz0 * 1e-3} \\frac{{km}}{{s}}$, $r_{{p}} = {rp*1e3}$ mm')

# plt.plot(r, uz * 1e-4, label='Pure-flow uz', color='blue')
# plt.plot(pinch_neghalf['Radius (mm)'].values * 1e-3, pinch_neghalf['uz (10^{4} m / s)'].values, 'o', label='Zap 2001 data', color='red')
# plt.title(f'Pure-flow cubic vortex fit of Zap 2001 negative half-chord $u_{{z}}$ \n $n_{{0}}$ = {n0} $m^{-3}$, $T_{{p}}$ = {Tp/11604.5/1e3} keV, $u_{{z0}} = {uz0 * 1e-3} \\frac{{km}}{{s}}$, $r_{{p}} = {rp*1e3}$ mm')
# plt.xlabel('Radius (mm)')
# plt.ylabel('uz ($10^{4}$ m/s)')

"""
Full-chord plots
"""
# fullpinch = pd.concat([pinch_neghalf, pinch_poshalf], ignore_index=True)

# plt.plot(r, uz * 1e-4, label='Pure-flow uz', color='blue')
# plt.plot(fullpinch['Radius (mm)'].values * 1e-3, fullpinch['uz (10^{4} m / s)'].values, 'o', label='Zap 2001 data', color='red')
# plt.xlabel('Radius (mm)')
# plt.ylabel('uz ($10^{4}$ m/s)')
# plt.title(f'Pure-flow cubic vortex fit of Zap 2001 $u_{{z}}$ \n $n_{{0}}$ = {n0} $m^{-3}$, $T_{{p}}$ = {Tp/11604.5/1e3} keV, $u_{{z0}} = {uz0 * 1e-3} \\frac{{km}}{{s}}$, $r_{{p}} = {rp*1e3}$ mm')

# Calculate RMSE
# Positive half-chord
uz_fit = uz_mod.uz_pureflow(pinch_poshalf['Radius (mm)'].values * 1e-3, n0, uz0, rp, Tp) + 4.5e4
mse = mean_squared_error(pinch_poshalf['uz (10^{4} m / s)'].values * 1e4, uz_fit)
rmse = np.sqrt(mse)
rrmse = rmse / np.mean(pinch_poshalf['uz (10^{4} m / s)'].values * 1e4)
mse = np.mean((uz_fit - pinch_poshalf['uz (10^{4} m / s)'].values * 1e4) ** 2) / np.mean(pinch_poshalf['uz (10^{4} m / s)'].values * 1e4)

# Full chord
# uz_fit = uz_mod.uz_pureflow(fullpinch['Radius (mm)'].values * 1e-3, n0, uz0, rp, Tp) + 4.5e4
# mse = mean_squared_error(fullpinch['uz (10^{4} m / s)'].values * 1e4, uz_fit)
# rmse = np.sqrt(mse)
# rrmse = rmse / np.mean(fullpinch['uz (10^{4} m / s)'].values * 1e4)

print(f'Mean Squared Error (MSE) of fit: {mse:.2e} (m/s)^2')
print(f'Root Mean Squared Error (RMSE) of fit: {rmse:.2e} m/s')
print(f'Relative Root Mean Squared Error (RRMSE) of fit: {rrmse:.2%}')

plt.show()