from module import C_BT, p0_CUBIC, eta_SPITZER, BMAX, kappa_PERP, TAU_CONFINEMENT, P_ELECTRICAL

rp = 1e3 # pinch radius [m]
uz0 = 5e0 # root of the axial velocity [m/s]
Tp = 5e2 # edge plasma temperature [degK]
n0 = 1e6 # plasma number density [m^-3]

L = 500e3 # pinch length [m]

Cbt = C_BT(n0, uz0, rp, Tp)
p0 = p0_CUBIC(n0, uz0, rp, Tp)
eta_e = eta_SPITZER(Tp)
Bmax = BMAX(n0, uz0, rp, Tp)
kappa_perp = kappa_PERP(n0, uz0, rp, Tp)
tau_conf = TAU_CONFINEMENT(n0, Tp, rp, p0, Bmax)
p_electrical = P_ELECTRICAL(n0, uz0, rp, Tp)

print(f"The cubic Bennett parameter Cbt is {Cbt:.2e} meters")
print(f"The axial pressure p0 is {p0:.2e} Pascals")
print(f"The energy confinement time is {tau_conf:.2e} seconds")
print(f"The electrical power is {p_electrical:.2e} Watts")