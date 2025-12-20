import numpy as np 

k_B = 1.380649e-23  # J/K
mu0 = 4e-7 * np.pi  # H/m
e = 1.602176634e-19  # C

def C_BT(n0: float, uz0: float, rp: float, Tp: float) -> float:
    """
    Tp in Kelvin
    uz0 in m/s
    rp in meters
    n0 in m^-3
    """
    return (mu0 * n0 * e**2 * uz0**2 * rp**3) / (16 * k_B * Tp)

def uz0(Tp: float, n0: float, rp: float, uzrp: float) -> float:
    """
    Return the uz0 needed to achieve uzrp at rp for a pure-flow profile
    Tp in Kelvin
    uzrp in m/s
    rp in meters
    n0 in m^-3
    """
    A = mu0 * e**2 * n0 * rp**2 * uzrp
    B = -16 * k_B * Tp
    C = uzrp * 16 * k_B * Tp
    uz0_neg = (-B - np.sqrt(B**2 - 4 * A * C)) / (2 * A)
    uz0_pos = (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A)
    
    print(uz0_neg) 
    print(uz0_pos)
    
    return 

def uz_pureflow(r: np.ndarray, n0: float, uz0: float, rp: float, Tp: float) -> np.ndarray:
    """
    Return a pure-flow (\chi = 2) uz profile
    """
    Cbt = C_BT(n0, uz0, rp, Tp)
    return uz0 * r**2 / (Cbt + r)**2
