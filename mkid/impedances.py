from __future__ import division
import numpy as np

def omega_0(L, C):
    return (L * C)**(-1/2)

def omega_RL(R, L):
    return R / L

def omega_hat(L, R, C):
    return (omega_0(L, C)**2 - omega_RL(R, L)**2)**(1/2)

def Z(omega, L, R, C):
    w = omega
    w_0 = omega_0(L, C)
    w_RL = omega_RL(R, L)
    return R * ((1 + 1j * (w / w_RL - w * w_RL / w_0**2 - w**3 / (w_0**2 * w_RL))) /
                 ((1 - w**2 / w_0**2)**2 + w**2 * w_RL**2 / w_0**4))

def Z_LRC(omega, L, R, C):
    w = omega
    return ((R + 1j * w * (L - R**2 * C - w**2 * L**2 * C)) /
            ((1 - w**2 * L * C)**2 + w**2 * R**2 * C**2))

def series(*impedances):
    return np.sum(impedances, axis=0)

def parallel(*impedances):
    return np.sum([i**-1 for i in impedances], axis=0)**-1

L_skip = 88.73e-9
C_skip = 28.5476117554e-12
C_c_skip = 0.255337601922e-12

def Z_R(R, f):
    return R * np.ones_like(f)

def Z_L(L, f):
    return 2j * np.pi * f * L

def Z_C(C, f):
    return (2j * np.pi * f * C)**-1

# G is a conductance.
def Z_G(G, f):
    return G**-1 * np.ones_like(f)

def Z_parallel(f, R, L, C):
    return parallel(Z_R(R, f), Z_L(L, f), Z_C(C, f))

def Z_series(f, R, L, C):
    return series(Z_R(R, f), Z_L(L, f), Z_C(C, f))

def Z_RLpC(f, R, L, C):
    return parallel(series(Z_R(R, f), Z_L(L, f)), Z_C(C, f))

def Z_RLpCG(f, R, L, C, G):
    return parallel(series(Z_R(R, f), Z_L(L, f)), parallel(Z_C(C, f), Z_G(G, f)))

def S_21(Z_in, Z_out, Z_couple, Z_resonator):
    Z_p = parallel(Z_out, series(Z_couple, Z_resonator))
    return (Z_p /
            series(Z_in, Z_p))
            
