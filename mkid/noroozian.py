from __future__ import division
import numpy as np
from numpy import pi
from scipy.constants import e, h, k
from scipy.special import i0, k0

# Aluminum electron-phonon interaction time, in seconds.
tau_0_Al = 438e-9

# The Boltzmann constant in eV K^-1.
k_eV = k / e

# From Noroozian's thesis, in um^-3 eV^-1.
N_0_Al_um_eV = 1.72e10

# Aluminum single-spin density of electron states at the Fermi energy, in m^-3 J^-1.
N_0_Al = N_0_Al_um_eV * 1e6**3 / e

def BCS_Delta_0(T_c):
    return 3.52 * k * T_c / 2

def equation_2_3(T, Delta_0, N_0=N_0_Al):
    return 2 * N_0 * (2 * pi * k * T * Delta_0)**(1/2) * np.exp(-Delta_0 / (k * T))

# Write a version of 2.7 and 2.8 that cancels N_0; the natural scale
# for n_{qp} is N_0 \Delta_0.

# It's important to keep all numbers as close to 1 as possible,
# especially in calculating xi.
def equation_2_7(f, T, Delta_0, N_0, n_qp):
    return pi / 2 * (S_1(f, T, Delta_0) * n_qp /
                     (N_0 * h * f))
    
def equation_2_8(f, T, Delta_0, N_0, n_qp):
    return pi * Delta_0 / (h * f) * (1 - (S_2(f, T, Delta_0) * n_qp /
                                          (2 * N_0 * Delta_0)))

def equation_2_9(f, T):
    return (h / k) * f / (2 * T)

def equation_2_20(f, T, Delta_0):
    xi = equation_2_9(f, T)
    S_1 = (2 / pi *
           (2 * Delta_0 / (pi * k * T))**(1/2) *
           (np.exp(xi) - np.exp(-xi)) / 2 *
           k0(xi))
    return S_1

def equation_2_21(f, T, Delta_0):
    xi = equation_2_9(f, T)
    S_2 = 1 + ((2 * Delta_0 / (pi * k * T))**(1/2) *
               np.exp(-xi) *
               i0(xi))
    return S_2

S_1 = np.vectorize(equation_2_20)
S_2 = np.vectorize(equation_2_21)

def beta(f, T, Delta_0):
    return (S_2(f, T, Delta_0) /
            S_1(f, T, Delta_0))

def quasiparticle_lifetime(tau_max):
    """
    Return the quasiparticle lifetime. This is a modified form of equation 2.23, with $R 
    """
    pass
                           
def recombination_constant(Delta, T_c, N_0=N_0_Al, tau_0=tau_0_Al):
    """
    Equation 2.24: return the recombination constant R. Defaults are
    for aluminum. Note that this equation doesn't seem to match
    Jonas's Equation 99. It also seems not to produce numbers that
    agree with the given values of \tau_{max} and n_* given there and
    here. Come back to this.
    """
    return (Delta / k)**2 * 2 / (N_0 * tau_0) / k / T_c**3

def equation_2_31(Gamma_e, V, tau_max, n_star):
    """
    Return the quasiparticle density n_{qp}. The units of n_{qp} are
    the same as the units of n_star. V and n_star must have reciprocal
    units.
    """
    return n_star * ((1 + 2 * Gamma_e * tau_max / V / n_star)**(1/2) - 1)

def equation_2_32(Gamma_e, V, tau_max, n_star):
    """
    Return the quasiparticle lifetime \tau_{qp}.
    """
    return tau_max * (1 + 2 * Gamma_e * tau_max / V / n_star)**(-1/2)
