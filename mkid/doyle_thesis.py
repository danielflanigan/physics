from __future__ import division

import numpy as np
from numpy import pi
from scipy.constants import e, m_e, mu_0, k

# The coherence length \xi_0 for aluminum in meters.
xi_aluminum = 1600e-6

# Chapter 3

# 3.1 The conductivity of normal metals

def sigma_n(sigma_0, omega, tau):
    """
    Equation 3.1: Return the complex AC conductivity of a normal metal
    with DC conductivity \sigma_0 and scattering time \tau at angular
    frequency \omega. Note that the sign of the complex term given in
    (3.1) is inconsistent with the rest of the text and is corrected
    in this function. This sign depends on the sign convention used
    for the phase of the signal, and must reflect the physical fact
    that at high frequency the current lags the electric field by \pi/2
    radians.
    """
    return sigma_0 / (1 + 1j * omega * tau)

def sigma_0(n_n, tau):
    """
    Return the Drude model DC conductivity of a normal metal.
    """
    return n_n * e**2 * tau / m_e

def sigma_s(n_s, omega):
    """
    Equation 3.5: Return the complex conductivity of an electron gas
    in the limit of infinite scattering time.
    """
    return -1j * n_s * e**2 / (omega * m_e)

def lambda_L0(n_s):
    """
    Equation 3.10: Return the London penetration depth \lambda_L.
    """
    return (m_e / (mu_0 * n_s * e**2))**(1/2)

# 3.3 The two fluid model

def superconducting_electron_fraction(T, T_c):
    """
    Equation 3.11a: Return the density of electrons condensed into the
    superconducting ground state divided by the total electron
    density.
    """
    return 1 - (T / T_c)**4

def lambda_L(n_s, T, T_c):
    """
    Equation 3.13: Return the London penetration depth \lambda_L as a
    function of temperature.
    """
    return lambda_L0(n_s) * (1 - (T / T_c)**4)**(-1/2)

# 3.4 Internal inductance from the London model

def kinetic_inductance(lambda_L, w, t):
    """
    Equation 3.19: Return the kinetic inductance in H/m; lambda_L is
    the London penetration depth, t is the film thickness, and w is
    the film width. Note that Doyle's 'cosec' means 'csch'.
    """
    x = t / (2 * lambda_L)
    return mu_0 * lambda_L / (4 * w) * (np.tanh(x)**-1 + x * np.sinh(x)**-2)

def magnetic_inductance(lambda_L, w, t):
    """
    Equation 3.20: Return the magnetic inductance in H/m; lambda_L is
    the London penetration depth, t is the film thickness, and w is
    the film width. Note that Doyle's 'cosec' means 'csch'.
    """
    x = t / (2 * lambda_L)
    return mu_0 * lambda_L / (4 * w) * (np.tanh(x)**-1 - x * np.sinh(x)**-2)

# 3.5 The microscopic theory of superconductivity
# Code up 3.22, 3.23, and 3.24 if useful.

# 3.6 The density of states and quasi-particle excitations
# Code up 3.25 and 3.26 if useful.

# 3.7 Mattis-Bardeen theory
# Code up 3.27-3.31 if useful.

# These require the gap as function of temperature, and should be
# implemented in terms of frequency, not angular frequency.
def real_conductivity(T, f):
    """
    Equation 3.32: Return the ratio of the real part of the
    approximate Mattis-Bardeen conductivity to the normal state
    conductivity.
    """
    pass


def imaginary_conductivity(T, f):
    """
    Equation 3.33: Return the ratio of the imaginary part of the
    approximate Mattis-Bardeen conductivity to the normal state
    conductivity.
    """
    pass

# 3.8 Impedance of a superconducting strip
# Code up 3.37 and 3.38 if useful.

# Think about whether to implement this in terms of derived quantities
# like L_k and the conductivity or to calculate them for each
# frequency - probably better since the conductivity is dependent.
def strip_impedance():
    """
    Equation 3.39: Return the complex impedance of a superconducting strip.
    """
    pass

# 3.9 The quasi-particle equilibrium state

def quasiparticle_density(N_0, T, Delta_0):
    """
    Equation 3.40: Return the approximate number of quasiparticles per
    unit volume, valid for T \ll T_c. The quantities N_0 and Delta_0
    are the values at T = 0. The units don't make sense here: in 3.22
    and 3.23, N_0 has dimensions 1/energy, so this equation should be
    dimensionless.
    """
    return 2 * N_0 * (2 * pi * k * T * Delta_0)**1/2 * np.exp(-Delta_0 / (k * T))

def quasiparticle_lifetime(tau_0, T_c, T, Delta_0):
    """
    From Equation 3.41: Return the quasiparticle lifetime \tau. Using
    values from Kaplan gives lifetimes much smaller than those
    observed in typical aluminum films.
    """
    return tau_0 / pi**1/2 * (k * T_c / (2 * Delta_0))**5/2 * (T_c / T)**1/2 * np.exp(Delta_0 / k * T)

def excess_quasiparticle_number(eta, P, tau_qp, Delta):
    """
    Equation 3.42: Return the number of excess quasiparticles under
    power load P absorbed with efficiency \eta.
    """
    return eta * P * tau_qp / Delta

