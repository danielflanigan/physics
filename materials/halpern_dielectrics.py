from __future__ import division

import numpy as np
from scipy.constants import c

"""
M. Halpern, et al. 'Far infrared transmission of dielectrics at
cryogenic and room temperatures: glass, Fluorogold, Eccosorb, Stycast,
and various plastics.' Applied Optics, 1986.
"""

# This paper uses inverse centimeters (ICM) internally.
# Convention:
# f is frequency in Hz;
# nu is frequency in inverse centimeters (ICM).
# d is sample thickness in centimeters.
# All other quantities are SI.

# Non-lossy admittance (1)
# This is the \alpha = 0 limit of (6), and is not used elsewhere.
def Y_ideal(nu, n, d):
    x = 2 * np.pi * n * nu * d
    return n * ((np.cos(x) + 1j * n * np.sin(x)) /
                (n * np.cos(x) + 1j * np.sin(x)))

# Field reflection coefficient (2), calculated using the lossy
# admittance (6), not the ideal admittance (1).
def rho(nu, n, a, b, d):
    return ((1 - Y(nu, n, a, b, d)) /
            (1 + Y(nu, n, a, b, d)))

# Lossy admittance (6)
def Y(nu, n, a, b, d):
    n_p = n_prime(nu, n, a, b)
    x = gamma(nu, n, a , b) * d
    return 1 / n_p * ((n_p * np.cosh(x) + np.sinh(x)) /
                          (np.cosh(x) + n_p * np.sinh(x)))

# Note: (7) and (8) are calculated using the rightmost equalities,
# which assume that (\sigma / \omega \epsilon) \ll 1 and \mu = 1. This
# is necessary because we don't know the conductivity \sigma.

# Complex wave vector (7), in the same units as \nu, which should be ICM.
def gamma(nu, n, a, b):
    return alpha(nu, a, b) / 2 + 2j * np.pi * n * nu

# Bulk impedance (8)
def n_prime(nu, n, a, b):
    return n**-1 * (1 + alpha(nu, a, b) / (2j * np.pi * n * nu))**(-1/2)

# Complex dielectric constant, calculated using the same assumptions
# as for (7) and (8) above.
def epsilon_c(nu, n, a, b):
    return n_prime(nu, n, a, b)**-2

# Power reflection coefficient (9)
def R(nu, n, a, b, d):
    return abs(rho(nu, n, a, b, d))**2

# Power transmission coefficient (9)
# The fractional transmitted power is the fraction not reflected times
# the fraction not absorbed.
def T(nu, n, a, b, d):
    return (1 - R(nu, n, a, b, d)) * (1 - A(nu, a, b, d))

# Power absorption: this is the fraction of power that enters the
# material that is absorbed. Use this to calculate transmission
# assuming an AR coating.
def A(nu, a, b, d):
    return 1 - np.exp(-alpha(nu, a, b) * d)

# This is the absorption coefficient alpha, defined below (9),
# expressed in cm^{-1}
def alpha(nu, a, b):
    return a * nu**b


def Hz_to_ICM(f):
    return f / (100 * c)

def ICM_to_Hz(nu):
    return 100 * c * nu
