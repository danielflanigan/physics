from __future__ import division

import numpy as np
from numpy import pi
from scipy.constants import c, k, hbar
from scipy.special import zeta

# All units are SI.

"""
This module contains blackbody functions from Kittel with temperature
in energy units and frequency in radians.
"""

def energy_density(tau):
    """
    Return the photon energy density of blackbody radiation at
    fundamental temperature \tau. This is the energy per unit volume.
    """
    return ((pi**2 * tau**4) /
            (15 * hbar**3 * c**3))

def energy_spectral_density(tau, omega):
    """
    Return the energy spectral density u_{\omega} in J m^{-3} rad^{-1}
    of a blackbody at fundamental temperature \tau. This is the energy
    per unit volume per unit angular frequency.
    """
    return ((hbar * omega**3) /
            (pi**2 * c**3 * (np.exp(hbar * omega / tau) - 1)))

def number_density(tau):
    """
    Return the photon number density in m^{-3} of a blackbody at
    fundamental temperature \tau. This is the number of photons per
    unit volume.
    """
    return ((2 * zeta(3, 1) * tau**3) /
            (pi**2 * hbar**3 * c**3))

def number_spectral_density(tau, omega):
    """
    Return the photon number spectral density in m^{-3} rad^{-1} of
    blackbody radiation at fundamental temperature \tau. This is the
    number of photons per unit volume per unit angular frequency.
    """
    return (omega**2 /
            (pi**2 * c**3 * (np.exp(hbar * omega / tau) - 1)))

