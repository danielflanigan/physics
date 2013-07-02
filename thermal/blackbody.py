from __future__ import division

import numpy as np
from numpy import pi
from scipy.constants import c, k, h
from scipy.special import zeta

# All units are SI.

def energy_peak_frequency(T):
    """
    Return the peak frequency in Hz of the energy density (or radiated
    power) per unit frequency of a blackbody at kelvin temperature T..
    """
    pass

def energy_density(T):
    """
    Return the photon energy density in J m^{-3} of blackbody
    radiation at kelvin temperature T.
    """
    return ((8 * pi**5 * k**4 * T**4) /
            (15 * h**3 * c**3))

def radiated_power_per_area(T):
    """
    Return the radiated power per unit emitting area
    J_u = \sigma T^4,
    where \sigma is the Stefan-Boltzmann constant, in W K^{-4} m^{-2}.
    """
    return c / 4 * energy_density(T)

def radiated_power_per_throughput(T):
    """
    Return the radiated power per unit emitting throughput
    P = \sigma T^4 / \pi,
    where \sigma is the Stefan-Boltzmann constant, in W K^{-4} m^{-2}
    sr^{-1}. Integrating this function times \cos(\theta) to account
    for the projected area over 2 \pi steradians gives the above power
    per unit emitting area.
    """
    return radiated_power_per_area(T) / pi

def energy_spectral_density(T, f):
    """
    Return the energy spectral density u_f in J m^{-3} Hz^{-1} of a
    blackbody at kelvin temperature T. This is the energy per unit
    volume per unit frequency.
    """
    return ((8 * pi * h * f**3) /
            (c**3 * (np.exp(h * f / (k * T)) - 1)))

# From NRAO site; this is compatible with the above.
def spectral_brightness(T, f):
    """
    Return the spectral brightness B_\nu in W m^{-2} Hz^{-1} sr^{-1}
    of a blackbody at kelvin temperature T. This is the power per unit
    frequency per unit throughput, where throughput has dimensions of
    area times solid angle. Integrating this over 2 pi steradians
    gives the Stefan-Boltzmann law for power radiated per unit
    emitting area.
    """
    return ((2 * h * f**3) /
            (c**2 * (np.exp(h * f / (k * T)) - 1)))


def number_peak_frequency(T):
    """
    Return the peak frequency in Hz of the number density (or radiated
    power) per unit frequency of a blackbody at kelvin temperature T..
    """
    pass

# Check.
def number_density(T):
    """
    Return the photon number density in m^{-3} of a blackbody at
    kelvin temperature T. This is the number of photons per unit
    volume.
    """
    return 16 * pi * zeta(3, 1) * (k * T / (h * c))**2

def number_spectral_density(T, f):
    """
    Return the photon number spectral density in m^{-3} Hz^{-1} of
    blackbody radiation at kelvin temperature T. This is the number of
    photons per unit volume per unit frequency.
    """
    return (8 * pi * f**2 /
            (c**3 * (np.exp(h * f / (k * T)) - 1)))

def number_spectral_brightness(T, f):
    """
    Return the number spectral brightness N_\nu in s^{-1} m^{-2}
    Hz^{-1} sr^{-1} of a blackbody at kelvin temperature T. This is
    the number of radiated photons per second per unit frequency per
    unit throughput, where throughput has dimensions of area times
    solid angle.
    """
   
# Check this. Replace with pi times number_spectral_brightness.
def radiated_number(T):
    """
    Return the number of photons radiated per unit emitting area in s^{-1} m^{-2}.
    """
    return c / 4 * number_density_K(T)
