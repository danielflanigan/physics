from __future__ import division
import numpy as np
from scipy.constants import epsilon_0, hbar, k as k_B, pi
from scipy.special import ellipk
import lmfit

def interdigitated_capacitance(epsilon_r, area, width, gap):
    """
    Return the capacitance C of an interdigitated capacitor with the given surface area, tine width, and gap between
    tines; epsilon_r is the dielectric constant of the substrate. The formula is given in Jonas's MKID design memo. Note
    that all lengths and areas must use SI units (m and m^2) for the return value to be in farads.
    """
    pitch = width + gap
    k = np.tan(pi * width / (4 * pitch))**2
    K = ellipk(k**2)
    Kp = ellipk(1 - k**2)
    C = epsilon_0 * (1 + epsilon_r) * (area / pitch) * (K / Kp)
    return C

# TODO: finish this.
def find_interdigital_capacitance(capacitance, params):
    def parameterized_interdigitated_capacitance(params):
        base = params['base'].value
        height = params['height'].value

    return lmfit.minimize(lambda params: capacitance - int)
