"""
This module contains equations from Khalil et al. 'An analysis method
for asymmetric resonator transmission applied to superconducting
devices.' Journal of Applied Physics 2012.
"""

from __future__ import division

import numpy as np

def one_plus_epsilon(f, Z_in, Z_out, C_c, L_1, M, L):
    """
    This implements the definition of the quantity
    1 + \hat{\epsilon},
    defined just after Equation 1; the paper assumes that
    |\hat{\epsilon}| \ll 1,
    which sets some constraints on the values of the circuit
    parameters.
    """
    omega = 2 * np.pi * f
    Z_in_prime = Z_in + 1j * omega * L_1 - 1j * omega * M**2 / L
    return 2 / (1 + (1j * omega * C_c + 1 / Z_out)) / Z_in_prime

def S_21_11(f, A, f_0, Q, Q_e):
    """
    This implements equation 11. The parameter A is a complex
    prefactor:
    A = 1 + \hat{\epsilon}.
    """
    return A * (1 - (Q * Q_e**-1 /
                     (1 + 2j * Q * (f - f_0) / f_0)))

