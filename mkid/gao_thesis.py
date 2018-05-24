from __future__ import division, print_function

import numpy as np
from scipy.constants import epsilon_0, mu_0, pi
from scipy.special import ellipk

epsilon_silicon_JG = 12  # This makes the capacitances in Table 3.1 come out correctly to the four digits he gives


# From Equations 3.16 and 3.17
# The half-capacitance per unit length of a zero-thickness CPW, divided by epsilon_0
def C_zero_half(k, epsilon):
    return 2 * epsilon * ellipk(m=k**2) / ellipk(m=1 - k**2)


# From Equation 3.20
# The total inductance per unit length of a zero-thickness CPW, divided by mu_0
def L_zero(k):
    return ellipk(m=1 - k**2) / (4 * ellipk(m=k**2))


# Both from Equation 3.27
# The lengths a, b, and t must have the same units.
def u1(a, b, t):
    d = 2 * t / pi
    return a + d / 2 * (1 + 3 * np.log(2) - np.log(d / a) + np.log((b - a) / (a + b)))


def u2(a, b, t):
    d = 2 * t / pi
    return b + d / 2 * (-1 - 3 * np.log(2) + np.log(d / b) - np.log((b - a) / (a + b)))


# From Equation 3.28
# The half-capacitance per unit length of a CPW with half-thickness t, divided by epsilon_0
def C_half(a, b, t, epsilon):
    kt = u1(a, b, t) / u2(a, b, t)
    return 2 * epsilon * ellipk(m=kt**2) / ellipk(m=1 - kt**2)


# From Equation 3.29
# The half inductance per unit length of a CPW with half-thickness t, divided by mu_0;
# relative permeability 1 assumed.
def L_half(a, b, t):
    kt = u1(a, b, t) / u2(a, b, t)
    return ellipk(m=1 - kt**2) / (2 * ellipk(m=kt**2))


# From Equation 3.30; JG says this is the best approximation for a CPW on a dielectric substrate in vacuum.
# The half-capacitance per unit length of a thickness t CPW in vacuum plus
# the half-capacitance of a zero-thickness CPW on the substrate.
def C_best(a, b, t, epsilon_substrate):
    return C_half(a=a, b=b, t=t, epsilon=1) + C_zero_half(k=a / b, epsilon=epsilon_substrate)


# From Equation 3.31
# The total inductance per unit length of a CPW with thickness t, calculated as
# the parallel inductance of two t / 2 thickness CPWs in parallel, divided by mu_0;
# relative permeability 1 assumed.
def L_total(a, b, t):
    return 1 / 2 * L_half(a=a, b=b, t=t / 2)


# From Equation 3.49
# ToDo: according to JG, these formulas were copied incorrectly from Collin, and they need to be re-derived.
def g_total(a, b, t):
    return g_center(a=a, b=b, t=t) + g_ground(a=a, b=b, t=t)


def g_center(a, b, t):
    k = a / b
    K = ellipk(m=k**2)
    return 1 / (4 * a * K**2 * (1 - k**2)) * (pi + np.log(4 * pi * a / t) - k * np.log((1 + k) / (1 - k)))


def g_ground(a, b, t):
    k = a / b
    K = ellipk(m=k**2)
    return k / (4 * a * K**2 * (1 - k**2)) * (pi + np.log(4 * pi * b / t) - (1 / k) * np.log((1 + k) / (1 - k)))