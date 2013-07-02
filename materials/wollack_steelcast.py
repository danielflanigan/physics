from __future__ import division

import numpy as np
from np import pi
from scipy.constants import c

"""
E. J. Wollack, et al. 'Electromagnetic and Thermal Properties of a Conductively Loaded Epoxy.' Int J Infrared Milli Waves, 2007.
"""

# The measured data between 3 and 90 GHz, expressed in Hz. Two data
# points taken at higher frequencies are plotted in Figure 5 but the
# values are not given in the paper.
f_measured = np.array([3.2, 8.3, 10, 22, 33, 90]) * 1e9
epsilon_r_measured = np.array([14.2 + 1.3j, 13.0 + 2.3j, 12.5 + 2.5j, 11.3 + 2.6j, 10.7 + 2.5j, 10.0 + 2.0j])

# Equation 3a
# omega is the angular frequency of radiation;
# lambda_c is the cutoff wavelength for the guide.
def gamma_0(omega, lambda_c):
    return -1j * ((omega / c)**2 - (2 * pi / lambda_c)**2)**(1/2)

# Equation 3b
# epsilon_star is the complex relative permittivity \epsilon_r^*;
# mu_star is the complex relative permeability \mu_r^*.
def gamma(omega, lambda_c, epsilon_star, mu_star):
    return -1j * ((omega / c)**2 * epsilon_star * mu_star - (2 * pi / lambda_c)**2)**(1/2)

#def Gamma(...


# Equation 4a

# z = np.exp(-gamma(...) * L)

# Equation 4b

# Equation 5a
def n(nu):
    """
    Return the real part of the complex index of refraction as a
    function of frequency \nu in Hz.
    """
    e_r = epsilon_r(nu)
    return np.sqrt(1/2 * (np.sqrt(e_r.real**2 + e_r.imag**2) + e_r.real))

# Equation 5b
def kappa(nu):
    """
    Return the imaginary part of the complex index of refraction as a
    function of frequency \nu in Hz.
    """
    e_r = epsilon_r(nu)
    return np.sqrt(1/2 * (np.sqrt(e_r.real**2 + e_r.imag**2) - e_r.real))

# Supplied fit to data shown in Table 2 and Figure 5.
def epsilon_r(nu):
    """
    Return the complex dielectric constant
    \epsilon_r = \epsilon_r' + i \epsilon_r'',
    for a stainless steel volume fraction of 0.3 as a function of
    frequency \nu in Hz. Note that the paper doesn't specify the units
    for frequency \nu. The closest match to the Theory lines in Figure
    5 seems to be given when \nu is expressed in GHz. The fit was
    obtained for data between 3 and 300 GHz.
    """
    return (13.8 + 3.4j) * (nu / 1e9)**-0.07

