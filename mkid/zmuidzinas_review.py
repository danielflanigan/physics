from __future__ import division
import numpy as np
from scipy.constants import h, k
from scipy.integrate import trapz

# In Kelvin.
aluminum_bulk_T_c = 1.2

def aluminum_Delta_0(T_c=aluminum_bulk_T_c):
    """
    Return the superconductor gap energy in Joules at zero
    temperature.
    """
    return h * 74e9 * T_c / 2

def fermi_dirac(T, E):
    """
    Return the Fermi-Dirac occupancy of a state with energy E in
    Joules at Kelvin temperature T.
    """
    return (np.exp(E / (k * T)) + 1)**-1

def conductivity_real(f, T, Delta, occupancy=fermi_dirac):
    """
    Equation 1: return sigma_1(f, T) / sigma_n.

    *f* is the frequency in Hz, not the angular frequency.
    *Delta* is a function Delta(T) of Kelvin temperature T that
     returns the superconductor gap energy for one electron.
    *occupancy* is a function f(T, E) of Kelvin temperature T and
     energy E in Joules that returns the average occupancy of a state
     with energy E; for a thermal distribution of fermions, this is
     the Fermi-Dirac distribution.
    """
    D = Delta(T)
    E = np.linspace(D, 5 * D, 1e4)
    return 2 / (h * f) * trapz(conductivity_real_integrand(f, T, Delta, occupancy, E),
                               x=E)

def conductivity_real_integrand(E, f, T, Delta, occupancy):
    D = Delta(T)
    F_E = occupancy(T, E)
    F_Ehf = occupancy(T, E + h * f)
    return ((E**2 + D**2 + h * f * E) * (F_E - F_Ehf) /
            ((E**2 - D**2)**(1/2) * ((E + h * f)**2 - D**2)**(1/2)))

def conductivity_imag(f, T, Delta, occupancy):
    """
    Equation 2: return sigma_2(f, T) / sigma_n.

    *f* is the frequency in Hz, not the angular frequency.
    *Delta* is a function Delta(T) of Kelvin temperature T that
     returns the superconductor gap energy for one electron.
    *occupancy* is a function f(T, E) of Kelvin temperature T and
     energy E in Joules that returns the average occupancy of a state
     with energy E; for a thermal distribution of fermions, this is
     the Fermi-Dirac distribution.
    """
    D = Delta(T)
    E = np.linspace(D, 1e3 * D, 1e4)
    return 1 / (h * f) * trapz(conductivity_imag_integrand(f, T, Delta, occupancy, E))

def conductivity_imag_integrand(f, T, Delta, occupancy, E):
    D = Delta(T)
    F_E = occupancy(T, E)
    return ((E**2 + D**2 - h * f * E) * (1 - 2 * F_E) /
            ((E**2 - D**2)**(1/2) * (D**2 - (E - h * f)**2)**(1/2)))


def n_qp(Delta, occupancy, N_0):
    """
    Return the quasiparticle density in number per cubic 
    
    *Delta* is a function Delta(T) of Kelvin temperature T that
     returns the superconductor gap energy for one electron.
    *occupancy* is a function f(T, E) of Kelvin temperature T and
     energy E in Joules that returns the occupancy probability for a
     state withenergy E; for a thermal distribution of fermions, this
     is the Fermi-Dirac distribution.
    *N_0* is the single-spin density of electron states at the Fermi
     level.
    """
    pass
     
