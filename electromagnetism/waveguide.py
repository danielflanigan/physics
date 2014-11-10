from __future__ import division

import numpy as np
from scipy.constants import c, pi
from scipy.special import jnjnp_zeros


class Waveguide(object):

    def cutoffs_below(self, f):
        raise NotImplementedError()

    def propagating_modes(self, ff):
        return np.searchsorted(self.cutoffs_below(ff[-1]), ff)


class Rectangular(Waveguide):

    # Handle non-unity permittivity and permeability.
    def __init__(self, a, b):
        if a >= b:
            self.x = a
            self.y = b
        else:
            self.x = b
            self.y = a
        self.f_0 = c / (2 * self.x)
        
    def cutoffs_below(self, f):
        """
        Return the cutoff frequencies of all propagating modes below
        frequency f in Hz for an x by y rectangular waveguide. A mode
        is not considered to be propagating exactly at its cutoff
        frequency.
        """
        mm, nn = np.meshgrid(np.arange(2 * f * self.x / c), np.arange(2 * f * self.y / c), indexing='ij')
        ff = c / 2 * (mm**2 / self.x**2 + nn**2 / self.y**2)**(1/2)
        # Exclude m = n = 0.
        TE_cutoffs = ff[ff < f][1:]
        TM_cutoffs = ff[1:,1:][ff[1:,1:] < f]
        cutoffs = np.concatenate((TE_cutoffs, TM_cutoffs))
        cutoffs.sort()
        return cutoffs


class Circular(Waveguide):

    # As of numpy 1.9, scipy.special.jnjnp_zeros computes only the smallest 1200 roots of J_n and J_n' for all n. This
    # sets an upper bound on cutoff frequencies that can be reliably computed for a given waveguide radius. This largest
    # root is jnjnp_zeros(1200)[0][-1].
    LARGEST_ZERO = 68.563138567071263
    
    def __init__(self, a):
        """
        Return a Circular waveguide with radius a in meters.
        """
        self.a = float(a)
        # See below for explanation of indices.
        self.f_0 = (jnjnp_zeros(2)[0][1] * c /
                   (2 * pi * self.a))
        # See above for explanation.
        self.f_max = (self.LARGEST_ZERO * c /
                      (2 * pi * self.a))
                      

    def cutoffs_below(self, f):
        """
        Return the cutoff frequencies of all propagating modes below
        frequency f in Hz for an x by y rectangular waveguide. A mode
        is not considered to be propagating exactly at its cutoff
        frequency.
        """
        if f > self.f_max:
            raise ValueError("The given frequency is too large.")
        zo, m, n, t = jnjnp_zeros(1200)
        # The Bessel functions J_n(x) for n > 0 all have a root at x = 0; the derivative of the Bessel function J_0(x)
        # has a root at x = 0.  The function jnjnp_zeros returns an array of the roots that starts with a single zero,
        # which it attributes to J_0', and this root is ignored in the analysis below.
        ff = (zo[1:] * c /
              (2 * pi * self.a))
        return ff[ff < f]
        
