from __future__ import division

import numpy as np
from scipy.constants import c

class Rectangular(object):

    # Handle non-unity permittivity and permeability.
    def __init__(self, a, b):
        if a >= b:
            self.x = a
            self.y = b
        else:
            self.x = b
            self.y = a
        self.f0 = c / (2 * self.x)
        
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

