from __future__ import division

import numpy as np

class GaussianBeam(object):
    """
    This class represents a Gaussian beam of radiation propagating
    in the positive z direction with waist at z = 0 with the given
    wavelength, beam waist w_0, and initial amplitude A_0.

    The equations seem to work for both positive (diverging beam) and
    negative (converging beam) values of z.

    Most properties are written in terms of the confocal distance
    z_{c} = \pi w_{0}^{2} / \lambda;
    see Goldsmith, Quasioptical Systems.
    """

    def __init__(self, wavelength, w_0, A_0=None):
        """
        If no initial amplitude is given, the beam is normalized so
        that the integral of the electric field squared over the z = 0
        plane is 1: A_{0} = (2 / \pi)^(1/2) / w_{0}.
        """
        self.wavelength = float(wavelength)
        self.w_0 = float(w_0)
        if A_0 is None:
            self.A_0 = np.sqrt(2 / np.pi) / self.w_0
        else:
            self.A_0 = float(A_0)
        self.z_c = np.pi * self.w_0**2 / self.wavelength
        self.k = 2 * np.pi / self.wavelength

    def A(self, z):
        """
        Return the beam peak amplitude at the points z.
        """
        return self.A_0 * (1 + 1j * z / self.z_c) / (1 + (z / self.z_c)**2)
    
    def R(self, z):
        """
        Return the radius of curvature at the points z; this is
        negative for negative z.
        """
        return z + self.z_c**2 / z

    def w(self, z):
        """
        Return the beam waist at the points z.
        """
        return self.w_0 * (1 + (z / self.z_c)**2)**(1/2)

    def equiphase(self, z, z_0):
        """
        Return the radius values at the points z of the surface having
        the same phase as the beam at the single point z_0.
        """
        R_0 = self.R(z_0)
        return (R_0**2 - (z + R_0 - z_0)**2)**(1/2)
    

    def phi_0(self, z):
        return np.arctan(z / self.z_c)
    
    def u(self, r, z):
        rr, zz = np.meshgrid(r, z)
        ww = self.w(zz)
        # np.meshgrid returns an array of shape (z.size, r.size).
        return (self.A_0 * self.w_0 / ww *
                np.exp(- rr**2 / ww**2
                       - 1j* np.pi * rr**2 / (self.wavelength * self.R(zz))
                       + 1j * self.phi_0(zz))).T

    def E(self, r, z):
        return self.u(r, z) * np.exp(- 1j * self.k * z)
