from __future__ import division
import numpy as np
from scipy.special import i0, k0

from scipy.constants import h, k as k_B, pi, physical_constants
k_B_eV = physical_constants['Boltzmann constant in eV/K'][0]
BCS = 1.764


class KID(object):

    def __init__(self, active_metal, active_volume_um3, inactive_metal, inactive_volume_um3, substrate,
                 phonon_trapping_factor, alpha, f_r, iQc, iQi0, S_TLS_at_P_c, P_c):
        self.substrate = substrate
        self.active_metal = active_metal
        self.active_volume_um3 = active_volume_um3
        self.inactive_metal = inactive_metal
        self.inactive_volume_um3 = inactive_volume_um3
        self.phonon_trapping_factor = phonon_trapping_factor
        self.alpha = alpha
        self.f_r = f_r
        self.iQc = iQc
        self.iQi0 = iQi0
        self.S_TLS_at_P_c = S_TLS_at_P_c
        self.P_c = P_c

    def iQqp(self, Gamma, T_qp=None):
        return self.d_iQi_d_N_qp(T_qp=T_qp) * self.N_qp(Gamma=Gamma)

    def iQi(self, Gamma, T_qp=None):
        return self.iQi0 + self.iQqp(Gamma=Gamma, T_qp=T_qp)

    def x(self, Gamma, T_qp=None):
        return self.d_x_d_N_qp(T_qp=T_qp) * self.N_qp(Gamma=Gamma)

    def tau_qp(self, Gamma):
        return self.N_qp(Gamma=Gamma) / (2 * Gamma)

    def N_qp(self, Gamma):
        V = self.active_volume_um3
        R_star = self.effective_recombination_um3_per_s
        return (V * Gamma / R_star)**(1/2)

    def chi_c(self, iQi):
        return (4 * iQi * self.iQc /
                (iQi + self.iQc)**2)

    def chi_g(self, iQi, x):
        return (1 + (2 * x / (iQi + self.iQc))**2)**-1

    # The default assumes that we can always tune on-resonance.
    def chi_a(self, iQi, x=0):
        return 1 / 2 * self.chi_c(iQi=iQi) * self.chi_g(iQi=iQi, x=x)

    def optical_generation_rate(self, P, nu):
        return self.active_metal.quasiparticles_per_photon(nu) * P / (h * nu)

    @property
    def thermal_generation_rate(self):
        return self.active_volume_um3 * self.active_metal.thermal_generation_per_um3(T=self.substrate.T)

    @property
    def effective_recombination_um3_per_s(self):
        return self.active_metal.R_um3_per_s / self.phonon_trapping_factor

    # Responsivity equations

    def d_Gamma_A_d_P_A(self, nu):
        return self.active_metal.quasiparticles_per_photon(nu) / (h * nu)

    def d_N_qp_d_Gamma_A(self, Gamma):
        return self.tau_qp(Gamma=Gamma)

    def d_iQi_d_N_qp(self, T_qp=None):
        if T_qp is None:
            T_qp = self.substrate.T
        s1 = self.active_metal.S_1(self.f_r, T_qp)
        V = self.active_volume_um3
        N0 = self.active_metal.N0_per_um3_per_eV
        Delta = self.active_metal.Delta_eV
        return 2 * self.alpha * s1 / (4 * N0 * Delta * V)

    def d_x_d_N_qp(self, T_qp=None):
        if T_qp is None:
            T_qp = self.substrate.T
        s2 = self.active_metal.S_2(self.f_r, T_qp)
        V = self.active_volume_um3
        N0 = self.active_metal.N0_per_um3_per_eV
        Delta = self.active_metal.Delta_eV
        return self.alpha * s2 / (4 * N0 * Delta * V)

    def d_S21_d_iQi(self, iQi, x=0):
        return self.chi_c(iQi=iQi) * self.chi_g(iQi=iQi, x=x) / (4 * iQi)

    def d_S21_d_x(self, iQi, x=0):
        return 1j * self.chi_c(iQi=iQi) * self.chi_g(iQi=iQi, x=x) / (2 * iQi)

    def NEP2_photon_simple(self, P_A, P_B, nu, bandwidth):
        return 2 * h * nu * (P_A + P_B) + 2 * P_A**2 / bandwidth

    def NEP2_photon(self, Gamma_A, Gamma_B, nu, bandwidth):
        m = self.active_metal.quasiparticles_per_photon(nu=nu)
        S_Gamma = 2 * m * Gamma_A * (1 + Gamma_A / (m * bandwidth)) + 4 * Gamma_B
        return S_Gamma / self.d_Gamma_A_d_P_A(nu=nu)**2

    def NEP2_recombination(self, Gamma, nu):
        """Gamma is the total generation rate, including all sources."""
        return 4 * Gamma / self.d_Gamma_A_d_P_A(nu=nu)**2

    def NEP2_TLS(self, P_g, Gamma, nu):
        P_i = self.chi_a(iQi=self.iQi0 + self.iQqp(Gamma=Gamma)) * P_g
        return ((self.S_TLS_at_P_c * (self.P_c / P_i) ** (1 / 2)) /
                (self.d_x_d_N_qp() *
                 self.d_N_qp_d_Gamma_A(Gamma=Gamma) *
                 self.d_Gamma_A_d_P_A(nu=nu)) ** 2)

    def NEP2_amp(self, T_amp, P_g, Gamma, nu):
        return ((k_B * T_amp / P_g) /
                (np.abs(self.d_S21_d_x(iQi=self.iQi0 + self.iQqp(Gamma=Gamma))) *
                 self.d_x_d_N_qp() *
                 self.d_N_qp_d_Gamma_A(Gamma=Gamma) *
                 self.d_Gamma_A_d_P_A(nu=nu))**2)


class Superconductor(object):

    N0_per_um3_per_eV = None
    tau_0 = None

    def __init__(self, sigma_n=None, T_c=None):
        self.sigma_n = sigma_n
        self.T_c = T_c

    @property
    def Delta(self):
        return BCS * k_B * self.T_c

    @property
    def Delta_eV(self):
        return BCS * k_B_eV * self.T_c

    @property
    def nu_gap(self):
        return 2 * BCS * k_B * self.T_c / h

    @property
    def R_um3_per_s(self):
        return (2 * BCS)**3 / (4 * self.N0_per_um3_per_eV * self.Delta_eV * self.tau_0)

    def eta_pb(self, nu):
        nu_g = self.nu_gap
        if nu < nu_g:
            return 0
        elif 2 * nu_g < nu:
            return 1 / 2
        else: # For 1 <= nu / nu_g < 2
            return nu_g / nu

    def quasiparticles_per_photon(self, nu):
        return 2 * self.eta_pb(nu) * nu / self.nu_gap

    def S_1(self, f, T):
        Delta = BCS * k_B * self.T_c
        xi = h * f / (2 * k_B * T)
        return 2 / pi * (2 * Delta / (pi * k_B * T)) ** (1 / 2) * np.sinh(xi) * k0(xi)

    def S_2(self, f, T):
        Delta = BCS * k_B * self.T_c
        xi = h * f / (2 * k_B * T)
        return 1 + (2 * Delta / (pi * k_B * T)) ** (1 / 2) * np.exp(-xi) * i0(xi)

    def thermal_generation_per_um3(self, T):
        n_qp_per_um3 = (4 * self.N0_per_um3_per_eV * self.Delta_eV *
                        (pi * k_B * T / (2 * self.Delta)) ** (1 / 2) *
                        np.exp(-self.Delta / (k_B * T)))
        return self.R_um3_per_s * n_qp_per_um3 ** 2


class Aluminum(Superconductor):

    N0_per_um3_per_eV = 1.72e10
    tau_0 = 438e-9

    def __init__(self, sigma_n=3.5e7, T_c=1.2):
        super(Aluminum, self).__init__(sigma_n=sigma_n, T_c=T_c)


class Substrate(object):

    def __init__(self, T):
        self.T = T
