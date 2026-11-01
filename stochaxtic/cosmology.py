# cosmology.py
# Background cosmology (Planck18) and merger-rate models, all vectorised.
#
# Public interface:
#   COSMO                          the shared Planck18 Cosmology instance
#   get_rate_model(name, **kw)  -> a RateModelBase instance
#
# Rate models implement merger_rate(z) (comoving, Gpc^-3 yr^-1, array in
# array out).  The base class turns that into the observed event rate
#     dN/(dz dt_obs) = R(z) / (1+z) * dVc/dz          [s^-1]
#
# To add a model, subclass RateModelBase, implement merger_rate(z) and
# register it in _REGISTRY.

import numpy as np
from astropy.cosmology import Planck18
from scipy.integrate import cumulative_trapezoid

from .constants import C_LIGHT_KMS, H0_KMS_MPC, H0_GYR, HUBBLE_TIME, YR_S


# ==========================================================
# Background cosmology
# ==========================================================

class Cosmology:
    """
    Distances and cosmic time from tables on a fine grid in xi = ln(1+z).

    comoving distance  chi(z) = c/H0 int_0^z dz'/E(z')
    luminosity distance dL    = (1+z) chi                        [Mpc]
    dVc/dz                    = 4 pi c/H0 chi^2 / E(z)          [Mpc^3]
    cosmic time        t(z)   = 1/H0 int_z^inf dz'/((1+z') E)   [Gyr]

    E(z) comes from astropy (Planck18, with radiation and massive
    neutrinos).  The trapezoid rule on the default grid is accurate to
    ~1e-8 relative in the tabulated range.
    """
    # ✅ Reviewd HZ 10/23/26
    def __init__(self, astropy_cosmo=Planck18, z_max=100.0, n_xi=40001,
                 z_inf=1e7):
        self._ap = astropy_cosmo
        self.dH  = C_LIGHT_KMS / astropy_cosmo.H0.value              # Mpc

        # distance table on [0, z_max]
        xi   = np.linspace(0.0, np.log1p(z_max), n_xi)
        z    = np.expm1(xi)
        Ez   = astropy_cosmo.efunc(z)
        chi  = cumulative_trapezoid((1.0 + z) / Ez, xi, initial=0.0)  # dz = (1+z) dxi
        self._xi, self._chi, self._E = xi, chi, Ez
        self.z_max = z_max

        # cosmic time table: integrate from z_inf down to 0 in ln(1+z)
        xit  = np.linspace(0.0, np.log1p(z_inf), 4 * n_xi)
        Et   = astropy_cosmo.efunc(np.expm1(xit))
        tail = cumulative_trapezoid((1.0 / Et)[::-1], -xit[::-1], initial=0.0)[::-1]
        self._xit, self._t = xit, tail * (H0_KMS_MPC / astropy_cosmo.H0.value) / H0_GYR

    # ---------- public, vectorised ----------
    # ✅ Reviewd HZ 10/23/26
    def E(self, z):
        return np.interp(np.log1p(z), self._xi, self._E)
    # ✅ Reviewd HZ 10/23/26
    def comoving_distance(self, z):
        return self.dH * np.interp(np.log1p(z), self._xi, self._chi)
    # ✅ Reviewd HZ 10/23/26
    def luminosity_distance(self, z):
        return (1.0 + np.asarray(z)) * self.comoving_distance(z)
    # ✅ Reviewd HZ 10/23/26
    def dVc_dz(self, z):
        chi = self.comoving_distance(z)
        return 4.0 * np.pi * self.dH * chi ** 2 / self.E(z)
    # ✅ Reviewd HZ 10/23/26
    def cosmic_time(self, z):
        return np.interp(np.log1p(z), self._xit, self._t)
    # ✅ Reviewd HZ 10/23/26
    def redshift_at_time(self, t):
        """Inverse of cosmic_time (t in Gyr)."""
        return np.expm1(np.interp(t, self._t[::-1], self._xit[::-1])) # np.interpolate is monotonic, so reverse the arrays


COSMO = Cosmology()


# ==========================================================
# Merger-rate models
# ==========================================================

class RateModelBase:
    """Common interface.  Subclasses implement merger_rate(z) only."""
# ✅ Reviewd HZ 10/23/26
    def merger_rate(self, z):
        """Comoving merger-rate density R(z) in Gpc^-3 yr^-1 (vectorised)."""
        raise NotImplementedError
# ✅ Reviewd HZ 10/23/26
    def event_rate_density(self, z, cosmo=COSMO):
        """dN/(dz dt_obs) = R(z) dVc/dz / (1+z), in s^-1 (vectorised)."""
        z = np.asarray(z, dtype=float)
        R_si = self.merger_rate(z) / 1e9 / YR_S                     # Mpc^-3 s^-1
        return R_si * cosmo.dVc_dz(z) / (1.0 + z)
# ✅ Reviewd HZ 10/23/26
    def total_rate(self, z_min, z_max, n_z=4000, cosmo=COSMO):
        """int_{z_min}^{z_max} dN/(dz dt) dz in s^-1 (trapezoid in ln(1+z))."""
        xi = np.linspace(np.log1p(z_min), np.log1p(z_max), n_z)
        z  = np.expm1(xi)
        return np.trapezoid(self.event_rate_density(z, cosmo) * (1.0 + z), xi)

# ✅ Reviewd HZ 10/23/26
class MadauDickinsonModel(RateModelBase):
    """
    R(z) = R0 psi(z) / psi(0),   
    psi(z) = (1+z)^alpha / (1 + ((1+z)/(1+zp))^beta)
    
    normalized so that R(0) = R0 exactly (as in gwpopulation).  
    """
# ✅ Reviewd HZ 10/23/26
    def __init__(self, R0=17.99719280349701, alpha=2.850042173338945,
                 zp=2.270700222839242, beta=6.990668900157332):
        self.R0, self.alpha, self.zp, self.beta = R0, alpha, zp, beta
# ✅ Reviewd HZ 10/23/26
    def merger_rate(self, z):
        z = np.asarray(z, dtype=float)
        return (self.R0 * (1 + (1 + self.zp) ** -self.beta) * (1 + z) ** self.alpha
                / (1 + ((1 + z) / (1 + self.zp)) ** self.beta))

# ✅ Reviewd HZ 10/23/26
class PowerLawRedshiftModel(RateModelBase):
    """R(z) = R0 (1+z)^lam."""

    def __init__(self, R0=20.0, lam=2.0):
        self.R0, self.lam = R0, lam

    def merger_rate(self, z):
        return self.R0 * (1 + np.asarray(z, dtype=float)) ** self.lam

# ✅ Reviewd HZ 10/23/26
class ConstantRateModel(RateModelBase):
    """R(z) = R0."""

    def __init__(self, R0=20.0):
        self.R0 = R0

    def merger_rate(self, z):
        return np.full(np.shape(z), float(self.R0))

# ✅ Reviewd HZ 10/23/26
class TimeDelayedMadauDickinsonModel(RateModelBase):
    """
    Madau-Dickinson formation rate convolved with a t_d^-1 time delay:

        R(z) = norm * int_{z_f > z} f_form(z_f) P(t_d) |dt/dz_f| dz_f
        t_d  = t(z) - t(z_f),   P(t_d) = 1 / [t_d ln(td_max/td_min)]

    normalised so that R(0) = R0.  Since |dt/dz_f| dz_f = dt_d, the integral
    is done in u = ln t_d,
        R(z) ∝ int_{ln td_min}^{ln td_up(z)} f_form(z_f(t(z) - e^u)) du,
        td_up(z) = min(td_max, t(z) - t(zf_max)),
    by Gauss-Legendre with exact limits (the 1/ln(td_max/td_min) cancels in
    the normalisation).  The MOGWAS version integrated over z_f on a fixed
    grid with the t_d window as a mask, whose edges fall between grid points:
    R(z) then carried percent-level discretisation noise.  R(z) is tabulated
    once on {0} + a log grid in [1e-4, zf_max] and linearly interpolated;
    above zf_max the rate is zero.
    """

    def __init__(self, R0=23.4, alpha=2.6, zp=2.2, beta=6.2,
                 td_min=0.02, td_max=HUBBLE_TIME, zf_max=10.0,
                 n_td=256, n_interp=20000, cosmo=COSMO):
        if td_min <= 0 or td_max <= td_min:
            raise ValueError("require 0 < td_min < td_max")
        self.R0, self.alpha, self.zp, self.beta = R0, alpha, zp, beta
        self.td_min, self.td_max, self.zf_max = td_min, td_max, zf_max
        self._gl = np.polynomial.legendre.leggauss(n_td)

        self._z_grid = np.concatenate([[0.0], np.geomspace(1e-4, zf_max, n_interp)])
        raw = self._raw(self._z_grid, cosmo)
        if raw[0] <= 0.0:
            raise ValueError("the time-delay integral vanishes at z = 0; "
                             "check td_min / td_max / zf_max")
        self._R_grid = R0 * raw / raw[0]

    def _formation_rate(self, z):
        return (1.0 + z) ** self.alpha / (1.0 + ((1.0 + z) / (1.0 + self.zp)) ** self.beta)

    def _raw(self, z, cosmo):
        t_z  = cosmo.cosmic_time(z)
        lo   = np.log(self.td_min)
        up   = np.minimum(self.td_max, t_z - cosmo.cosmic_time(self.zf_max))
        hi   = np.log(np.maximum(up, self.td_min))
        half = 0.5 * (hi - lo)
        u    = (0.5 * (hi + lo))[:, None] + half[:, None] * self._gl[0][None, :]
        z_f  = cosmo.redshift_at_time(t_z[:, None] - np.exp(u))
        return half * (self._formation_rate(z_f) @ self._gl[1])

    def merger_rate(self, z):
        return np.interp(z, self._z_grid, self._R_grid, right=0.0)

# ✅ Reviewd HZ 10/23/26
_REGISTRY = {
    "madau_dickinson":           MadauDickinsonModel,
    "power_law":                 PowerLawRedshiftModel,
    "constant":                  ConstantRateModel,
    "madau_dickinson_timedelay": TimeDelayedMadauDickinsonModel,
}

# ✅ Reviewd HZ 10/23/26
def get_rate_model(name="madau_dickinson", **kwargs):
    """
    Usage:
        get_rate_model("madau_dickinson", R0=15.9)
        get_rate_model("madau_dickinson_timedelay", R0=23.4)
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown rate model '{name}'; available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
