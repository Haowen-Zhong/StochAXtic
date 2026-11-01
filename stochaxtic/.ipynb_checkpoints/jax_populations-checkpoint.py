# jax_populations.py
# JAX versions of the population models, for fast_omega.FastOmega.
#
# Mass models return the *unnormalised* joint density p(m1, m2) on m2 <= m1;
# FastOmega normalises it on its own grid, so only the shape matters (but
# conditional normalisations that are part of a model's definition, like
# p(m2 | m1) of bpl_double_gaussian, are kept).  Rate models return the
# comoving merger rate R(z) in Gpc^-3 yr^-1.
#
# Every model is a function of (arrays, params) where params is a dict of
# scalars; missing keys fall back to DEFAULTS, so numpyro can sample any
# subset of the hyperparameters.  The shapes and parameterisations follow
# mass_models.py and cosmology.py exactly.

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import erf

from .constants import HUBBLE_TIME, H0_GYR
from .cosmology import COSMO
from .mass_models import GWTC5_BPL2G, NSBHPowerLawGaussianModel, PowerLawPlusPeakModel


# ==========================================================
# Shape helpers (gradient safe: no nan/inf in masked-out branches)
# ==========================================================

def planck_taper(m, m_min, delta_m):
    f   = (m - m_min) / delta_m
    fs  = jnp.clip(f, 1e-6, 1.0 - 1e-6)
    arg = jnp.clip(1.0 / fs - 1.0 / (1.0 - fs), -700.0, 700.0)
    mid = jax.nn.sigmoid(-arg)        # = 1/(1+e^arg); its gradient stays finite at |arg| = 700
    return jnp.where(f <= 0.0, 0.0, jnp.where(f >= 1.0, 1.0, mid))


def _safe_pow(x, a):
    return jnp.where(x > 0, jnp.where(x > 0, x, 1.0) ** a, 0.0)


def interp_uniform(x, x0, dx, table):
    """
    Linear interpolation of `table` sampled at x0 + dx * arange(n), clamped
    at both ends.  O(1) per point, unlike jnp.interp's binary search, which
    dominates the cost on CPU for ~1e5 queries.
    """
    table = jnp.asarray(table)
    n = table.shape[0]
    s = jnp.clip((x - x0) / dx, 0.0, n - 1.0)
    i = jnp.clip(jnp.floor(s).astype(jnp.int32), 0, n - 2)
    f = s - i
    return table[i] * (1.0 - f) + table[i + 1] * f


def _between(m, lo=None, hi=None, w=0.0):
    """
    Indicator of lo <= m <= hi.  With w > 0 each edge is a logistic step of
    width w in ln m (FastOmega passes w ~ its ln M sample spacing), so that
    hard mass cut-offs have useful gradients; w = 0 gives the hard mask.
    """
    out = jnp.ones_like(m)
    for x in ((None if lo is None else jnp.log(m) - jnp.log(lo)),
              (None if hi is None else jnp.log(hi) - jnp.log(m))):
        if x is not None:
            ws  = jnp.where(w > 0, w, 1.0)
            out = out * jnp.where(w > 0, jax.nn.sigmoid(x / ws), (x >= 0).astype(m.dtype))
    return out


def _expm1_over_x(x, L):
    """
    expm1(x L) / x, with a series for |x| < 1e-4 so that the value and d/dx
    are both right at x = 0.  Power-law integrals:
        int_a^b m^-alpha dm = a^x expm1(x L) / x,  x = 1 - alpha, L = ln(b/a).
    """
    small = jnp.abs(x) < 1e-4
    xs    = jnp.where(small, 1.0, x)                     # keeps the unused branch finite
    return jnp.where(small, L * (1 + x * L / 2 + (x * L) ** 2 / 6), jnp.expm1(xs * L) / xs)


# ==========================================================
# Mass models: unnormalised p(m1, m2)
# ==========================================================

# int_{m2_min}^{m1} p(m2) dm2 of bpl_double_gaussian on a fixed grid uniform
# in ln m2, so that m_2_min, beta_q, delta_m_2 can be sampled
_LN_M2_0, _D_LN_M2, _N_M2 = np.log(0.5), np.log(4000.0) / 5999, 6000
_M2_GRID = np.exp(_LN_M2_0 + _D_LN_M2 * np.arange(_N_M2))


def bpl_double_gaussian(m1, m2, p):
    a1, a2, mb = p["alpha_m_1"], p["alpha_m_2"], p["m_break"]
    mmin, mmax = p["m_1_min"], p["m_max"]
    bpl_norm = mb * (_expm1_over_x(a1 - 1, jnp.log(mb / mmin))       # int_mmin^mb (m/mb)^-a1 dm / mb
                     + _expm1_over_x(1 - a2, jnp.log(mmax / mb)))    # int_mb^mmax (m/mb)^-a2 dm / mb
    w   = p.get("_w", 0.0)
    bpl = jnp.where(m1 < mb, _safe_pow(m1 / mb, -a1), _safe_pow(m1 / mb, -a2)) / bpl_norm
    bpl = bpl * jnp.where(m1 >= mmin, 1.0, 0.0)          # taper(m1) vanishes below mmin anyway

    def gauss(mu, sigma):
        Z = 0.5 * (1.0 + erf((mu - mmin) / (sigma * jnp.sqrt(2.0))))
        return jnp.exp(-0.5 * ((m1 - mu) / sigma) ** 2) / (sigma * jnp.sqrt(2 * jnp.pi)) / Z

    l1, l2 = p["lambda_peak_1"], p["lambda_peak_2"]
    p1 = (l1 * bpl + l2 * gauss(p["mu_m_1"], p["sigma_m_1"])
          + (1 - l1 - l2) * gauss(p["mu_m_2"], p["sigma_m_2"]))
    p1 = p1 * _between(m1, hi=mmax, w=w) * planck_taper(m1, mmin, p["delta_m_1"])

    def p_m2(m):
        return _safe_pow(m, p["beta_q"]) * planck_taper(m, p["m_2_min"], p["delta_m_2"])

    g   = jnp.asarray(_M2_GRID)
    pg  = p_m2(g)
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (pg[1:] + pg[:-1]) * jnp.diff(g))])
    Z2  = interp_uniform(jnp.log(m1), _LN_M2_0, _D_LN_M2, cum)
    # Z2 > 1e-150, not > 0: just above m_2_min the taper makes Z2 ~ 1e-200, and the
    # 1/Z2^2 in its derivative overflows to inf -> NaN gradients for every parameter
    ok  = (m2 <= m1) & (Z2 > 1e-150)
    return jnp.where(ok, p1 * p_m2(m2) / jnp.where(ok, Z2, 1.0), 0.0)


def uniform(m1, m2, p):
    w = p.get("_w", 0.0)
    return (_between(m1, p["m1_min"], p["m_max"], w) * _between(m2, lo=p["m2_min"], w=w)
            * jnp.where(m2 <= m1, 1.0, 0.0))


def truncated_powerlaw(m1, m2, p):
    w = p.get("_w", 0.0)
    return (_between(m1, p["m1_min"], p["m_max"], w) * _between(m2, lo=p["m2_min"], w=w)
            * jnp.where(m2 <= m1, _safe_pow(m1 * m2, -p["alpha"]), 0.0))


def nsbh(m1, m2, p):
    w = p.get("_w", 0.0)
    return (_between(m1, p["M1_MIN"], p["M_MAX"], w) * _between(m2, p["M2_MIN"], p["M2_MAX"], w)
            * jnp.where(m2 <= m1, 1.0 / m1, 0.0))


def nsbh_powerlaw_gaussian(m1, m2, p):
    ns_min = NSBHPowerLawGaussianModel.NS_MIN
    p1 = _safe_pow(m1, -p["alpha_BH"])
    q  = m2 / m1
    lo = ns_min / m1
    hi = jnp.minimum(p["m_max_NS"] / m1, 1.0)
    s2 = p["sigma_NS"] * jnp.sqrt(2.0)
    Z  = 0.5 * (erf((hi - p["mu_NS"]) / s2) - erf((lo - p["mu_NS"]) / s2))
    pq = (jnp.exp(-0.5 * ((q - p["mu_NS"]) / p["sigma_NS"]) ** 2)
          / (p["sigma_NS"] * jnp.sqrt(2 * jnp.pi)) / jnp.where(Z > 0, Z, 1.0))
    # q in [lo, hi] is the same as m2 in [ns_min, m_max_NS] together with q <= 1
    w  = p.get("_w", 0.0)
    ok = (m2 <= m1) & (Z > 0)
    return jnp.where(ok, p1 * pq / m1, 0.0) * (_between(m1, p["m_min_BH"], p["m_max_BH"], w)
                                              * _between(m2, ns_min, p["m_max_NS"], w))


def power_law_plus_peak(m1, m2, p):
    # see mass_models.PowerLawPlusPeakModel; p(m2 | m1) normalised as in bpl_double_gaussian
    a, mmin, mmax, lam = p["alpha_m"], p["m_min"], p["m_max"], p["lambda_peak"]
    pl_norm = mmin ** (1 - a) * _expm1_over_x(1 - a, jnp.log(mmax / mmin))
    mu, sig = p["mu_m"], p["sigma_m"]
    gauss = jnp.exp(-0.5 * ((m1 - mu) / sig) ** 2) / (sig * jnp.sqrt(2 * jnp.pi))
    p1 = (1 - lam) * _safe_pow(m1, -a) / pl_norm + lam * gauss
    p1 = p1 * _between(m1, hi=mmax, w=p.get("_w", 0.0)) * planck_taper(m1, mmin, p["delta_m"])

    def p_m2(m):
        return _safe_pow(m, p["beta_q"]) * planck_taper(m, mmin, p["delta_m"])

    g   = jnp.asarray(_M2_GRID)
    pg  = p_m2(g)
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (pg[1:] + pg[:-1]) * jnp.diff(g))])
    Z2  = interp_uniform(jnp.log(m1), _LN_M2_0, _D_LN_M2, cum)
    ok  = (m2 <= m1) & (Z2 > 1e-150)
    return jnp.where(ok, p1 * p_m2(m2) / jnp.where(ok, Z2, 1.0), 0.0)


MASS_MODELS = {
    "bpl_double_gaussian": (bpl_double_gaussian, dict(GWTC5_BPL2G)),
    "uniform":             (uniform, dict(m1_min=1.0, m2_min=1.0, m_max=2.0)),
    "truncated_powerlaw":  (truncated_powerlaw,
                            dict(alpha=2.35, m1_min=5.0, m2_min=3.0, m_max=100.0)),
    "nsbh":                (nsbh, dict(M1_MIN=3.0, M_MAX=50.0, M2_MIN=1.0, M2_MAX=2.0)),
    "nsbh_powerlaw_gaussian": (nsbh_powerlaw_gaussian, dict(NSBHPowerLawGaussianModel.DEFAULTS)),
    "power_law_plus_peak": (power_law_plus_peak, dict(PowerLawPlusPeakModel.DEFAULTS)),
}


# ==========================================================
# Rate models: R(z) in Gpc^-3 yr^-1
# ==========================================================

def madau_dickinson(z, p):
    """R0 psi(z) / psi(0): R(0) = R0 exactly (see cosmology.MadauDickinsonModel)."""
    return (p["R0"] * (1 + (1 + p["zp"]) ** -p["beta"]) * (1 + z) ** p["alpha"]
            / (1 + ((1 + z) / (1 + p["zp"])) ** p["beta"]))


def power_law(z, p):
    return p["R0"] * (1 + z) ** p["lam"]


def constant(z, p):
    return p["R0"] * jnp.ones_like(z)


class TimeDelayedMadauDickinson:
    """
    Same model and quadrature as cosmology.TimeDelayedMadauDickinsonModel
    (Gauss-Legendre in ln t_d with exact limits), for any z:
        rate = TimeDelayedMadauDickinson(cosmo);  rate(z, p)    (z of any shape)
    Differentiable in R0, alpha, zp, beta, td_min and td_max.  The cosmic time
    t(z) and its inverse z_f(t) need cosmo, which cannot run inside jit, so both
    are numpy tables built here (uniform in ln(1+z) and in t) and interpolated
    linearly in JAX; the t(z) table changes R by ~1e-8.  R = 0 above zf_max.
    RATE_MODELS holds an instance for cosmology.COSMO; FastOmega builds one with
    its own cosmology.  (Until 2026-10-22 the object was built for one z array
    and silently ignored the z passed to __call__.)
    """

    def __init__(self, cosmo, zf_max=10.0, n_td=64, n_tab=4000, n_t=40001):
        # numpy constants only: this object may be built outside and used
        # inside jit-compiled functions
        xi = np.linspace(0.0, np.log1p(zf_max), n_t)                   # xi = ln(1+z)
        self.xi0, self.dxi = xi[0], xi[1] - xi[0]
        self.t_of_xi = cosmo.cosmic_time(np.expm1(xi))
        self.t_end = float(cosmo.cosmic_time(zf_max))
        self.u, self.w = np.polynomial.legendre.leggauss(n_td)
        # ln(1+z_f) as a function of t on a grid uniform in t, t(zf_max)..t(0)
        t_tab = np.linspace(self.t_end, float(cosmo.cosmic_time(0.0)), n_tab)
        self.t0, self.dt = t_tab[0], t_tab[1] - t_tab[0]
        self.xi_of_t = np.log1p(cosmo.redshift_at_time(t_tab))

    def __call__(self, z, p):
        # z = 0 first (R(0) = R0 normalises); the table is clamped, so z > zf_max gets t_end
        xi   = jnp.log1p(jnp.concatenate([jnp.zeros(1), jnp.ravel(jnp.asarray(z, float))]))
        t_z  = interp_uniform(xi, self.xi0, self.dxi, self.t_of_xi)
        lo   = jnp.log(p["td_min"])
        up   = jnp.minimum(p["td_max"], t_z - self.t_end)
        hi   = jnp.log(jnp.maximum(up, p["td_min"]))
        half = 0.5 * (hi - lo)
        u    = (0.5 * (hi + lo))[:, None] + half[:, None] * self.u[None, :]
        z_f  = jnp.expm1(interp_uniform(t_z[:, None] - jnp.exp(u),
                                        self.t0, self.dt, self.xi_of_t))
        f_form = (1 + z_f) ** p["alpha"] / (1 + ((1 + z_f) / (1 + p["zp"])) ** p["beta"])
        raw  = half * (f_form @ self.w)
        return (p["R0"] * raw[1:] / raw[0]).reshape(jnp.shape(z))


RATE_DEFAULTS = {
    "madau_dickinson": dict(R0=17.99719280349701, alpha=2.850042173338945,
                            zp=2.270700222839242, beta=6.990668900157332),
    "power_law": dict(R0=20.0, lam=2.0),
    "constant":  dict(R0=20.0),
    "madau_dickinson_timedelay": dict(R0=23.4, alpha=2.6, zp=2.2, beta=6.2,
                                      td_min=0.02, td_max=HUBBLE_TIME),
}
RATE_MODELS = {"madau_dickinson": madau_dickinson, "power_law": power_law,
               "constant": constant,
               "madau_dickinson_timedelay": TimeDelayedMadauDickinson(COSMO)}


# ==========================================================
# Spin models: density of (a1, a2, cos t1, cos t2), mass independent
# ==========================================================
# The in-plane azimuth difference phi12 is uniform and cancels against the
# proposal.  Magnitudes live on [0, a_max] (a_max is fixed by FastOmega's
# spin table, e.g. 1 for BBH, ~0.05 for BNS): a / a_max ~ Beta.

def gwtc3_default_spin(a1, a2, c1, c2, p, a_max=1.0):
    """
    LVK "Default" spin model (GWTC-2/3):
      a_i / a_max ~ Beta(alpha_chi, beta_chi)            (iid)
      p(cos t1, cos t2) = zeta N(c1) N(c2) + (1 - zeta)/4,
      N = Gaussian(1, sigma_t) truncated to [-1, 1]
    """
    from jax.scipy.stats import beta as _beta
    am1, am2 = (a_max, a_max) if np.ndim(a_max) == 0 else a_max      # per-component bounds
    pa = (_beta.pdf(a1 / am1, p["alpha_chi"], p["beta_chi"])
          * _beta.pdf(a2 / am2, p["alpha_chi"], p["beta_chi"])) / (am1 * am2)
    s  = p["sigma_t"]
    Zt = 0.5 - 0.5 * (1.0 + erf(-2.0 / (s * jnp.sqrt(2.0))))       # int_{-1}^{1} N(c; 1, s)
    g  = lambda c: jnp.exp(-0.5 * ((c - 1.0) / s) ** 2) / (s * jnp.sqrt(2 * jnp.pi)) / Zt
    pt = p["zeta"] * g(c1) * g(c2) + (1.0 - p["zeta"]) / 4.0
    return pa * pt


def _truncnorm_pdf(x, mu, sig, lo, hi):
    """Normal(mu, sig) truncated to [lo, hi]."""
    Z = 0.5 * (erf((hi - mu) / (sig * jnp.sqrt(2.0))) - erf((lo - mu) / (sig * jnp.sqrt(2.0))))
    g = jnp.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * jnp.sqrt(2 * jnp.pi)) / Z
    return jnp.where((x >= lo) & (x <= hi), g, 0.0)


def gwtc5_default_spin(a1, a2, c1, c2, p, a_max=1.0):
    """
    LVK "Default BBH" spin model of GWTC-4.0/5.0 (arXiv:2605.27226, App. B.4, Eqs. B15-B16):
      a_i ~ Normal(mu_chi, sigma_chi) truncated to [0, chi_max]          (iid; chi_max = 1 for BHs)
      p(cos t1, cos t2) = zeta N(c1) N(c2) + (1 - zeta)/4,
      N = Normal(mu_t, sigma_t) truncated to [-1, 1]   (mu_t free, Vitale et al. 2022)
    gwtc3_default is the special case with a Beta magnitude and mu_t = 1.  The magnitudes
    are absolute (not a / a_max); FastOmega's self-normalised weights renormalise the
    density on the table's box [0, a_max], so a_max < chi_max silently drops a > a_max.
    """
    pa = (_truncnorm_pdf(a1, p["mu_chi"], p["sigma_chi"], 0.0, p["chi_max"])
          * _truncnorm_pdf(a2, p["mu_chi"], p["sigma_chi"], 0.0, p["chi_max"]))
    nt = lambda c: _truncnorm_pdf(c, p["mu_t"], p["sigma_t"], -1.0, 1.0)
    return pa * (p["zeta"] * nt(c1) * nt(c2) + (1.0 - p["zeta"]) / 4.0)


SPIN_MODELS = {
    # GWTC-3 medians of the fiducial analysis (O1-O3, Power Law + Peak, power-law redshift;
    # Zenodo 10.5281/zenodo.5655785): mu_chi = 0.283 and sigma_chi^2 = 0.0299 (mean and variance
    # of the Beta magnitudes, LVK's parameters) -> alpha_chi = 1.64, beta_chi = 4.15; zeta (xi_spin)
    # = 0.656, sigma_t (sigma_spin) = 1.54
    "gwtc3_default": (gwtc3_default_spin, dict(alpha_chi=1.64, beta_chi=4.15, zeta=0.656, sigma_t=1.54)),
    # GWTC-5 medians of the "Default BBH" analysis (variance cut 1; popsummary
    # gwtc5_updated_default_mmax_..._popsummary_result.h5): mu_chi = 0.0633, sigma_chi = 0.365,
    # zeta (xi_spin) = 0.671, mu_t (mu_spin) = 0.319, sigma_t (sigma_spin) = 1.11
    "gwtc5_default": (gwtc5_default_spin, dict(mu_chi=0.0633, sigma_chi=0.365, chi_max=1.0,
                                               mu_t=0.319, sigma_t=1.11, zeta=0.671)),
}
SPIN_MODELS["lvk_default"] = SPIN_MODELS["gwtc3_default"]       # old name (renamed 2026-10-15)


# ==========================================================
# Tide models: Lambda(m) of a neutron star, source-frame mass in Msun
# ==========================================================

def kappa2T_coefficients(q):
    """kappa2T = c1 Lambda1 + c2 Lambda2 for q = m2/m1 (LAL NRTidal definition, = 3/16 Lambda-tilde)."""
    X1, X2 = 1.0 / (1.0 + q), q / (1.0 + q)
    return ((3.0 / 13.0) * (1.0 + 12.0 * X2 / X1) * X1 ** 5,
            (3.0 / 13.0) * (1.0 + 12.0 * X1 / X2) * X2 ** 5)


def eos_lnpoly_lambda(m, p, w=0.0):
    """
    ln Lambda(m) = ln Lambda_14 + b1 ln(m/1.4) + b2 ln(m/1.4)^2  for m <= m_NS_max,
    Lambda = 0 (black hole) above m_NS_max (logistic edge of width w in ln m).
    b1 ~ -6 reproduces Lambda ∝ m^-6 of realistic EOSs near 1.4 Msun; b2 < 0 for
    every EOS checked (fits to LAL's SLY, APR4_EPP, WFF1, MPA1, ENG, H4, MS1B on
    1-2 Msun give b2 = -1.2 ... -4.9): Lambda falls faster than a power law towards
    the maximum mass.  The default b2 = -3 is a typical value; with b2 = 0 Omega_GW
    is off by 3-7% at 1 kHz (13-46% at 1.5 kHz) for SLY/H4-like EOSs.
    """
    lm = jnp.log(m / 1.4)
    lam = jnp.exp(jnp.log(p["Lambda_14"]) + p["b1"] * lm + p["b2"] * lm ** 2)
    return lam * _between(m, hi=p["m_NS_max"], w=w)


TIDE_MODELS = {
    # one deterministic EOS: Lambda is a function of m (no per-star scatter; EOS
    # uncertainty is common to all stars and belongs in the sampled hyperparameters)
    "eos_lnpoly": (eos_lnpoly_lambda, dict(Lambda_14=400.0, b1=-6.0, b2=-3.0, m_NS_max=2.3)),
}
