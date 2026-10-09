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
from .mass_models import GWTC5_BPL2G, NSBHModSmoothModel


# ==========================================================
# Shape helpers (gradient safe: no nan/inf in masked-out branches)
# ==========================================================

def planck_taper(m, m_min, delta_m):
    f   = (m - m_min) / delta_m
    fs  = jnp.clip(f, 1e-6, 1.0 - 1e-6)
    arg = jnp.clip(1.0 / fs - 1.0 / (1.0 - fs), -700.0, 700.0)
    mid = jax.nn.sigmoid(-arg)        # = 1/(1+e^arg); its gradient stays finite at |arg| = 700
    return jnp.where(f <= 0.0, 0.0, jnp.where(f >= 1.0, 1.0, mid))


def polynomial_filter(x, edge, sig):
    t = jnp.clip((x - edge) / sig, 0.0, 1.0)
    return t ** 3 * (10.0 - 15.0 * t + 6.0 * t ** 2)


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
    bpl_norm = mb * ((1 - (mmin / mb) ** (1 - a1)) / (1 - a1)
                     + ((mmax / mb) ** (1 - a2) - 1) / (1 - a2))
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


def nsbh_modsmooth(m1, m2, p):
    ns_min = NSBHModSmoothModel.NS_MIN
    p1 = _safe_pow(m1, -p["alpha_BH"]) * (
        polynomial_filter(m1, p["m_min_BH"] - p["sigma_l"], p["sigma_l"])
        * (1.0 - polynomial_filter(m1, p["m_max_BH"], p["sigma_h"])))
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


MASS_MODELS = {
    "bpl_double_gaussian": (bpl_double_gaussian, dict(GWTC5_BPL2G)),
    "uniform":             (uniform, dict(m1_min=1.0, m2_min=1.0, m_max=2.0)),
    "truncated_powerlaw":  (truncated_powerlaw,
                            dict(alpha=2.35, m1_min=5.0, m2_min=3.0, m_max=100.0)),
    "nsbh":                (nsbh, dict(M1_MIN=3.0, M_MAX=50.0, M2_MIN=1.0, M2_MAX=2.0)),
    "nsbh_modsmooth":      (nsbh_modsmooth, dict(NSBHModSmoothModel.DEFAULTS)),
}


# ==========================================================
# Rate models: R(z) in Gpc^-3 yr^-1
# ==========================================================

def madau_dickinson(z, p):
    return (p["R0"] * (1 + z) ** p["alpha"]
            / (1 + ((1 + z) / (1 + p["zp"])) ** p["beta"]))


def power_law(z, p):
    return p["R0"] * (1 + z) ** p["lam"]


def constant(z, p):
    return p["R0"] * jnp.ones_like(z)


class TimeDelayedMadauDickinson:
    """
    Same model and quadrature as cosmology.TimeDelayedMadauDickinsonModel
    (Gauss-Legendre in ln t_d with exact limits), evaluated directly at a
    fixed set of z values.  Differentiable in R0, alpha, zp, beta, td_min
    and td_max.
    """

    def __init__(self, z, cosmo, zf_max=10.0, n_td=64, n_tab=4000):
        # numpy constants only: this object may be built outside and used
        # inside jit-compiled functions
        z_all = np.concatenate([[0.0], np.asarray(z, float)])
        self.t_z   = cosmo.cosmic_time(z_all)
        self.t_end = float(cosmo.cosmic_time(zf_max))
        self.u, self.w = np.polynomial.legendre.leggauss(n_td)
        # ln(1+z_f) as a function of t on a grid uniform in t, t(zf_max)..t(0)
        t_tab = np.linspace(self.t_end, float(cosmo.cosmic_time(0.0)), n_tab)
        self.t0, self.dt = t_tab[0], t_tab[1] - t_tab[0]
        self.x_of_t = np.log1p(cosmo.redshift_at_time(t_tab))

    def __call__(self, z, p):
        lo   = jnp.log(p["td_min"])
        up   = jnp.minimum(p["td_max"], self.t_z - self.t_end)
        hi   = jnp.log(jnp.maximum(up, p["td_min"]))
        half = 0.5 * (hi - lo)
        u    = (0.5 * (hi + lo))[:, None] + half[:, None] * self.u[None, :]
        z_f  = jnp.expm1(interp_uniform(self.t_z[:, None] - jnp.exp(u),
                                        self.t0, self.dt, self.x_of_t))
        f_form = (1 + z_f) ** p["alpha"] / (1 + ((1 + z_f) / (1 + p["zp"])) ** p["beta"])
        raw  = half * (f_form @ self.w)
        return p["R0"] * raw[1:] / raw[0]


RATE_DEFAULTS = {
    "madau_dickinson": dict(R0=17.99719280349701, alpha=2.850042173338945,
                            zp=2.270700222839242, beta=6.990668900157332),
    "power_law": dict(R0=20.0, lam=2.0),
    "constant":  dict(R0=20.0),
    "madau_dickinson_timedelay": dict(R0=23.4, alpha=2.6, zp=2.2, beta=6.2,
                                      td_min=0.02, td_max=HUBBLE_TIME),
}
RATE_MODELS = {"madau_dickinson": madau_dickinson, "power_law": power_law,
               "constant": constant}


# ==========================================================
# Spin models: density of (a1, a2, cos t1, cos t2), mass independent
# ==========================================================
# The in-plane azimuth difference phi12 is uniform and cancels against the
# proposal.  Magnitudes live on [0, a_max] (a_max is fixed by FastOmega's
# spin table, e.g. 1 for BBH, ~0.05 for BNS): a / a_max ~ Beta.

def lvk_default_spin(a1, a2, c1, c2, p, a_max=1.0):
    """
    LVK "Default" spin model (GWTC-2/3):
      a_i / a_max ~ Beta(alpha_chi, beta_chi)            (iid)
      p(cos t1, cos t2) = zeta N(c1) N(c2) + (1 - zeta)/4,
      N = Gaussian(1, sigma_t) truncated to [-1, 1]
    """
    from jax.scipy.stats import beta as _beta
    pa = (_beta.pdf(a1 / a_max, p["alpha_chi"], p["beta_chi"])
          * _beta.pdf(a2 / a_max, p["alpha_chi"], p["beta_chi"])) / a_max ** 2
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
    lvk_default is the special case with a Beta magnitude and mu_t = 1.  The magnitudes
    are absolute (not a / a_max); FastOmega's self-normalised weights renormalise the
    density on the table's box [0, a_max], so a_max < chi_max silently drops a > a_max.
    """
    pa = (_truncnorm_pdf(a1, p["mu_chi"], p["sigma_chi"], 0.0, p["chi_max"])
          * _truncnorm_pdf(a2, p["mu_chi"], p["sigma_chi"], 0.0, p["chi_max"]))
    nt = lambda c: _truncnorm_pdf(c, p["mu_t"], p["sigma_t"], -1.0, 1.0)
    return pa * (p["zeta"] * nt(c1) * nt(c2) + (1.0 - p["zeta"]) / 4.0)


SPIN_MODELS = {
    # defaults roughly at the GWTC-3 medians
    "lvk_default": (lvk_default_spin, dict(alpha_chi=1.6, beta_chi=4.1, zeta=0.66, sigma_t=1.5)),
    # PROVISIONAL defaults: matched to the GWTC-5.0 summary statistics P(chi <= 0.5) = 0.80 and
    # P(cos t > 0) = 0.54 (Sec. 3 of arXiv:2605.27226); replace by the popsummary medians
    "gwtc5_default": (gwtc5_default_spin, dict(mu_chi=0.15, sigma_chi=0.32, chi_max=1.0,
                                               mu_t=0.3, sigma_t=0.8, zeta=0.4)),
}


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
