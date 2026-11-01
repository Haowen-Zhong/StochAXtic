# mass_models.py
# Source-frame mass distributions p(m1, m2), vectorised and normalised.
#
# Public interface:
#   get_mass_model(name, **kw)  ->  a MassModelBase instance
#
# Every model returns a *normalised* density on the domain m2 <= m1:
#     int int_{m2 <= m1} p(m1, m2) dm1 dm2 = 1
# MassModelBase.check_normalisation() verifies this on a fine grid; the
# tests run it for every registered model.
#
# To add a model, subclass MassModelBase, set the bounds M1_MIN, M2_MIN,
# M_MAX in __init__, implement density(m1, m2) with numpy broadcasting and
# register it in _REGISTRY.

import numpy as np
from scipy.special import erf
from scipy.integrate import cumulative_trapezoid


# ==========================================================
# Base class
# ==========================================================

class MassModelBase:
    """
    Bounds (Msun) that define the integration domain:
        M1_MIN <= m1 <= M_MAX,  M2_MIN <= m2 <= m1
    density() must vanish outside this domain.
    """

    M1_MIN: float
    M2_MIN: float
    M_MAX:  float

    def density(self, m1, m2):
        """Normalised p(m1, m2), broadcasting over m1 and m2."""
        raise NotImplementedError

    # ---------- chirp mass / mass ratio ----------

    @staticmethod
    def mc_q_to_m1m2(mc, q):
        m1 = mc * (1.0 + q) ** 0.2 / q ** 0.6
        return m1, q * m1

    def density_mc_q(self, mc, q):
        """p(mc, q) = p(m1, m2) |d(m1,m2)/d(mc,q)| = p(m1, m2) m1 m2 / (q mc)."""
        m1, m2 = self.mc_q_to_m1m2(mc, q)
        with np.errstate(all="ignore"):
            p = self.density(m1, m2)
        return np.where(p > 0, p * m1 * m2 / (q * mc), 0.0)

    @property
    def q_min(self):
        return self.M2_MIN / self.M_MAX

    @property
    def mc_min(self):
        return (self.M1_MIN * self.M2_MIN) ** 0.6 / (self.M1_MIN + self.M2_MIN) ** 0.2

    @property
    def mc_max(self):
        return self.M_MAX * 2.0 ** -0.2

    # ---------- validation ----------

    def check_normalisation(self, n=4000):
        """int p dm1 dm2 on a fine log grid; should be 1 to ~1e-3."""
        m = np.geomspace(self.M2_MIN, self.M_MAX, n)
        P = self.density(m[:, None], m[None, :])
        return float(np.trapezoid(np.trapezoid(P, m, axis=1), m))


def _pl_integral(a, lo, hi):
    """int_lo^hi m^-a dm = lo^x expm1(x L) / x (x = 1 - a, L = ln(hi/lo)), exact near a = 1."""
    x, L = 1.0 - a, np.log(hi / lo)
    return L if x == 0 else lo ** x * np.expm1(x * L) / x


def _normalise_1d(f, lo, hi, n=20000):
    """int_lo^hi f(m) dm on a log grid fine enough for sub-Msun features."""
    m = np.geomspace(lo, hi, n)
    return float(np.trapezoid(f(m), m))


# ==========================================================
# Shared shape functions
# ==========================================================

def planck_taper(m, m_min, delta_m):
    """Planck taper window, 0 below m_min and 1 above m_min + delta_m."""
    m   = np.asarray(m, dtype=float)
    out = np.where(m >= m_min + delta_m, 1.0, 0.0)
    mid = (m > m_min) & (m < m_min + delta_m)
    if np.any(mid):
        f = (m[mid] - m_min) / delta_m
        with np.errstate(over="ignore", invalid="ignore"):
            v = 1.0 / (1.0 + np.exp(1.0 / f - 1.0 / (1.0 - f)))
        out[mid] = np.where(np.isfinite(v), v, 0.0)
    return out


def trunc_gaussian(x, mu, sigma, lower, upper):
    """Gaussian normalised on [lower, upper]; lower/upper may be arrays."""
    x   = np.asarray(x, dtype=float)
    s2  = sigma * np.sqrt(2.0)
    phi = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    Z   = 0.5 * (erf((upper - mu) / s2) - erf((lower - mu) / s2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((x >= lower) & (x <= upper) & (Z > 0), phi / Z, 0.0)


# ==========================================================
# Broken power law + two Gaussian peaks (GWTC-5.0 BBH)
# ==========================================================

GWTC5_BPL2G = {
    "alpha_m_1": 1.5754278453802124, "alpha_m_2": 5.663286019662541,
    "beta_q": 0.5572915358861987, "m_break": 38.06724241851806,
    "mu_m_1": 10.135808238284675, "sigma_m_1": 0.6612470197196616,
    "mu_m_2": 35.07723569532331, "sigma_m_2": 3.1243701779940696,
    "m_1_min": 4.121538752270456, "m_max": 300.0,
    "lambda_peak_1": 0.49341832652035744, "lambda_peak_2": 0.47992177777376155,
    "delta_m_1": 4.352759439353479, "m_2_min": 3.42818414788367,
    "delta_m_2": 5.8893002445024685,
}


class BrokenPowerLawDoubleGaussianModel(MassModelBase):
    """
    p(m1, m2) = p(m1) p(m2 | m1)
      p(m1)      ∝ [l1 BPL(m1) + l2 N(m1; mu1, s1) + (1-l1-l2) N(m1; mu2, s2)]
                   x taper(m1; m_1_min, delta_m_1)
      p(m2 | m1) ∝ m2^beta_q taper(m2; m_2_min, delta_m_2),  m2 in [M2_MIN, m1]
    Both factors are normalised numerically.  Any subset of the GWTC-5.0
    parameters can be overridden through `params`.
    """

    def __init__(self, params=None, n_m2_grid=20000):
        p = dict(GWTC5_BPL2G)
        p.update(params or {})
        self._p     = p
        self.M1_MIN = p["m_1_min"]
        self.M2_MIN = p["m_2_min"]
        self.M_MAX  = p["m_max"]

        a1, a2, mb = p["alpha_m_1"], p["alpha_m_2"], p["m_break"]
        mmin, mmax = p["m_1_min"], p["m_max"]
        self._bpl_norm = mb * (_pl_integral(a1, mmin / mb, 1.0) + _pl_integral(a2, 1.0, mmax / mb))

        # int_{M2_MIN}^{m1} p_m2 dm2, tabulated once
        self._m2_grid = np.geomspace(self.M2_MIN, self.M_MAX, n_m2_grid)
        self._m2_cum  = cumulative_trapezoid(self._p_m2(self._m2_grid),
                                             self._m2_grid, initial=0.0)
        self._norm_m1 = _normalise_1d(self._p_m1, self.M1_MIN, self.M_MAX)

    def _gauss(self, m, mu, sigma):
        """Gaussian truncated to [m_1_min, inf)."""
        mmin = self._p["m_1_min"]
        Z = 0.5 * (1.0 + erf((mu - mmin) / (sigma * np.sqrt(2))))
        if Z < 1e-300:
            return np.zeros_like(m)
        return np.exp(-0.5 * ((m - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)) / Z

    def _p_m1(self, m1):
        p  = self._p
        mb = p["m_break"]
        m1 = np.asarray(m1, dtype=float)
        bpl = np.where(m1 < mb, (m1 / mb) ** (-p["alpha_m_1"]),
                       (m1 / mb) ** (-p["alpha_m_2"])) / self._bpl_norm
        bpl = np.where((m1 >= p["m_1_min"]) & (m1 <= p["m_max"]), bpl, 0.0)
        l1, l2 = p["lambda_peak_1"], p["lambda_peak_2"]
        mix = (l1 * bpl + l2 * self._gauss(m1, p["mu_m_1"], p["sigma_m_1"])
               + (1 - l1 - l2) * self._gauss(m1, p["mu_m_2"], p["sigma_m_2"]))
        mix = np.where(m1 <= p["m_max"], mix, 0.0)
        return mix * planck_taper(m1, p["m_1_min"], p["delta_m_1"])

    def _p_m2(self, m2):
        p = self._p
        return np.asarray(m2, float) ** p["beta_q"] * planck_taper(m2, p["m_2_min"], p["delta_m_2"])

    def density(self, m1, m2):
        m1, m2 = np.broadcast_arrays(np.asarray(m1, float), np.asarray(m2, float))
        norm_m2 = np.interp(m1, self._m2_grid, self._m2_cum, left=0.0)
        ok = (m2 <= m1) & (m2 >= self.M2_MIN) & (norm_m2 > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = self._p_m1(m1) * self._p_m2(m2) / norm_m2
        return np.where(ok, out, 0.0) / self._norm_m1


# ==========================================================
# Uniform in (m1, m2)
# ==========================================================

class UniformModel(MassModelBase):
    """
    p(m1, m2) = 1/A on  m1_min <= m1 <= m_max,  m2_min <= m2 <= m1,
    with A the area of that region.  (The MOGWAS version used 1 instead of
    1/A, i.e. integrated to 1/2 for the defaults.)
    """

    def __init__(self, m1_min=1.0, m2_min=1.0, m_max=2.0):
        if not (m2_min <= m_max and m1_min < m_max):
            raise ValueError("need m2_min <= m_max and m1_min < m_max")
        self.M1_MIN, self.M2_MIN, self.M_MAX = m1_min, m2_min, m_max
        lo = max(m1_min, m2_min)
        self._area = 0.5 * ((m_max - m2_min) ** 2 - (lo - m2_min) ** 2)

    def density(self, m1, m2):
        m1, m2 = np.broadcast_arrays(np.asarray(m1, float), np.asarray(m2, float))
        ok = ((m1 >= self.M1_MIN) & (m1 <= self.M_MAX)
              & (m2 >= self.M2_MIN) & (m2 <= m1))
        return np.where(ok, 1.0 / self._area, 0.0)


# ==========================================================
# Truncated power law
# ==========================================================

class TruncatedPowerLawModel(MassModelBase):
    """
    p(m1, m2) ∝ m1^-alpha m2^-alpha on m1_min <= m1 <= m_max,
    m2_min <= m2 <= m1, normalised over that region.
    """

    def __init__(self, alpha=2.35, m1_min=5.0, m2_min=3.0, m_max=100.0):
        self.alpha = alpha
        self.M1_MIN, self.M2_MIN, self.M_MAX = m1_min, m2_min, m_max

        def m2_integral(m1):     # int_{m2_min}^{m1} m2^-alpha dm2
            if np.isclose(alpha, 1.0):
                return np.log(m1 / m2_min)
            return (m1 ** (1 - alpha) - m2_min ** (1 - alpha)) / (1 - alpha)

        self._norm = _normalise_1d(
            lambda m1: np.where(m1 >= m1_min, m1 ** -alpha * m2_integral(m1), 0.0),
            max(m1_min, m2_min), m_max)

    def density(self, m1, m2):
        m1, m2 = np.broadcast_arrays(np.asarray(m1, float), np.asarray(m2, float))
        ok = ((m1 >= self.M1_MIN) & (m1 <= self.M_MAX)
              & (m2 >= self.M2_MIN) & (m2 <= m1))
        with np.errstate(divide="ignore"):
            return np.where(ok, (m1 * m2) ** -self.alpha / self._norm, 0.0)


# ==========================================================
# NSBH: log-uniform BH, uniform NS
# ==========================================================

class NSBHModel(MassModelBase):
    """
    p(m1, m2) = 1 / [m1 ln(M_MAX/M1_MIN) (M2_MAX - M2_MIN)]
    on M1_MIN <= m1 <= M_MAX, M2_MIN <= m2 <= M2_MAX (requires M2_MAX <= M1_MIN
    so that m2 <= m1 everywhere).  (The MOGWAS version divided by the
    inverse of this normalisation.)
    """

    def __init__(self, M1_MIN=3.0, M_MAX=50.0, M2_MIN=1.0, M2_MAX=2.0):
        if M2_MAX > M1_MIN:
            raise ValueError("NSBHModel needs M2_MAX <= M1_MIN")
        self.M1_MIN, self.M_MAX, self.M2_MIN, self.M2_MAX = M1_MIN, M_MAX, M2_MIN, M2_MAX
        self._norm = np.log(M_MAX / M1_MIN) * (M2_MAX - M2_MIN)

    def density(self, m1, m2):
        m1, m2 = np.broadcast_arrays(np.asarray(m1, float), np.asarray(m2, float))
        ok = ((m1 >= self.M1_MIN) & (m1 <= self.M_MAX)
              & (m2 >= self.M2_MIN) & (m2 <= self.M2_MAX))
        with np.errstate(divide="ignore"):
            return np.where(ok, 1.0 / (m1 * self._norm), 0.0)


# ==========================================================
# NSBH: power-law BH, truncated-Gaussian mass ratio (arXiv:2207.01568)
# ==========================================================

class NSBHPowerLawGaussianModel(MassModelBase):
    """
    Defined in (m1, q):
      p(m1)     ∝ m1^-alpha_BH  on [m_min_BH, m_max_BH]  (hard edges)
      p(q | m1) = truncated Gaussian(mu_NS, sigma_NS) on
                  [1/m1, min(m_max_NS/m1, 1)]
    and p(m1, m2) = p(m1) p(q | m1) / m1.

    Formerly 'nsbh_modsmooth', which multiplied p(m1) by GWFAST's polynomial
    edge filter (sigma_l, sigma_h).  That filter is identically 1 on
    [m_min_BH, m_max_BH] and is not part of the published model, so it was
    removed without changing any result.
    """

    DEFAULTS = {"alpha_BH": 2.51, "mu_NS": 0.333, "sigma_NS": 0.54,
                "m_min_BH": 3.31, "m_max_BH": 13.0, "m_max_NS": 2.23}
    NS_MIN = 1.0

    def __init__(self, params=None):
        p = dict(self.DEFAULTS)
        p.update(params or {})
        self._p = p
        self.M1_MIN, self.M_MAX = p["m_min_BH"], p["m_max_BH"]
        self.M2_MIN, self.M2_MAX = self.NS_MIN, p["m_max_NS"]
        self._norm_m1 = _pl_integral(p["alpha_BH"], self.M1_MIN, self.M_MAX)

    def _p_m1(self, m1):
        m1 = np.asarray(m1, float)
        return np.where((m1 >= self.M1_MIN) & (m1 <= self.M_MAX),
                        m1 ** (-self._p["alpha_BH"]), 0.0)

    def density(self, m1, m2):
        p = self._p
        m1, m2 = np.broadcast_arrays(np.asarray(m1, float), np.asarray(m2, float))
        ok = ((m1 >= self.M1_MIN) & (m1 <= self.M_MAX)
              & (m2 >= self.M2_MIN) & (m2 <= self.M2_MAX) & (m2 <= m1))
        with np.errstate(divide="ignore", invalid="ignore"):
            q  = m2 / m1
            pq = trunc_gaussian(q, p["mu_NS"], p["sigma_NS"],
                                self.NS_MIN / m1, np.minimum(p["m_max_NS"] / m1, 1.0))
            out = self._p_m1(m1) / self._norm_m1 * pq / m1
        return np.where(ok, out, 0.0)


# ==========================================================
# Power law + peak (GWTC-2/3 BBH)
# ==========================================================

class PowerLawPlusPeakModel(MassModelBase):
    """
    Power law + peak (Talbot & Thrane 2018), as GWFAST's
    PowerLawPlusPeak_MassDistribution:
      p(m1)      ∝ [(1-lambda_peak) PL(m1) + lambda_peak N(m1; mu_m, sigma_m)]
                   x taper(m1; m_min, delta_m),  m1 <= m_max
      p(m2 | m1) ∝ m2^beta_q taper(m2; m_min, delta_m),  m2 in [m_min, m1]
    PL ∝ m1^-alpha_m is normalised on [m_min, m_max] and the Gaussian on the
    real line, so lambda_peak is the fraction in the peak.  p(m2 | m1) is the
    q^beta_q of the LVK papers, normalised per m1.  gwpopulation's
    SinglePeakSmoothedMassDistribution (used by popstock) instead truncates the
    Gaussian to [m_min, 100] and does not cut it at m_max; the two agree when
    the peak lies well inside [m_min, m_max] (the defaults, close to GWTC-3).
    """

    DEFAULTS = {"alpha_m": 3.4, "beta_q": 1.1, "m_min": 5.1, "m_max": 87.0,
                "lambda_peak": 0.039, "delta_m": 4.8, "mu_m": 34.0, "sigma_m": 3.6}

    def __init__(self, params=None, n_m2_grid=20000):
        p = dict(self.DEFAULTS)
        p.update(params or {})
        self._p = p
        self.M1_MIN = self.M2_MIN = p["m_min"]
        self.M_MAX  = p["m_max"]
        self._pl_norm = _pl_integral(p["alpha_m"], p["m_min"], p["m_max"])

        # int_{m_min}^{m1} p_m2 dm2, tabulated once
        self._m2_grid = np.geomspace(self.M2_MIN, self.M_MAX, n_m2_grid)
        self._m2_cum  = cumulative_trapezoid(self._p_m2(self._m2_grid),
                                             self._m2_grid, initial=0.0)
        self._norm_m1 = _normalise_1d(self._p_m1, self.M1_MIN, self.M_MAX)

    def _p_m1(self, m1):
        p  = self._p
        m1 = np.asarray(m1, dtype=float)
        pl = np.where((m1 >= p["m_min"]) & (m1 <= p["m_max"]),
                      m1 ** (-p["alpha_m"]) / self._pl_norm, 0.0)
        g  = (np.exp(-0.5 * ((m1 - p["mu_m"]) / p["sigma_m"]) ** 2)
              / (p["sigma_m"] * np.sqrt(2 * np.pi)))
        mix = np.where(m1 <= p["m_max"], (1 - p["lambda_peak"]) * pl + p["lambda_peak"] * g, 0.0)
        return mix * planck_taper(m1, p["m_min"], p["delta_m"])

    def _p_m2(self, m2):
        p = self._p
        return np.asarray(m2, float) ** p["beta_q"] * planck_taper(m2, p["m_min"], p["delta_m"])

    def density(self, m1, m2):
        m1, m2 = np.broadcast_arrays(np.asarray(m1, float), np.asarray(m2, float))
        norm_m2 = np.interp(m1, self._m2_grid, self._m2_cum, left=0.0)
        ok = (m2 <= m1) & (m2 >= self.M2_MIN) & (norm_m2 > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = self._p_m1(m1) * self._p_m2(m2) / norm_m2
        return np.where(ok, out, 0.0) / self._norm_m1


_REGISTRY = {
    "bpl_double_gaussian": BrokenPowerLawDoubleGaussianModel,
    "uniform":             UniformModel,
    "truncated_powerlaw":  TruncatedPowerLawModel,
    "nsbh":                NSBHModel,
    "nsbh_powerlaw_gaussian": NSBHPowerLawGaussianModel,
    "power_law_plus_peak": PowerLawPlusPeakModel,
}


def get_mass_model(name="bpl_double_gaussian", **kwargs):
    """
    Usage:
        get_mass_model("bpl_double_gaussian", params={"alpha_m_1": 2.0})
        get_mass_model("uniform", m1_min=1.0, m2_min=1.0, m_max=2.5)
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown mass model '{name}'; available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
