# spin_models.py
# numpy spin population models for OmegaIntegrator (FastOmega uses the JAX
# versions in jax_populations.py; both define the same densities).
#
# A model is a density over (a1, a2, cos t1, cos t2) with magnitudes on
# [0, a_max]; the in-plane azimuth difference phi12 is uniform.

import numpy as np
from scipy.special import erf
from scipy.stats import beta as _beta


def lvk_default_spin(a1, a2, c1, c2, p, a_max=1.0):
    """LVK "Default" model: a/a_max ~ Beta(alpha_chi, beta_chi) iid;
    p(c1, c2) = zeta N(c1) N(c2) + (1 - zeta)/4, N = Gaussian(1, sigma_t) on [-1, 1]."""
    pa = (_beta.pdf(a1 / a_max, p["alpha_chi"], p["beta_chi"])
          * _beta.pdf(a2 / a_max, p["alpha_chi"], p["beta_chi"])) / a_max ** 2
    s  = p["sigma_t"]
    Zt = 0.5 - 0.5 * (1.0 + erf(-2.0 / (s * np.sqrt(2.0))))
    g  = lambda c: np.exp(-0.5 * ((c - 1.0) / s) ** 2) / (s * np.sqrt(2 * np.pi)) / Zt
    return pa * (p["zeta"] * g(c1) * g(c2) + (1.0 - p["zeta"]) / 4.0)


def _truncnorm_pdf(x, mu, sig, lo, hi):
    Z = 0.5 * (erf((hi - mu) / (sig * np.sqrt(2.0))) - erf((lo - mu) / (sig * np.sqrt(2.0))))
    g = np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi)) / Z
    return np.where((x >= lo) & (x <= hi), g, 0.0)


def gwtc5_default_spin(a1, a2, c1, c2, p, a_max=1.0):
    """GWTC-4.0/5.0 "Default BBH": a_i ~ Normal(mu_chi, sigma_chi) on [0, chi_max] iid;
    p(c1, c2) = zeta N(c1) N(c2) + (1 - zeta)/4, N = Normal(mu_t, sigma_t) on [-1, 1]."""
    pa = (_truncnorm_pdf(a1, p["mu_chi"], p["sigma_chi"], 0.0, p["chi_max"])
          * _truncnorm_pdf(a2, p["mu_chi"], p["sigma_chi"], 0.0, p["chi_max"]))
    nt = lambda c: _truncnorm_pdf(c, p["mu_t"], p["sigma_t"], -1.0, 1.0)
    return pa * (p["zeta"] * nt(c1) * nt(c2) + (1.0 - p["zeta"]) / 4.0)


SPIN_MODELS = {
    "lvk_default": (lvk_default_spin, dict(alpha_chi=1.6, beta_chi=4.1, zeta=0.66, sigma_t=1.5)),
    # provisional defaults, see jax_populations.SPIN_MODELS
    "gwtc5_default": (gwtc5_default_spin, dict(mu_chi=0.15, sigma_chi=0.32, chi_max=1.0,
                                               mu_t=0.3, sigma_t=0.8, zeta=0.4)),
}
