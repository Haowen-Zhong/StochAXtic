# numpy spin models for OmegaIntegrator; FastOmega uses the same densities from
# jax_populations.py.  A model is a density over (a1, a2, cos t1, cos t2), magnitudes
# on [0, a_max]; the in-plane azimuth difference phi12 is uniform.

import numpy as np
from scipy.special import erf
from scipy.stats import beta as _beta

from .mass_models import trunc_gaussian as _truncnorm_pdf    # tail-stable (in logs)


def gwtc3_default_spin(a1, a2, c1, c2, p, a_max=1.0):
    """LVK "Default" model: a/a_max ~ Beta(alpha_chi, beta_chi) iid;
    p(c1, c2) = zeta N(c1) N(c2) + (1 - zeta)/4, N = Gaussian(1, sigma_t) on [-1, 1]."""
    am1, am2 = (a_max, a_max) if np.ndim(a_max) == 0 else a_max      # per-component bounds
    pa = (_beta.pdf(a1 / am1, p["alpha_chi"], p["beta_chi"])
          * _beta.pdf(a2 / am2, p["alpha_chi"], p["beta_chi"])) / (am1 * am2)
    s  = p["sigma_t"]
    Zt = 0.5 - 0.5 * (1.0 + erf(-2.0 / (s * np.sqrt(2.0))))
    g  = lambda c: np.exp(-0.5 * ((c - 1.0) / s) ** 2) / (s * np.sqrt(2 * np.pi)) / Zt
    return pa * (p["zeta"] * g(c1) * g(c2) + (1.0 - p["zeta"]) / 4.0)


def gwtc5_default_spin(a1, a2, c1, c2, p, a_max=1.0):
    """GWTC-4.0/5.0 "Default BBH": a_i ~ Normal(mu_chi, sigma_chi) on [0, chi_max] iid;
    p(c1, c2) = zeta N(c1) N(c2) + (1 - zeta)/4, N = Normal(mu_t, sigma_t) on [-1, 1]."""
    pa = (_truncnorm_pdf(a1, p["mu_chi"], p["sigma_chi"], 0.0, p["chi_max"])
          * _truncnorm_pdf(a2, p["mu_chi"], p["sigma_chi"], 0.0, p["chi_max"]))
    nt = lambda c: _truncnorm_pdf(c, p["mu_t"], p["sigma_t"], -1.0, 1.0)
    return pa * (p["zeta"] * nt(c1) * nt(c2) + (1.0 - p["zeta"]) / 4.0)


def uniform_isotropic_spin(a1, a2, c1, c2, p, a_max=1.0):
    """Neutron stars (BNS), no hyperparameters: a_i uniform on [0, a_max] iid (a_max = 0.05:
    the LVK "low-spin" prior); cos t1, cos t2 uniform on [-1, 1] (isotropic)."""
    am1, am2 = (a_max, a_max) if np.ndim(a_max) == 0 else a_max      # per-component bounds
    return np.full(np.broadcast_shapes(*(np.shape(v) for v in (a1, a2, c1, c2))), 0.25 / (am1 * am2))


SPIN_MODELS = {
    # defaults: GWTC-3 / GWTC-5 medians (jax_populations.SPIN_MODELS)
    "gwtc3_default": (gwtc3_default_spin, dict(alpha_chi=1.64, beta_chi=4.15, zeta=0.656, sigma_t=1.54)),
    "gwtc5_default": (gwtc5_default_spin, dict(mu_chi=0.0633, sigma_chi=0.365, chi_max=1.0,
                                               mu_t=0.319, sigma_t=1.11, zeta=0.671)),
    "uniform_isotropic": (uniform_isotropic_spin, {}),          # neutron stars
}
SPIN_MODELS["lvk_default"] = SPIN_MODELS["gwtc3_default"]       # former name
