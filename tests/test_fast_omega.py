# Tests for fast_omega.FastOmega (JAX).  Skipped without jax or lalsuite.

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("lalsimulation")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import stochaxtic as oi  # noqa: E402
from stochaxtic import jax_populations as jp  # noqa: E402
from stochaxtic.fast_omega import FastOmega  # noqa: E402

FREQS = np.array([10.0, 30.0, 100.0, 300.0, 1000.0])


@pytest.fixture(scope="module")
def fo():
    return FastOmega(freqs=FREQS, cache_dir=None, verbose=False)


@pytest.mark.parametrize("mass, rate, R0", [
    ("bpl_double_gaussian", "madau_dickinson", 15.9),
    ("uniform", "madau_dickinson_timedelay", 23.4),
    ("nsbh_modsmooth", "madau_dickinson_timedelay", 15.9),
])
def test_matches_monte_carlo(fo, mass, rate, R0):
    mc = oi.OmegaIntegrator(mass, rate, rate_kwargs={"R0": R0}, freqs=FREQS,
                            verbose=False).run(n_rep=8, log2_points=12)
    fast = np.asarray(fo.make(mass, rate)({}, {"R0": R0}))
    tol = 1e-3 + 4 * mc.omega_err / mc.omega
    assert np.all(np.abs(fast / mc.omega - 1) < tol)


def test_time_delay_rate_matches_numpy():
    z  = np.expm1(np.linspace(np.log1p(0.2), np.log1p(9.5), 200))
    p  = {**jp.RATE_DEFAULTS["madau_dickinson_timedelay"], "R0": 23.4, "td_min": 0.05}
    Rj = np.asarray(jp.TimeDelayedMadauDickinson(z, oi.COSMO)(z, p))
    Rn = oi.get_rate_model("madau_dickinson_timedelay", R0=23.4, td_min=0.05).merger_rate(z)
    np.testing.assert_allclose(Rj, Rn, rtol=1e-4)


@pytest.mark.parametrize("mass, rate, key, value, rel_h", [
    ("bpl_double_gaussian", "madau_dickinson", "alpha_m_1", 1.6, 1e-3),
    ("bpl_double_gaussian", "madau_dickinson", "m_max", 120.0, 1e-2),     # soft edge
    ("uniform", "madau_dickinson_timedelay", "m_max", 2.2, 1e-2),         # soft edge
    ("uniform", "madau_dickinson_timedelay", "td_min", 0.03, 1e-3),
])
def test_gradients_match_finite_differences(fo, mass, rate, key, value, rel_h):
    f = fo.make(mass, rate)
    in_mass = key in jp.MASS_MODELS[mass][1]

    def L(x):
        mp, rp = ({key: x}, {}) if in_mass else ({}, {key: x})
        return jnp.sum(jnp.log(f(mp, rp)))

    h  = rel_h * value
    fd = (L(value + h) - L(value - h)) / (2 * h)
    assert float(jax.grad(L)(value)) == pytest.approx(float(fd), rel=5e-3)


def test_full_table_and_low_rank_agree():
    kw = dict(freqs=FREQS, cache_dir=None, verbose=False)
    a = FastOmega(rank=40, **kw).make("bpl_double_gaussian", "madau_dickinson")({}, {})
    b = FastOmega(rank=12, **kw).make("bpl_double_gaussian", "madau_dickinson")({}, {})
    np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-4)


# ----------------------------------------------------------------------
# spins and tides
# ----------------------------------------------------------------------

from stochaxtic.orientation import OrientationAverager, ensure_lal_data_path  # noqa: E402

T_SUN = 4.925490947641267e-06


def _rom(name):
    """skip unless the ROM data file (see README) is on LAL_DATA_PATH"""
    import os
    ensure_lal_data_path()
    have = any(os.path.isfile(os.path.join(d, name))
               for d in os.environ.get("LAL_DATA_PATH", "").split(":") if d)
    return pytest.mark.skipif(not have, reason=f"{name} not on LAL_DATA_PATH")


@pytest.mark.parametrize("approximant, M, q, spins, lambdas", [
    ("IMRPhenomXAS", 20.0, 0.5, ((0, 0, 0.4), (0, 0, -0.2)), None),
    ("IMRPhenomXP", 20.0, 0.5, ((0.3, 0.1, 0.2), (0.0, 0.2, -0.1)), None),
    ("IMRPhenomXPHM", 20.0, 0.3, ((0.3, 0.1, 0.2), (0.0, 0.2, -0.1)), None),
    ("IMRPhenomXP_NRTidalv2", 2.8, 0.8, ((0.02, 0.01, 0.03), (0.0, 0.02, -0.01)), (400.0, 600.0)),
    pytest.param("SEOBNRv5_ROM", 40.0, 0.4, ((0, 0, 0.3), (0, 0, -0.2)), None, marks=_rom("SEOBNRv5ROM_v1.0.hdf5")),
    pytest.param("SEOBNRv5HM_ROM", 40.0, 0.2, ((0, 0, 0.3), (0, 0, -0.2)), None, marks=_rom("SEOBNRv5HMROM_v1.0.hdf5")),
])
def test_orientation_average_is_exact(approximant, M, q, spins, lambdas):
    x = np.geomspace(1e-4, 0.2, 60)
    f = x / (M * T_SUN)
    m1, m2 = M / (1 + q), M * q / (1 + q)
    kw = dict(f_ref=0.005 / (M * T_SUN), lambdas=lambdas)
    fast  = OrientationAverager(approximant)(f, m1, m2, *spins, **kw)
    dense = OrientationAverager(approximant, rule=(12, 16))(f, m1, m2, *spins, **kw)
    ok = dense > 1e-8 * dense.max()
    np.testing.assert_allclose(fast[ok], dense[ok], rtol=1e-6)


def test_aligned_spins_match_monte_carlo():
    freqs = np.array([30.0, 100.0, 300.0])
    sp = {"alpha_chi": 1.6, "beta_chi": 4.1, "zeta": 0.66, "sigma_t": 1.5}
    mc = oi.OmegaIntegrator("bpl_double_gaussian", "madau_dickinson", rate_kwargs={"R0": 15.9},
                            freqs=freqs, spins="aligned", spin_params=sp,
                            verbose=False).run(n_rep=8, log2_points=12)
    fo = FastOmega(freqs=freqs, waveform="IMRPhenomXAS", spins="aligned", n_spin=64,
                   cache_dir=None, verbose=False)
    fast = np.asarray(fo.make("bpl_double_gaussian", "madau_dickinson")({}, {"R0": 15.9}, sp))
    assert np.all(np.abs(fast / mc.omega - 1) < 3e-3 + 4 * mc.omega_err / mc.omega)


def test_gwtc5_default_spins_match_monte_carlo():
    """GWTC-5 Default BBH spin model (truncated-Gaussian magnitudes, free tilt peak) vs MC."""
    freqs = np.array([30.0, 100.0, 300.0])
    sp = {"mu_chi": 0.2, "sigma_chi": 0.3, "chi_max": 1.0, "mu_t": 0.3, "sigma_t": 0.8, "zeta": 0.4}
    mc = oi.OmegaIntegrator("bpl_double_gaussian", "madau_dickinson", rate_kwargs={"R0": 15.9},
                            freqs=freqs, spins="aligned", spin_model="gwtc5_default", spin_params=sp,
                            verbose=False).run(n_rep=8, log2_points=12)
    fo = FastOmega(freqs=freqs, waveform="IMRPhenomXAS", spins="aligned", n_spin=64,
                   cache_dir=None, verbose=False)
    om = fo.make("bpl_double_gaussian", "madau_dickinson", spin_model="gwtc5_default")
    fast = np.asarray(om({}, {"R0": 15.9}, sp))
    assert np.all(np.abs(fast / mc.omega - 1) < 3e-3 + 4 * mc.omega_err / mc.omega)
    g = jax.grad(lambda m: jnp.sum(jnp.log(om({}, {"R0": 15.9}, {**sp, "mu_t": m}))))(0.3)
    assert np.isfinite(float(g))


def test_eos_tides_match_brute_force():
    """Omega(tides)/Omega(no tides) against direct sampling of masses, z and Lambdas."""
    freqs = np.array([100.0, 300.0, 600.0])
    rng = np.random.default_rng(3)
    N = 1500
    rm = oi.get_rate_model("madau_dickinson_timedelay", R0=23.4)
    zg = np.expm1(np.linspace(np.log1p(0.2), np.log1p(10.0), 4000))
    cdf = np.cumsum(rm.event_rate_density(zg) * (1 + zg))
    z = np.interp(rng.random(N), cdf / cdf[-1], zg)
    dL = oi.COSMO.luminosity_distance(z)
    m = rng.uniform(1.0, 2.0, (N, 2))
    m1, m2 = m.max(1), m.min(1)
    L = 400.0 * (np.stack([m1, m2], 1) / 1.4) ** -6.0
    avg = OrientationAverager("IMRPhenomXAS_NRTidalv2")
    It = sum(avg(freqs, m1[i] * (1 + z[i]), m2[i] * (1 + z[i]), lambdas=tuple(L[i])) / dL[i] ** 2
             for i in range(N))
    I0 = sum(avg(freqs, m1[i] * (1 + z[i]), m2[i] * (1 + z[i]), lambdas=(0.0, 0.0)) / dL[i] ** 2
             for i in range(N))
    fo = FastOmega(freqs=freqs, waveform="IMRPhenomXAS_NRTidalv2", tides=True,
                   m_tot_range=(1.8, 6.0), q_min=0.3, n_q=41, cache_dir=None, verbose=False)
    f = fo.make("uniform", "madau_dickinson_timedelay")
    tp = {"Lambda_14": 400.0, "b1": -6.0, "b2": 0.0}
    ratio_fast = np.asarray(f({}, {}, None, tp)) / np.asarray(f({}, {}, None, {"Lambda_14": 1e-8}))
    np.testing.assert_allclose(ratio_fast, It / I0, rtol=1e-2, atol=2e-4)


def test_spin_and_tide_gradients_match_finite_differences():
    freqs = np.array([50.0, 200.0, 600.0])
    fo = FastOmega(freqs=freqs, waveform="IMRPhenomXAS_NRTidalv2", tides=True, spins="aligned",
                   a_max=0.05, n_spin=32, m_tot_range=(1.8, 6.0), q_min=0.3, n_q=21,
                   n_kappa=16, cache_dir=None, verbose=False)
    f = fo.make("uniform", "madau_dickinson_timedelay")
    for group, key, value in [("spin", "alpha_chi", 1.6), ("spin", "zeta", 0.5),
                              ("tide", "Lambda_14", 400.0), ("tide", "b2", -3.0)]:
        def L(v):
            sp = {key: v} if group == "spin" else {}
            tp = {key: v} if group == "tide" else {}
            return jnp.sum(jnp.log(f({}, {}, sp, tp)))
        h  = 1e-3 * value
        fd = (L(value + h) - L(value - h)) / (2 * h)
        assert float(jax.grad(L)(value)) == pytest.approx(float(fd), rel=1e-2, abs=1e-8)


def test_check_support_flags_populations_outside_the_grid():
    kw = dict(freqs=np.array([20.0, 200.0]), n_q=21, cache_dir=None, verbose=False)
    fo = FastOmega(m_tot_range=(1.8, 6.0), q_min=0.3, **kw)
    assert fo.check_support("uniform")["total"] < 1e-6
    with pytest.warns(UserWarning, match="outside the grid"):
        r = fo.check_support("uniform", {"m_max": 3.5})
    assert r["above_M"] > 0.05
    fo = FastOmega(m_tot_range=(15.0, 700.0), **kw)          # ~9% of the default BBH below M = 15
    with pytest.warns(UserWarning, match="M < 15"):
        fo.make("bpl_double_gaussian", "madau_dickinson")
