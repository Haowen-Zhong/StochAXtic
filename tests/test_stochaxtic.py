# Tests for stochaxtic.  Run with:  pytest -q tests
#
# test_regression_against_mogwas compares replica by replica with the
# original MOGWAS all-frequency code in ../CC_moments_related when it is
# present (skipped otherwise).

import glob
import os
import subprocess
import sys

import numpy as np
import pytest
from astropy import units as u
from astropy.cosmology import Planck18

import stochaxtic as oi
from stochaxtic.mass_models import _REGISTRY as MASS_MODELS

HERE   = os.path.dirname(os.path.abspath(__file__))
MOGWAS = os.path.join(HERE, "..", "..", "CC_moments_related")


def _has_lal():
    try:
        import lalsimulation  # noqa: F401
        return True
    except ImportError:
        return False


# ----------------------------------------------------------------------
# populations
# ----------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(MASS_MODELS))
def test_mass_models_are_normalised(name):
    assert oi.get_mass_model(name).check_normalisation(n=6000) == pytest.approx(1.0, abs=2e-3)


def test_uniform_density_value():
    # uniform on the triangle 1 <= m2 <= m1 <= 2 (area 1/2)
    assert oi.get_mass_model("uniform").density(1.8, 1.2) == pytest.approx(2.0)


def test_density_mc_q_jacobian():
    # int p(mc, q) dmc dq over the (mc, q) box equals int p(m1, m2) dm1 dm2
    mm = oi.get_mass_model("nsbh_modsmooth")
    mc = np.linspace(mm.mc_min, mm.mc_max, 3000)
    q  = np.geomspace(mm.q_min, 1.0, 3000)
    P  = mm.density_mc_q(mc[:, None], q[None, :])
    assert np.trapezoid(np.trapezoid(P, q, axis=1), mc) == pytest.approx(1.0, abs=3e-3)


# ----------------------------------------------------------------------
# cosmology and rates
# ----------------------------------------------------------------------

def test_cosmology_matches_astropy():
    z = np.array([0.01, 0.2, 1.0, 3.0, 10.0])
    c = oi.COSMO
    np.testing.assert_allclose(c.luminosity_distance(z),
                               Planck18.luminosity_distance(z).to_value(u.Mpc), rtol=1e-6)
    np.testing.assert_allclose(c.dVc_dz(z),
                               4 * np.pi * Planck18.differential_comoving_volume(z)
                               .to_value(u.Mpc ** 3 / u.sr), rtol=1e-6)
    np.testing.assert_allclose(c.cosmic_time(z), Planck18.age(z).to_value(u.Gyr), rtol=1e-5)


def test_time_delay_rate_normalised_to_R0():
    rm = oi.get_rate_model("madau_dickinson_timedelay", R0=23.4)
    assert rm.merger_rate(0.0) == pytest.approx(23.4, rel=1e-3)
    # formation stops at zf_max = 10 and td >= 20 Myr, so R vanishes for z >~ 9.7
    assert np.all(rm.merger_rate(np.linspace(0, 9.5, 50)) > 0)
    assert rm.merger_rate(9.9) == 0.0


def test_kernel_total_rate():
    rm = oi.get_rate_model("madau_dickinson", R0=15.9)
    k  = oi.RedshiftKernel(oi.get_mass_model("uniform"), rm, 0.2, 10.0, 1000)
    assert k.N_tot == pytest.approx(rm.total_rate(0.2, 10.0), rel=1e-5)


# ----------------------------------------------------------------------
# integrator
# ----------------------------------------------------------------------

needs_lal = pytest.mark.skipif(not _has_lal(), reason="lalsuite not installed")


@needs_lal
def test_run_save_load_merge(tmp_path):
    kw = dict(freqs=np.array([10.0, 100.0, 500.0]), verbose=False)
    oi_ = oi.OmegaIntegrator("uniform", "madau_dickinson", **kw)
    a = oi_.run(n_rep=2, log2_points=6, rep_start=0)
    b = oi_.run(n_rep=2, log2_points=6, rep_start=2)
    a.save(tmp_path / "a.npz")
    b.save(tmp_path / "b.npz")
    m   = oi.OmegaResult.merge([oi.OmegaResult.load(tmp_path / "a.npz"),
                                oi.OmegaResult.load(tmp_path / "b.npz")])
    ref = oi_.run(n_rep=4, log2_points=6, rep_start=0)
    np.testing.assert_allclose(m.omega, ref.omega, rtol=1e-12)
    np.testing.assert_allclose(m.omega_err, ref.omega_err, rtol=1e-12)
    assert np.all(m.omega > 0)


@needs_lal
@pytest.mark.skipif(not os.path.isdir(MOGWAS), reason="MOGWAS code not found")
def test_regression_against_mogwas(tmp_path):
    """
    Same seed and proposal -> the same samples, so every replica must agree
    with the MOGWAS all-frequency code up to the constants that changed:
    c = 299792.458 (was 3e5) and the Julian year (was 3.156e7 s).
    """
    args = ["--n-max", "1", "--terms", "mean", "--tilts", "-3", "0", "4", "--eps", "0.1",
            "--iota-power", "2", "--log2-points", "9", "--n-rep", "2", "--R0", "15.9"]
    subprocess.run([sys.executable, "sgwb_worker_allfreq.py", str(tmp_path), *args],
                   cwd=MOGWAS, check=True, capture_output=True)
    sys.path.insert(0, MOGWAS)
    try:
        from mass_models import get_mass_model as old_mass_model
        from mass_models_vec import get_vectorized_density
        from integrand_allfreq import to_kappa
    finally:
        sys.path.remove(MOGWAS)

    old = [np.load(f) for f in sorted(glob.glob(str(tmp_path / "allfreq_rep*.npz")))]
    old_omega = np.array([to_kappa(d["mean_scaled_mean"][0] * d["S_REF"], d["freqs"],
                                   d["G1_I"], float(d["R_tot"]), 1)[0] for d in old])

    new = oi.OmegaIntegrator(
        "bpl_double_gaussian", "madau_dickinson", rate_kwargs={"R0": 15.9},
        n_z_proposal=1000, verbose=False,
        proposal_kwargs=dict(eps=0.1, tilts=(-3, 0, 4), iota_power=2),
    ).run(n_rep=2, log2_points=9, antithetic=True)

    m1, m2 = np.array([30.0]), np.array([20.0])
    norm = (oi.get_mass_model("bpl_double_gaussian").density(m1, m2)
            / get_vectorized_density(old_mass_model("bpl_double_gaussian"))(m1, m2))[0]
    expected = (299792.458 / 3e5) * (3.156e7 / (365.25 * 86400)) * norm
    # rtol 1e-5, not 1e-9: MOGWAS zeroes whole waveforms with M_det > 3*4400/f_min
    # (~2600 Msun), which still ring down in band; this package keeps them
    # (a ~5e-6 effect at 5 Hz)
    np.testing.assert_allclose(new.replicas / old_omega, expected, rtol=1e-5)
