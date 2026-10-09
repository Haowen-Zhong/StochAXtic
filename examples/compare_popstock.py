# Compare FastOmega with popstock (Renzini et al.) for a few standard waveforms.
#
# Population (no spins): gwpopulation power_law_primary_mass_ratio
#   p(m1) ∝ m1^-alpha on [mmin, mmax], p(q|m1) ∝ q^beta on [mmin/m1, 1]
# and the Madau-Dickinson redshift model of gwpopulation (psi(0) = 1), z in [Z_MIN, 10].
#
# Why Z_MIN = 0.2: popstock's estimator averages E_i ∝ 1/dL^2 over sources drawn
# from p(z) ∝ z^2 near z = 0, so its variance diverges like 1/z_min (relative std per
# sample 196 for z_min = 1e-4, its default grid; 3.2 for 0.2).  1% accuracy would need
# ~4e8 samples down to 1e-4, ~1e5 from 0.2.  Both codes use [0.2, 10]; popstock gets
# the matching Rate_norm explicitly.
#
# popstock: Monte Carlo over sources drawn from the population, one random
# orientation each, energies from bilby/LAL on a 0.1 Hz grid.  The same source
# samples and orientations are used for every waveform (common random numbers).
# FastOmega: the same population as a JAX model registered at run time.
#
# Conventions aligned on the FastOmega side:
#   * gwpopulation normalises psi(0) = 1  ->  R0_FastOmega = rate * (1 + (1+z_p)^-kappa)
#   * popstock: dL from astropy Planck15, dVc/dz from wcosmo Planck15 (no radiation,
#     up to 0.8% different at z ~ 9), H0 of Planck15, 1 yr = 365 d.
#     FastOmega uses Cosmology(Planck15); its z weights are multiplied by
#     dVc_wcosmo / dVc_astropy, and Omega by (H0_P18/H0_P15)^2 * 365.25/365.
#
# Run with the sgwb python (the __main__ guard is needed on macOS).
import os
import time
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_popstock")
WAVEFORMS = ["IMRPhenomD", "IMRPhenomXAS", "IMRPhenomXHM"]
N_SAMPLES = 100000
Z_MIN = 0.2
SEED = 12345
FID = dict(alpha=3.4, beta=1.1, mmin=5.0, mmax=80.0,         # mass
           gamma=2.7, kappa=5.6, z_peak=1.9, rate=20.0)      # redshift, R(0) in Gpc^-3 yr^-1
FREQS = np.unique(np.round(np.geomspace(12.0, 2000.0, 160)))   # popstock interpolates to these


class PowerLawMassRatio:
    """gwpopulation power_law_primary_mass_ratio with the p_q attribute popstock asserts."""
    variable_names = ["alpha", "beta", "mmin", "mmax"]

    def __call__(self, dataset, alpha, beta, mmin, mmax):
        from gwpopulation.models.mass import power_law_primary_mass_ratio
        return power_law_primary_mass_ratio(dataset, alpha=alpha, beta=beta, mmin=mmin, mmax=mmax)

    def p_q(self, dataset, beta, mmin):          # only used by popstock's alternative samplers
        q, m1 = dataset["mass_ratio"], dataset["mass_1"]
        lo = mmin / m1
        return np.where((q >= lo) & (q <= 1), q ** beta * (1 + beta) / (1 - lo ** (1 + beta)), 0.0)


def run_popstock():
    from gwpopulation.models.redshift import MadauDickinsonRedshift
    from popstock.PopulationOmegaGW import PopulationOmegaGW
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    models = {"mass_model": PowerLawMassRatio(), "redshift_model": MadauDickinsonRedshift(z_max=10.0)}
    pop = PopulationOmegaGW(models=models, frequency_array=FREQS)
    # popstock's rejection sampler proposes q ~ q^-1.1 on [1e-4, 1]; almost all of it lies
    # below mmin/m1, so its efficiency falls under its threshold and it falls back to a
    # sampler that only knows other mass models.  Draw the masses exactly instead
    # (inverse CDFs) and hand them to popstock's own set_proposal_samples; redshifts
    # come from popstock's own sampler.
    a, b, lo, hi = FID["alpha"], FID["beta"], FID["mmin"], FID["mmax"]
    u = rng.uniform(size=N_SAMPLES)
    m1 = (lo ** (1 - a) + u * (hi ** (1 - a) - lo ** (1 - a))) ** (1 / (1 - a))
    qlo = lo / m1
    u = rng.uniform(size=N_SAMPLES)
    q = (qlo ** (1 + b) + u * (1 - qlo ** (1 + b))) ** (1 / (1 + b))
    z_grid = np.geomspace(Z_MIN, 10.0, 10000)
    samples = dict(mass_1_source=m1, mass_ratio=q)
    samples.update(pop.draw_redshift_proposal_samples(z_grid, FID, N_SAMPLES))
    pop.fiducial_parameters = FID.copy()
    pop.set_proposal_samples(proposal_samples=pop.calculate_pdraws(samples, FID))
    n = pop.N_proposal_samples
    pop.proposal_samples["theta_jn"] = np.arccos(rng.uniform(-1, 1, n))   # common orientations
    pop.proposal_samples["phase"] = rng.uniform(0, 2 * np.pi, n)
    # Rate_norm over [Z_MIN, 10]: popstock's default normalises over [0, z_max]
    zz = np.linspace(Z_MIN, 10.0, 200001)
    dv = np.asarray(models["redshift_model"].differential_spacetime_volume(
        {"redshift": zz}, **{k: FID[k] for k in ("gamma", "kappa", "z_peak")}))
    rate_norm = np.trapz(dv, zz) / 1e9 * FID["rate"] / (60 * 60 * 24 * 365)
    res = {}
    for wf in WAVEFORMS:
        pop.wave_energies_calculated = False
        t0 = time.time()
        pop.calculate_omega_gw(Lambda=None, Rate_norm=rate_norm, multiprocess=True,
                               waveform_approximant=wf)
        dt = time.time() - t0
        per = pop.omega_gw[None, :] * pop.wave_energies / np.mean(pop.wave_energies, axis=0)[None, :]
        res[wf] = dict(omega=np.asarray(pop.omega_gw), per_sample=np.nan_to_num(per), seconds=dt)
        print(f"popstock {wf}: {dt:.0f} s for {n} samples", flush=True)
    s = pop.proposal_samples
    samples = dict(m1=np.asarray(s["mass_1_source"]), q=np.asarray(s["mass_ratio"]),
                   z=np.asarray(s["redshift"]))
    return res, samples


def run_fastomega():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from astropy.cosmology import Planck15, Planck18
    from gwpopulation.models.redshift import MadauDickinsonRedshift
    from stochaxtic import jax_populations as jp
    from stochaxtic.cosmology import Cosmology
    from stochaxtic.fast_omega import FastOmega

    def pl_q(m1, m2, p):
        w = p.get("_w", 0.0)
        q = m2 / m1
        lo = p["mmin"] / m1
        pq = jp._safe_pow(q, p["beta"]) * (1 + p["beta"]) / (1 - jp._safe_pow(lo, 1 + p["beta"]))
        edges = (jp._between(m1, p["mmin"], p["mmax"], w) * jp._between(m2, lo=p["mmin"], w=w)
                 * jnp.where(m2 <= m1, 1.0, 0.0))
        return jp._safe_pow(m1, -p["alpha"]) * pq / m1 * edges          # p(m1) p(q|m1) dq/dm2
    jp.MASS_MODELS["pl_q"] = (pl_q, dict(alpha=FID["alpha"], beta=FID["beta"],
                                         mmin=FID["mmin"], mmax=FID["mmax"]))

    cosmo = Cosmology(Planck15)
    md = MadauDickinsonRedshift(z_max=10.0)
    rate = dict(R0=FID["rate"] * (1 + (1 + FID["z_peak"]) ** -FID["kappa"]),
                alpha=FID["gamma"], zp=FID["z_peak"], beta=FID["kappa"])
    fix = (Planck18.H0.value / Planck15.H0.value) ** 2 * 365.25 / 365.0
    res = {}
    for wf in WAVEFORMS:
        fo = FastOmega(FREQS, waveform=wf, z_min=Z_MIN, z_max=10.0, cosmo=cosmo, n_proc=4)
        out = {}
        for tag in ("consistent", "popstock_mixed"):
            if tag == "popstock_mixed":       # popstock's dVc/dz (wcosmo) with astropy dL
                ratio = np.asarray(md.dvc_dz(fo.z)) / cosmo.dVc_dz(fo.z)
                fo._zw = fo._zw * jnp.asarray(ratio)
            om = fo.make("pl_q", "madau_dickinson")
            om(None, rate).block_until_ready()
            t0 = time.time()
            val = np.asarray(om(None, rate).block_until_ready())
            out[tag] = val * fix
            out["call_ms"] = 1e3 * (time.time() - t0)
        out["setup_s"] = fo.info["setup_seconds"] + fo.info["table_seconds"]
        res[wf] = out
        print(f"FastOmega {wf}: setup {out['setup_s']:.1f} s, call {out['call_ms']:.1f} ms", flush=True)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    fast = run_fastomega()
    pops, samples = run_popstock()
    np.savez(os.path.join(OUT, "results.npz"), freqs=FREQS,
             **{f"fast_{k}_{wf}": fast[wf][k] for wf in WAVEFORMS for k in ("consistent", "popstock_mixed")},
             **{f"pop_{wf}": pops[wf]["omega"] for wf in WAVEFORMS},
             **{f"popsig_{wf}": pops[wf]["per_sample"].std(0) / np.sqrt(len(samples["z"])) for wf in WAVEFORMS},
             **{f"popratio_{wf}": pops[wf]["omega"] / pops["IMRPhenomD"]["omega"] for wf in WAVEFORMS},
             **{f"popratiosig_{wf}": (pops[wf]["per_sample"] / pops["IMRPhenomD"]["omega"]
                                      - pops["IMRPhenomD"]["per_sample"] * pops[wf]["omega"]
                                      / pops["IMRPhenomD"]["omega"] ** 2).std(0) / np.sqrt(len(samples["z"]))
                for wf in WAVEFORMS},
             **{f"time_pop_{wf}": pops[wf]["seconds"] for wf in WAVEFORMS},
             **{f"time_fast_{wf}": fast[wf]["call_ms"] for wf in WAVEFORMS},
             **{f"samples_{k}": v for k, v in samples.items()})
    print("saved", os.path.join(OUT, "results.npz"))


if __name__ == "__main__":
    main()
