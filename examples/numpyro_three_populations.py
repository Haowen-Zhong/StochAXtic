# numpyro_three_populations.py
# NUTS inference of BBH + BNS + NSBH background hyperparameters with FastOmega.
#
#   BBH : bpl_double_gaussian (broken power law + 2 peaks) + madau_dickinson
#   BNS : uniform                                            + madau_dickinson_timedelay
#   NSBH: nsbh_modsmooth                                     + madau_dickinson_timedelay
#
# Free parameters: BBH local rate and redshift evolution (R0, zp, alpha), BNS and
# NSBH local rates.  Masses are fixed (in practice: from the CBC catalogues).
# The noise curve is a TOY (bucket shape rescaled to a chosen SNR); replace sigma
# with a real sensitivity (e.g. from pygwb: PSDs + overlap reduction function).
#
# Speed (Apple M1): one BBH value+gradient ~3.4 ms.  The posterior is not
# Gaussian (zp and alpha are nearly degenerate: the background measures mainly one
# combination), so NUTS needs ~100 leapfrog steps per iteration; three things help:
#   - R0 is a linear factor: Omega = R0 * Omega(R0=1).  Populations whose shape is
#     fixed (here BNS, NSBH) become precomputed templates; BBH is evaluated once
#     with R0=1 per step.                                         (~2-3x)
#   - dense_mass=True learns the parameter correlations.           (~2x)
#   - parallel chains on several cores (set_host_device_count before any JAX work).
#
# Run in the sgwb environment:  python numpyro_three_populations.py
# In Jupyter, copy the body of main() into cells (no __main__ guard needed there);
# numpyro.set_host_device_count must then run in the first cell, before JAX starts.

import time

import numpy as np
import numpyro
numpyro.enable_x64()                       # FastOmega needs float64: call this first
N_CHAINS = 4
numpyro.set_host_device_count(N_CHAINS)    # one CPU "device" per chain (before JAX starts)
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from stochaxtic.fast_omega import FastOmega


def main():
    # ------------------------------------------------------------------
    # 1. Omega_GW functions: one table (IMRPhenomXAS, no spins) for all three
    # ------------------------------------------------------------------
    f = np.geomspace(20, 1000, 40)                        # frequency bins [Hz]
    fo = FastOmega(freqs=f, verbose=False)                # table cached in ~/.cache/stochaxtic
    om_bbh  = fo.make("bpl_double_gaussian", "madau_dickinson")
    om_bns  = fo.make("uniform", "madau_dickinson_timedelay")
    om_nsbh = fo.make("nsbh_modsmooth", "madau_dickinson_timedelay")
    # each om_*(mass_params, rate_params) returns Omega_GW(f); parameters not given take
    # their defaults (jax_populations.MASS_MODELS / RATE_DEFAULTS)

    # Omega is linear in R0: shapes of BNS and NSBH are fixed here -> templates at R0 = 1
    T_bns  = np.asarray(om_bns({}, {"R0": 1.0}))
    T_nsbh = np.asarray(om_nsbh({}, {"R0": 1.0}))

    def omega_total(p):
        return (p["R0_bbh"] * om_bbh({}, {"R0": 1.0, "zp": p["zp_bbh"], "alpha": p["alpha_bbh"]})
                + p["R0_bns"] * T_bns + p["R0_nsbh"] * T_nsbh)

    # ------------------------------------------------------------------
    # 2. Mock data: truth + toy Gaussian noise
    # ------------------------------------------------------------------
    truth = dict(R0_bbh=18.0, zp_bbh=2.27, alpha_bbh=2.85, R0_bns=100.0, R0_nsbh=30.0)
    O_true = np.asarray(omega_total(truth))

    shape = (f / 30.0) ** -3 + 1.0 + (f / 30.0) ** 2.5    # toy bucket, minimum near 30 Hz
    snr_target = 50.0
    sigma = shape * np.sqrt(np.sum((O_true / shape) ** 2)) / snr_target
    data = O_true + sigma * np.random.default_rng(1).standard_normal(len(f))

    for name, om, R0 in [("BBH", om_bbh, 18.0), ("BNS", om_bns, 100.0), ("NSBH", om_nsbh, 30.0)]:
        snr = np.sqrt(np.sum((np.asarray(om({}, {"R0": R0})) / sigma) ** 2))
        print(f"{name:5s} SNR = {snr:.1f}")      # BBH ~44, BNS ~3, NSBH ~3

    # ------------------------------------------------------------------
    # 3. numpyro model
    # ------------------------------------------------------------------
    def model(data, catalog=False):
        lnR_bbh  = numpyro.sample("lnR0_bbh",  dist.Uniform(np.log(1), np.log(200)))
        zp       = numpyro.sample("zp_bbh",    dist.Uniform(0.5, 5.0))
        alpha    = numpyro.sample("alpha_bbh", dist.Uniform(0.0, 6.0))
        lnR_bns  = numpyro.sample("lnR0_bns",  dist.Uniform(np.log(5), np.log(2000)))
        lnR_nsbh = numpyro.sample("lnR0_nsbh", dist.Uniform(np.log(1), np.log(500)))
        p = dict(R0_bbh=jnp.exp(lnR_bbh), zp_bbh=zp, alpha_bbh=alpha,
                 R0_bns=jnp.exp(lnR_bns), R0_nsbh=jnp.exp(lnR_nsbh))
        if catalog:
            # stand-in for the resolved-event catalogues: local rates known to
            # ~25% (BBH) and ~60% (BNS, NSBH)
            numpyro.factor("cat_bbh",  dist.Normal(np.log(18.0), 0.25).log_prob(lnR_bbh))
            numpyro.factor("cat_bns",  dist.Normal(np.log(100.0), 0.6).log_prob(lnR_bns))
            numpyro.factor("cat_nsbh", dist.Normal(np.log(30.0), 0.6).log_prob(lnR_nsbh))
        numpyro.sample("obs", dist.Normal(omega_total(p), sigma), obs=data)

    # ------------------------------------------------------------------
    # 4. NUTS: background only, then background + catalogue information
    # ------------------------------------------------------------------
    for catalog in (False, True):
        t = time.time()
        mcmc = MCMC(NUTS(model, target_accept_prob=0.9, dense_mass=True),
                    num_warmup=500, num_samples=1000, num_chains=N_CHAINS,
                    chain_method="parallel", progress_bar=False)
        mcmc.run(jax.random.PRNGKey(0), data, catalog, extra_fields=("num_steps",))
        print("mean leapfrog steps per iteration:",
              float(np.mean(mcmc.get_extra_fields()["num_steps"])))
        print(f"\n=== catalog = {catalog}: {time.time() - t:.0f} s ===")
        mcmc.print_summary()                               # check r_hat ~ 1 and n_eff
        samples = mcmc.get_samples()
        np.savez(f"samples_catalog{int(catalog)}.npz", **{k: np.asarray(v) for k, v in samples.items()})


if __name__ == "__main__":
    main()
