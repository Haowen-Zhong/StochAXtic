# StochAXtic

Fast $\Omega_{\rm GW}(f)$ of a compact-binary population: all frequencies from
one set of waveform calls, a few seconds per population on one core.

$$
\Omega_{\rm GW}(f) = \frac{4\pi^2 f^3}{3H_0^2}\int \mathrm{d}z\,\frac{\mathrm{d}N}{\mathrm{d}z\,\mathrm{d}t}
\Big\langle |\tilde h_+(f)|^2 + |\tilde h_\times(f)|^2 \Big\rangle_{m_1,m_2,\iota,\phi}
$$

Only the intensity of the waveform enters; no overlap reduction function or
Stokes decomposition is needed.  Two evaluators:

* **`FastOmega`** (JAX): differentiable, jit-compiled $\Omega_{\rm GW}(f;\Lambda)$
  in ~2 ms per call (~5 ms with gradients), for MCMC with numpyro.  Needs
  non-spinning waveforms (see below).
* **`OmegaIntegrator`** (numpy): quasi Monte Carlo reference, any waveform,
  ~10 s per population.  Extracted from the MOGWAS all-frequency integrator
  (`../CC_moments_related/integrand_allfreq.py`), restricted to $n=1$, and
  checked against it replica by replica (see Validation).

## Install

```bash
pip install -e ".[fast]"    # numpy, scipy, astropy, jax; lalsuite for waveforms
pytest -q                   # 28 tests, ~4 min
```

## FastOmega for numpyro

```python
import numpyro, numpyro.distributions as dist
numpyro.enable_x64()                              # FastOmega needs float64
from stochaxtic.fast_omega import FastOmega

fo    = FastOmega(freqs, waveform="IMRPhenomXAS",       # once, ~3 s
                  m_tot_range=(6, 700), q_min=0.01)     # must contain the prior's support
omega = fo.make("bpl_double_gaussian", "madau_dickinson")   # jitted function

def model():
    a1 = numpyro.sample("alpha_m_1", dist.Uniform(0.5, 3.0))
    R0 = numpyro.sample("R0", dist.Uniform(1.0, 50.0))
    numpyro.sample("obs", dist.Normal(omega({"alpha_m_1": a1}, {"R0": R0}), sigma), obs=data)
```

`omega(mass_params, rate_params)` takes dicts of the hyperparameters you
sample; all others keep the model defaults (`jax_populations.MASS_MODELS`,
`RATE_DEFAULTS`).  Every hyperparameter is differentiable, including the
mass cut-offs: hard edges are logistic steps one grid cell wide.

**How it works.**  Non-spinning GR waveforms are scale invariant,
$|h_+|^2+|h_\times|^2 = M^4 J(Mf;q)$ at fixed distance and orientation
(verified to $10^{-13}$), so a single table $J(x;q)$, averaged over
inclination and phase, serves every mass, redshift and frequency.  On
logarithmic grids with a common spacing the redshift integral becomes a
convolution in $\ln M$ and the frequency dependence a cross-correlation,
both done by FFT; a rank-12 factorisation of $J$ in $q$ cuts the FFT work
tenfold, and moving the inspiral power law $x^{-7/3}$ out of $J$ keeps the
FFT free of roundoff noise (gradients agree with finite differences to six
digits).  Only the population density on the $(\ln M,\ln q)$ grid and the
rate on the $z$ grid depend on $\Lambda$.

A NUTS run with two free parameters (40 frequencies, 800 iterations)
takes ~8 s including compilation.

**Accuracy** against `OmegaIntegrator` (16 replicas x $2^{13}$, 5-1000 Hz):

| population | max rel. difference | time per call (value / value+grad) |
|---|---|---|
| BBH `bpl_double_gaussian` + `madau_dickinson` | 5e-4 | 1.7 / 3.2 ms |
| BNS `uniform` + `madau_dickinson_timedelay` | 1.3e-4 | 2.5 / 6.3 ms |
| NSBH `nsbh_modsmooth` + `madau_dickinson_timedelay` | 2.7e-4 | 3.4 / ~7 ms |

### Spins, precession and tides

```python
# precessing BBH (LVK "Default" spin model), table built once (~4 min on 4 cores), then cached
fo    = FastOmega(freqs, waveform="IMRPhenomXPHM", spins="precessing", n_proc=4)
omega = fo.make("bpl_double_gaussian", "madau_dickinson")
omega({"alpha_m_1": 1.6}, {"R0": 15.9}, {"alpha_chi": 1.6, "beta_chi": 4.1, "zeta": 0.66, "sigma_t": 1.5})

# BNS with EOS tides (deterministic Lambda(m)) and small precessing spins
fo    = FastOmega(freqs, waveform="IMRPhenomXP_NRTidalv2", tides=True, spins="precessing",
                  a_max=0.05, m_tot_range=(1.8, 6.0), q_min=0.3, n_q=41, n_proc=4)
omega = fo.make("uniform", "madau_dickinson_timedelay")
omega({}, {"R0": 23.4}, {}, {"Lambda_14": 400.0, "b1": -6.0, "b2": -3.0})
```

* `spins=None | "aligned" | "precessing"`, `tides=False | True`, independently.
  Spin model `lvk_default` (magnitudes $a/a_{\max}\sim$ Beta, tilts isotropic + aligned
  Gaussian mixture); tide model `eos_lnpoly`
  ($\ln\Lambda=\ln\Lambda_{1.4}+b_1\ln(m/1.4)+b_2\ln^2(m/1.4)$, black hole above
  `m_NS_max`; one deterministic EOS, no per-star scatter; default $b_2=-3$, typical of
  realistic EOSs).  All hyperparameters are
  differentiable.
* How: the table gets columns (q node) x (kappa2T node, tides) x (Sobol spin
  sample, spins).  Each column is averaged **exactly** over the observer
  direction: IMRPhenomXPHM through its inertial-frame modes,
  $\langle I\rangle=\frac{1}{8\pi}\sum_{\ell m}(|h_{\ell m}(f)|^2+|h_{\ell m}(-f)|^2)$, in which
  the precession angles drop out; other models by product quadratures exact for
  their modes (`orientation.py`).  Per call, spins enter as self-normalised
  importance weights and tides as kappa2T interpolation weights computed on the
  source-frame mass grid.
* Spins are defined at a fixed dimensionless reference frequency
  $Mf_{\rm ref}T_\odot=$ `x_ref` (default 0.005; with tides 3e-4), which keeps the
  scale invariance exact (a fixed 20 Hz would break it for precessing spins).
* Validated against `OmegaIntegrator(spins=...)` (independent: true masses,
  sampled orientations) and against direct sampling of masses, redshift and EOS
  Lambdas for tides.

| configuration | table (4 cores, once) | Omega / value+grad |
|---|---|---|
| no spins, XAS | ~7 s | 1.9 / 3.4 ms |
| aligned spins, XAS | ~15 s | ~3 / ~8 ms |
| precessing, XPHM | ~4 min | ~5 / ~15 ms |
| BNS tides (+ precession), XP_NRTidalv2 | ~7 s (~4 min) | ~9 / ~18 ms |

Accuracy with spins: the spin average carries a fixed Monte Carlo error of the
table (`n_spin` samples per q node): ~1e-3 below 200 Hz, ~0.5% at 1 kHz for BBH
with `n_spin=128` (it scales as $n_{\rm spin}^{-1/2}$).  Tides agree with direct
sampling within its errors (1e-5 absolute at 50 Hz, ~1% of a 67% effect at 1 kHz).

**Caveats.**  (0) With `n_proc > 1` in a `.py` script, put the code under
`if __name__ == "__main__":` (macOS/Windows start worker processes by
re-importing the script); notebooks are fine.  NRTidal models stop at the BNS
merger, so above a few hundred Hz a large part of the "tidal effect" is the
missing post-merger emission; `IMRPhenomXPHM_NRTidalv2` has no
frequency-sequence interface in LAL, so tides + precession use
`IMRPhenomXP_NRTidalv2` (no higher modes, negligible for BNS).  (1) The $(M,q)$ grid (`m_tot_range`, `q_min`) must contain the
population support for every $\Lambda$ in the prior; mass outside it is
silently dropped.  A tight range is also faster.  (2) Spin and tide populations are assumed independent of the masses (except
through Lambda(m)).

## Usage

```python
from stochaxtic import OmegaIntegrator

oi  = OmegaIntegrator("bpl_double_gaussian", "madau_dickinson",
                      rate_kwargs={"R0": 15.9}, waveform="IMRPhenomXAS")
res = oi.run(n_rep=8)                    # 8 replicas x 2^12 points
res.freqs, res.omega, res.omega_err      # (996,) each, 5-1000 Hz
res.save("bbh.npz")                      # OmegaResult.load("bbh.npz")
```

Hyperparameters go through `mass_kwargs` / `rate_kwargs`, or pass model
instances directly:

```python
from stochaxtic import get_mass_model, get_rate_model
mass = get_mass_model("bpl_double_gaussian", params={"alpha_m_1": 2.0})
rate = get_rate_model("madau_dickinson_timedelay", R0=23.4, td_min=0.05)
res  = OmegaIntegrator(mass, rate, waveform="IMRPhenomXPHM").run()
```

Command line (one `.npz` per call; replica ranges can be split across jobs
and merged):

```bash
python -m stochaxtic run bns.npz --mass-model uniform --rate-model madau_dickinson_timedelay --R0 23.4
python -m stochaxtic merge all.npz part0.npz part1.npz
```

`examples/condor/` has a submit file running one population per job.

### Models

| mass (`get_mass_model`) | rate (`get_rate_model`) |
|---|---|
| `bpl_double_gaussian` (GWTC-5.0 BBH defaults) | `madau_dickinson` |
| `uniform` | `madau_dickinson_timedelay` |
| `nsbh`, `nsbh_modsmooth` | `power_law`, `constant` |
| `truncated_powerlaw` | |

Waveforms: any LAL approximant (`mode="sequence"` for FD models,
`mode="fd"` also for TD models such as NRSur7dq4).  Non-LAL models: subclass
`WaveformModelBase` and register it in `waveforms._REGISTRY`.
`waveform_kwargs={"isco_cut": 3}` reproduces the old per-frequency cut
(which removes most of the merger-ringdown; off by default).

## Algorithm

1. **Redshift kernel.** The waveform depends on $z$ only through
   $\mathcal M_{\rm det}=\mathcal M(1+z)$ and $1/d_L$, so the $z$ integral
   is done deterministically:
   $K(\mathcal M_{\rm det},q)=\int\mathrm dz\,\frac{\mathrm dN}{\mathrm dz\,\mathrm dt}\,
   p(\mathcal M_{\rm det}/(1+z),q)/[(1+z)d_L^2]$ (trapezoid, 1000 points in $\ln(1+z)$).
2. **Quasi Monte Carlo** over $(\ln\mathcal M_{\rm det},\ln q,\cos\iota,\phi)$
   with scrambled Sobol points and an importance proposal built from $K$;
   one waveform call gives every frequency.
3. **Replicas** (independent scramblings) give the error bar
   `omega_err` = std / $\sqrt R$.  `omega_err_mc` is the plain-MC error
   (conservative for QMC).  `res.ess` is the effective sample size per
   replica and frequency: values $\ll 0.01$ mean the proposal misses where
   the integrand lives and the error bar is not trustworthy.

The result is unbiased for any proposal (a uniform defensive component
covers the whole domain); the proposal settings only change the variance.
Defaults (`eps=0.1, tilts=(-5,-2,1,4), iota_power=1`, no antithetic pairs)
were tuned for $\Omega_{\rm GW}$ over 5-1000 Hz.

Measured with the defaults (8 replicas x $2^{12}$ points, IMRPhenomXAS,
5-1000 Hz, one core of an Apple laptop):

| population | median / max rel. error | time |
|---|---|---|
| BBH: `bpl_double_gaussian` + `madau_dickinson` | 0.04% / 0.08% | 11 s |
| BNS: `uniform` + `madau_dickinson_timedelay` | 0.01% / 0.01% | 7 s |
| NSBH: `nsbh_modsmooth` + `madau_dickinson_timedelay` | 0.05% / 0.08% | 10 s |

## Validation and differences from MOGWAS

With the same seed and proposal, every replica equals the MOGWAS
all-frequency result times a constant, to $2\times10^{-9}$
(`tests/test_stochaxtic.py::test_regression_against_mogwas`).  The
constant comes from deliberate changes:

* speed of light 299792.458 km/s (was 3e5): $\Omega\propto c$, $-0.069\%$;
* Julian year (was 3.156e7 s): $+0.0076\%$;
* mass normalisations are exact.  All models integrate to 1 over
  $m_2\le m_1$.  Before the 2026-09-29 fix of MOGWAS `mass_models.py`,
  `uniform` integrated to 1/2 and `nsbh` to ~7.9.

Also: $d_L$, $\mathrm dV_c/\mathrm dz$ and $t(z)$ come from tables
accurate to $10^{-6}$ against astropy's Planck18 (no per-point `quad`),
and the time-delay rate table extends to $z=0$, so $R(0)=R_0$ holds exactly.
