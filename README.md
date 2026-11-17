<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo-light.svg" alt="StochAXtic" width="520">
  </picture>
</p>

Fast $\Omega_{\rm GW}(f)$ of a compact-binary population, differentiable in the
population hyperparameters $\Lambda$ and ready for MCMC with numpyro.

$$
\Omega_{\rm GW}(f) = \frac{4\pi^2 f^3}{3H_0^2}\int \mathrm{d}z \frac{\mathrm{d}N}{\mathrm{d}z \mathrm{d}t}
\Big\langle |\tilde h_+(f)|^2 + |\tilde h_\times(f)|^2 \Big\rangle_{\rm masses,\ spins,\ tides,\ orientation}
$$

```bash
pip install -e ".[fast,lal]"    # numpy, scipy, astropy, jax, lalsuite
```

## How it works

Two evaluators compute the same quantity in independent ways.

**`FastOmega`** (JAX) evaluates $\Omega_{\rm GW}(f;\Lambda)$ in a few
milliseconds, with gradients.

1. **Scale invariance.** In GR the intensity of a binary of total mass $M$
   obeys $|\tilde h_+|^2+|\tilde h_\times|^2 = M^4 J(Mf;\theta)$, where $\theta$
   collects the dimensionless parameters (mass ratio, spins, tidal
   deformabilities).  One table of $J$, built once per waveform model and
   cached, serves every mass, redshift and frequency.
2. **Exact orientation average.** Each table column is averaged over the
   observer direction exactly, using the waveform modes, so no inclination or
   phase is sampled.
3. **Integrals as convolutions.** On logarithmic grids with a common spacing,
   the redshift integral is a convolution in $\ln M$ and the frequency
   dependence a cross-correlation; both are done by FFT.
4. **Low rank.** $J$ is compressed by a truncated SVD across its columns, so
   each call needs one FFT per retained mode (about ten without spins) instead
   of one per column.
5. **What depends on $\Lambda$.** Only the population weights on the grids:
   the mass density, the merger rate, and the weights of the spin samples and
   tide nodes.  The waveform is never called again.

**`OmegaIntegrator`** (numpy) is the reference: a quasi Monte Carlo integral
that works with any waveform, in about 10 s per population.  The redshift
integral is done deterministically (the waveform depends on $z$ only through
the detector-frame masses and $1/d_L$); the remaining parameters are sampled
with scrambled Sobol points and an importance proposal, one waveform call
giving every frequency; independent replicas give the error bar.

The two agree to better than $10^{-3}$ between 10 Hz and 1 kHz for BBH, BNS
and NSBH populations.

## Code structure

```
stochaxtic/
  fast_omega.py        FastOmega: table of J, SVD, FFT evaluation (JAX)
  jax_populations.py   mass, rate, spin and tide models for FastOmega (JAX)
  orientation.py       exact orientation average of the intensity
  integrator.py        OmegaIntegrator: quasi Monte Carlo reference (numpy)
  mass_models.py       source-frame mass distributions p(m1, m2)
  spin_models.py       spin distributions
  cosmology.py         Planck18 background and merger-rate models
  waveforms.py         intensity of one source; any LAL approximant
  constants.py         physical constants
  __main__.py          command line: python -m stochaxtic run | merge
tutorials/             six notebooks, from the grids and the table to numpyro
Data/                  data used to generate figures in the paper
```

## Citation
