"""
StochAXtic (import stochaxtic) -- fast Omega_GW(f) of a compact-binary population.

    from stochaxtic import OmegaIntegrator
    oi  = OmegaIntegrator("bpl_double_gaussian", "madau_dickinson",
                          rate_kwargs={"R0": 15.9}, waveform="IMRPhenomXAS")
    res = oi.run(n_rep=8)            # res.freqs, res.omega, res.omega_err
"""

from . import _compat  # noqa: F401  (numpy/scipy shims; keep first)
from .constants import S_REF, H0_SI
from .cosmology import COSMO, Cosmology, RateModelBase, get_rate_model
from .mass_models import MassModelBase, get_mass_model
from .waveforms import WaveformModelBase, get_waveform_model
from .integrator import (OmegaIntegrator, OmegaResult, RedshiftKernel, Proposal,
                         compute_omega)

__version__ = "0.1.0"

__all__ = ["OmegaIntegrator", "OmegaResult", "compute_omega", "RedshiftKernel",
           "Proposal", "get_mass_model", "get_rate_model", "get_waveform_model",
           "MassModelBase", "RateModelBase", "WaveformModelBase", "Cosmology",
           "COSMO", "S_REF", "H0_SI"]
