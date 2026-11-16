# Differentiable, jit-able Omega_GW(f; Lambda) for MCMC with numpyro, with optional
# spins (aligned or precessing) and tides.
#
#   fo    = FastOmega(freqs, waveform="IMRPhenomXPHM", spins="precessing")
#   omega = fo.make("bpl_double_gaussian", "madau_dickinson")
#   omega(mass_params, rate_params, spin_params)                  # ~ms, (Nf,)
#
# In GR the intensity of a source with dimensionless parameters theta (q, spins at a
# fixed dimensionless reference frequency, tidal deformabilities) scales as
#     I(f; M, theta) = M^4 J(M f; theta).
# J is tabulated once per waveform model as columns: q nodes x tide nodes (kappa_eff^T,
# and for NRTidalv3 its split between the stars) x spin samples (Sobol, per q node),
# each averaged exactly over the observer direction (orientation.py), and compressed by
# an SVD streamed from disk, J(x; col) ≈ sum_r B_r(x) C_r(col).  Per call only weights
# change:
#   spins: self-normalised importance weights p(s | Lambda_s) / g(s) per q node
#   tides: Lambda(m) of one EOS on the source-frame mass grid, interpolated between
#          the tide nodes
# giving P_r(l) = sum_{q,t} P(l, q) W(l, q, t) Cbar_r(q, t).  The redshift integral is
# an FFT convolution in ln M, the frequency dependence an FFT cross-correlation in ln f.
#
# Requirements: float64 in jax; the (M, q) grid contains the population for every
# Lambda of the prior; spin magnitudes in [0, a_max], kappa_eff^T in [0, kappa_max].

import hashlib
import os
import shutil
import tempfile
import time
import warnings
from multiprocessing import Pool

import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import CubicSpline
from scipy.stats import qmc, beta as _beta_dist, truncnorm as _truncnorm_dist

from .constants import H0_SI, YR_S, Z_MIN_DEFAULT, Z_MAX_DEFAULT
from .cosmology import COSMO
from .orientation import OrientationAverager, TIDAL_MODELS, PRECESSING_MODELS, NSBH_MODELS
from . import jax_populations as jp

T_SUN = 4.925490947641267e-06        # G Msun / c^3 in s

# Preconditioning exponent for the frequency correlation.  With
# ln x = ln T_sun + ln M_det + ln f, a factor x^A can be moved exactly from
# J onto M_det and f.  A = 7/3 flattens the inspiral J ∝ x^(-7/3), so the FFT
# no longer mixes terms spanning ~20 decades.
A_PRE = 7.0 / 3.0

# Omega below this fraction of its peak is set to zero.  There (above the last ringdown: from
# ~3 kHz for the default BBH, ~1 kHz for heavy high-z ones, ~8 kHz for NSBH) the FFTs leave
# only round-off
OMEGA_REL_CUT = 1e-12

# sub-samples per ln M cell: 1 suffices for smooth (tapered) mass models;
# models with hard edges need cell averages
DEFAULT_N_SUB = {"bpl_double_gaussian": 1}

# Proposal densities of the spin magnitudes x = a / a_max on [0, 1] (tilts and azimuths are
# uniform).  The defensive proposals mix a uniform part alpha (importance weights <= 1 / alpha
# for any population) with a shape h close to a measured BBH population:
#   defensive_gwtc3       h = Beta(1.5, 4)                ~ gwtc3_default (Beta(1.64, 4.15))
#   defensive_gwtc5       h = N(0.15, 0.32) on [0, 1]     ~ GWTC-5 magnitudes
#   defensive_gwtc5_wide  h = N(0.15, 0.45) on [0, 1]     wider, for priors around gwtc5
# The shapes are meant for a_max = 1 (BBH, NSBH); for BNS (a_max ~ 0.05) use "uniform".
# Seed scatter of Omega (BBH, XPHM precessing, 128 samples per q node, 100 Hz;
# "auto" (the default) uses defensive_gwtc5_wide for a_max >= 0.5, "uniform" otherwise.
SPIN_PROPOSALS = {
    "uniform": None,
    "defensive_gwtc3": dict(alpha=0.3, beta=(1.5, 4.0)),
    "defensive_gwtc5": dict(alpha=0.3, truncnorm=(0.15, 0.32)),
    "defensive_gwtc5_wide": dict(alpha=0.3, truncnorm=(0.15, 0.45)),
}


def _magnitude_proposal(name):
    """(inverse CDF, density) of x = a / a_max on [0, 1] for a spin proposal."""
    p = SPIN_PROPOSALS[name]
    if p is None:
        return (lambda u: u), (lambda x: np.ones_like(x))
    al = p["alpha"]
    if "beta" in p:
        h = _beta_dist(*p["beta"])
    else:
        mu, sig = p["truncnorm"]
        h = _truncnorm_dist((0.0 - mu) / sig, (1.0 - mu) / sig, loc=mu, scale=sig)
    x = np.linspace(0.0, 1.0, 40001)
    cdf = al * x + (1 - al) * h.cdf(x)
    return (lambda u: np.interp(u, cdf, x)), (lambda y: al + (1 - al) * h.pdf(y))


# How kappa_eff is shared between the stars.  For Omega_GW (|h|^2 only) an NRTidal model depends
# on two tidal numbers: the amplitude correction on kappa_eff = c1 Lambda1 + c2 Lambda2 = 3/16
# Lambda-tilde, and the merger frequency, where the waveform is tapered.  NRTidalv2 takes the
# merger frequency from kappa_eff as well; NRTidalv3 from the Damour-Nagar coupling kappa_2^T =
# 3 (X2 X1^4 Lambda1 + X1 X2^4 Lambda2) (arXiv:2311.07456, Eqs. 29, 40, 43), a different
# combination unless m1 = m2.  Its tables get a second tidal axis, the share w = c1 Lambda1 /
# kappa_eff of the heavier star, on which that coupling is linear.  For one EOS Lambda(m)
# decreases with m, so
#     0 <= w <= w0(q) = c1 / (c1 + c2)
# (w = 0: the heavier star is at or above the maximum NS mass; w = w0: Lambda1 = Lambda2).  The
# nodes are SPLIT_FRACTIONS of w0, interpolated cubically in w, so every EOS is interpolated.
SPLIT_MODELS = {"IMRPhenomXAS_NRTidalv3", "IMRPhenomXP_NRTidalv3", "SEOBNRv5_ROM_NRTidalv3"}
SPLIT_FRACTIONS = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)


def _split_nodes(q, fractions=None):
    """
    w = c1 Lambda1 / kappa_eff of the split nodes, (len(q), n): the fractions of w0 = c1/(c1 + c2)
    (SPLIT_MODELS), or (fractions None) the single split Lambda1 / Lambda2 = q^6.
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))[:, None]
    c1, c2 = jp.kappa_eff_coefficients(q)
    if fractions is None:
        return c1 * q ** 6 / (c1 * q ** 6 + c2)
    return np.asarray(fractions, dtype=float)[None, :] * c1 / (c1 + c2)


def _streaming_svd(table, colw, rank, rank_tol, chunk_bytes=2 ** 28):
    """
    Truncated SVD of A = diag(1/s) J diag(colw), J (n_x, n_col) given as `table` (n_col, n_x)
    (an array or a memory map, read chunk by chunk), s the RMS of J over the columns (rows
    with s = 0, beyond every merger, are zero).  Never holds more than a chunk of columns:
      pass 1  s;
      pass 2  R of the QR factorisation of A^T, by stacked QR of the chunks (TSQR); A = R^T Q^T
              with Q orthonormal, so A and R^T share their singular values and left vectors,
              which the SVD of R^T (n_x x n_x) gives to full precision (a Gram matrix A A^T
              would square the condition number and lose singular values below ~1e-8);
      pass 3  the first `rank` right vectors, V_r^T = Sigma_r^-1 U_r^T A, chunk by chunk.
    Returns s, all singular values, B = U_r Sigma_r (n_x, rank) and V_r^T (rank, n_col).
    rank None: the smallest rank with sigma_r / sigma_1 < rank_tol, at most 96.
    """
    n_col, n_x = table.shape
    step = max(1, chunk_bytes // (8 * n_x))
    chunks = [slice(i, min(i + step, n_col)) for i in range(0, n_col, step)]
    s2 = np.zeros(n_x)
    for c in chunks:
        s2 += np.sum(np.square(np.asarray(table[c])), axis=0)
    sx = np.sqrt(s2 / n_col)
    inv = np.where(sx > 0, 1.0 / np.where(sx > 0, sx, 1.0), 0.0)
    rows = lambda c: np.asarray(table[c]) * inv[None, :] * colw[c, None]     # A^T, chunk c
    R = np.zeros((0, n_x))
    for c in chunks:                                     # TSQR: R of [R; A^T chunk]
        R = np.linalg.qr(np.vstack([R, rows(c)]), mode="r")
    u, sv, _ = np.linalg.svd(R.T, full_matrices=False)   # singular values / left vectors of A
    if rank is None:
        # Omega's truncation error is ~0.03-0.1 x rank_tol (measured)
        rank = int(min(96, max(1, np.sum(sv / sv[0] > rank_tol))))
    rank = int(min(rank, np.sum(sv > 0)))                                    # <= n_columns
    U = u[:, :rank]
    vt = np.empty((rank, n_col))
    for c in chunks:
        vt[:, c] = (rows(c) @ U).T / sv[:rank, None]
    return sx, sv, U * sv[:rank], vt


def _fix_signs(B, vt):
    """
    Sign convention of the singular pairs (each is defined up to a common sign): the largest
    entry of every left vector B[:, r] is positive.  Makes the decomposition independent of the
    algorithm (dense SVD, TSQR) and of the LAPACK build; B V^T, and so Omega, are unchanged.
    """
    s = np.sign(B[np.argmax(np.abs(B), axis=0), np.arange(B.shape[1])])
    s[s == 0] = 1.0
    return B * s[None, :], vt * s[:, None]


def _next_fast_len(n):
    return int(2 ** np.ceil(np.log2(n)))


def _q_nodes(rule, n, a):
    """Nodes (ascending) and weights of the ln q quadrature on [a, 0]."""
    if rule == "simpson":
        n += 1 - n % 2
        x = np.linspace(a, 0.0, n)
        w = np.full(n, 2.0)
        w[1::2] = 4.0
        w[[0, -1]] = 1.0
        return x, w * (x[1] - x[0]) / 3.0
    if rule == "trapezoid":
        x = np.linspace(a, 0.0, n)
        w = np.full(n, x[1] - x[0])
        w[[0, -1]] *= 0.5
        return x, w
    if rule == "gauss":
        t, w = np.polynomial.legendre.leggauss(n)
        return 0.5 * a * (1.0 - t), -0.5 * a * w
    raise ValueError(f"q_rule must be 'simpson', 'gauss' or 'trapezoid', not {rule!r}")


# ----------------------------------------------------------------------
# table columns (module level so that they can run in a process pool)
# ----------------------------------------------------------------------

_AVG = None


def _cache_path(cache_dir, prefix, waveform, key):
    """<cache_dir>/<prefix>_<waveform>_<sha1 of key>.npz, or None without a cache."""
    if not cache_dir:
        return None
    h = hashlib.sha1((repr(sorted(key.items())) + waveform).encode()).hexdigest()[:16]
    return os.path.join(os.path.expanduser(cache_dir), f"{prefix}_{waveform}_{h}.npz")


def _lal_versions():
    """Versions of LAL and LALSimulation, part of the table key: the cached tables are their
    waveforms, so a LALSuite upgrade must not reuse tables built with an older one."""
    try:
        import lal
        import lalsimulation
    except ImportError:
        return "unavailable"
    return f"lal {lal.__version__}, lalsimulation {lalsimulation.__version__}"


def _init_table_worker(waveform, rule):
    global _AVG
    _AVG = OrientationAverager(waveform, rule)


def _table_column(task):
    """task = (f, f_ref, m_ref, q, spin 6-vector, aligned, lambdas or None) -> (n_kappa, n_f)."""
    f, f_ref, m_ref, q, s, aligned, lambdas = task
    a1, a2, c1, c2, p1, p12 = s
    # IMRPhenomXP/XPHM jump at exactly m1 == m2 (+10% at merger, x18 in the ringdown
    # tail; XAS/XHM are continuous): evaluate the q -> 1 limit
    q = min(q, 1.0 - 1e-6)
    m1, m2 = m_ref / (1 + q), m_ref * q / (1 + q)
    # aligned: keep only the projections a cos t on L (in-plane components dropped)
    r1, r2 = ((0.0, 0.0) if aligned else
              (np.sqrt(max(0.0, 1 - c1 * c1)), np.sqrt(max(0.0, 1 - c2 * c2))))
    # components in the frame z = L, x = orbital separation at f_ref: p1 is the azimuth of
    # spin 1's in-plane part relative to the separation, p1 + p12 that of spin 2
    s1 = (a1 * r1 * np.cos(p1), a1 * r1 * np.sin(p1), a1 * c1)
    s2 = (a2 * r2 * np.cos(p1 + p12), a2 * r2 * np.sin(p1 + p12), a2 * c2)
    out = []
    for lam in (lambdas if lambdas is not None else [None]):
        out.append(_AVG(f, m1, m2, s1, s2, f_ref, lam) / m_ref ** 4)
    return np.array(out)


class FastOmega:
    """
    freqs          frequencies of the output (default 5..1000 Hz, 1 Hz)
    waveform       LAL approximant.  Precessing spins need a precessing model
                   (IMRPhenomXPHM, IMRPhenomXP, IMRPhenomXP_NRTidalv2/v3,
                   IMRPhenomPv2/Pv3/Pv3HM; Pv3/Pv3HM columns use their co-precessing modes,
                   no MSA angles, see orientation.py 4)), tides
                   an NRTidal model (IMRPhenomXP_NRTidalv2/v3, IMRPhenomXAS_NRTidalv2/v3,
                   SEOBNRv5_ROM_NRTidalv3; NRTidalv3 tables have a second tidal axis, see
                   SPLIT_MODELS)
                   or IMRPhenomNSBH (aligned spins; Lambda1 ignored;
                   needs q_min >= 0.01, m_ref <= 6 and Lambda_NS <= 5000, see check_tides)
    spins          None, "aligned" or "precessing"
    tides          include tidal deformabilities (EOS model Lambda(m))
    m_tot_range    source-frame total mass range (Msun) of the grid; must
                   contain the support of every population you evaluate
    q_min          smallest mass ratio of the grid (same requirement)
    z_min, z_max   redshift range of the integral
    delta          spacing of the ln M, ln(1+z) and ln f grids
    n_q            points of the ln q grid (Simpson: made odd)
    q_rule         quadrature in ln q on [ln q_min, 0]: "simpson" (default), "gauss"
                   (Gauss-Legendre) or "trapezoid".  For populations smooth in q "gauss"
                   reaches 1e-5 at n_q ~ 31, Simpson at ~121; with hard q edges (uniform,
                   nsbh) every rule is O(1/n_q)
    n_spin         Sobol spin samples per q node (spins != None); narrow spin
                   populations need more (see check_spins)
    a_max          spin magnitudes live in [0, a_max] (use ~0.05 for BNS); a pair (a1_max, a2_max)
                   bounds the heavier and the lighter component separately, e.g. (1, 0.05)
                   for NSBH (black hole, neutron star)
    x_ref          spins are defined at f_ref = x_ref / (M T_sun), i.e. at a
                   fixed M f (default 0.005, i.e. 20 Hz for M ~ 50 Msun; with
                   tides 3e-4, i.e. 20 Hz for M ~ 3 Msun)
    kappa_min, kappa_max, n_kappa   kappa_eff grid (tides): 0 plus n_kappa-1 nodes log-spaced
                   on [kappa_min, kappa_max].  NRTidal tapers the waveform at a kappa-dependent
                   merger frequency, so J is very non-linear in kappa but smooth in ln kappa.
                   kappa_eff = 3/16 Lambda-tilde: ~10-400 for BNS, O(1) for NSBH.  Defaults:
                   [0.1, 3000], 24 nodes (NRTidal); [0.01, 300], 48 nodes (IMRPhenomNSBH, whose
                   disruption makes J non-linear over kappa ~ 0.3-15
    kappa_interp   interpolation between the kappa nodes: "cubic" (default; 4-point Lagrange in
                   ln kappa, linear in kappa below the first log node) or "linear" (in kappa).
                   BNS, 24 nodes: |dOmega/Omega| <= 5e-5 below 1 kHz ("cubic"), 1.4e-2 at 1 kHz.
                   Not part of the table.
    n_split        nodes of the split axis of SPLIT_MODELS (NRTidalv3): the share w of the primary
                   in kappa_eff, uniform on [0, w0(q)], Lagrange interpolation of degree n_split - 1
                   (default 4, SPLIT_FRACTIONS; 7 halves the intervals and keeps the default nodes)
    table_step     the J table is computed on an ln x grid table_step * delta
                   apart and its SVD basis interpolated to the delta grid
    rank           SVD rank of the table; None picks the smallest rank with sigma_r / sigma_1 <
                   rank_tol (at most 96, and at most the number of columns)
    orientation_rule  see orientation.OrientationAverager (None: exact default)
    m_ref          total mass at which the table is generated (default 20,
                   2.8 with tides)
    n_proc         processes for building the table
    cache_dir      the table (a memory-mapped .npy, read with table()) and its truncated SVD
                   are cached here (None: the table goes to a temporary file, deleted after the
                   SVD).  Written column by column and decomposed chunk by chunk, so memory does
                   not grow with the number of columns; disk space does (8 n_columns n_xJ bytes)
    seed           seed of the spin samples
    spin_proposal  distribution of the spin-magnitude samples (part of the table): "uniform" on
                   [0, a_max] or a defensive mixture of SPIN_PROPOSALS.  Default "auto":
                   "defensive_gwtc5_wide" for a_max >= 0.5 (BBH, NSBH; halves the spin Monte
                   Carlo error for gwtc5_default), "uniform" otherwise (BNS).  The default spin
                   population (self.spin_model) follows the same rule: "gwtc5_default" for
                   a_max >= 0.5, "uniform_isotropic" otherwise
    """

    def __init__(self, freqs=None, waveform="IMRPhenomXAS", spins=None, tides=False,
                 waveform_kwargs=None, m_tot_range=(1.5, 700.0), q_min=0.01,
                 z_min=Z_MIN_DEFAULT, z_max=Z_MAX_DEFAULT, delta=0.005, n_q=121,
                 q_rule="simpson", n_spin=128, a_max=1.0, x_ref=None, kappa_min=None, kappa_max=None,
                 n_kappa=None, kappa_interp="cubic", n_split=None, spin_proposal="auto",
                 table_step=2, rank=None, rank_tol=1e-4, orientation_rule=None,
                 m_ref=None, n_proc=1, cache_dir="~/.cache/stochaxtic",
                 seed=0, cosmo=COSMO, verbose=True):
        if jnp.asarray(1.0).dtype != jnp.float64:
            raise RuntimeError("FastOmega needs float64: call "
                               "jax.config.update('jax_enable_x64', True) "
                               "(or numpyro.enable_x64()) first")
        if waveform_kwargs:
            raise ValueError("waveform_kwargs are not supported by FastOmega")
        if spins not in (None, "aligned", "precessing"):
            raise ValueError("spins must be None, 'aligned' or 'precessing'")
        if spins == "precessing" and waveform not in PRECESSING_MODELS:
            raise ValueError(f"spins='precessing' needs a precessing waveform, not {waveform}")
        if tides and waveform not in TIDAL_MODELS:
            raise ValueError(f"tides=True needs one of {sorted(TIDAL_MODELS)}, not {waveform}")

        t0 = time.time()
        self.freqs = (np.arange(5.0, 1001.0, 1.0) if freqs is None
                      else np.asarray(freqs, dtype=float))
        if kappa_interp not in ("cubic", "linear"):
            raise ValueError("kappa_interp must be 'cubic' or 'linear'")
        if np.ndim(a_max) == 0:
            self.a_max = float(a_max)
            self._amax = (self.a_max, self.a_max)                    # (heavier, lighter) component
        else:
            self._amax = tuple(float(a) for a in a_max)
            if len(self._amax) != 2:
                raise ValueError("a_max must be a number or a pair (a1_max, a2_max)")
            self.a_max = self._amax
        self.spins, self.tides = spins, bool(tides)
        self.kappa_interp = kappa_interp
        if spin_proposal == "auto":           # magnitudes shaped like GWTC-5 BBHs; uniform for small a_max
            spin_proposal = "defensive_gwtc5_wide" if self._amax[0] >= 0.5 else "uniform"
        if spin_proposal not in SPIN_PROPOSALS:
            raise ValueError(f"spin_proposal must be 'auto' or one of {list(SPIN_PROPOSALS)}")
        self.spin_proposal = spin_proposal
        # default spin_model of omega, make, check_spins and table_intensity, same rule
        self.spin_model = "gwtc5_default" if self._amax[0] >= 0.5 else "uniform_isotropic"
        x_ref  = x_ref if x_ref is not None else (3e-4 if tides else 0.005)
        m_ref  = m_ref if m_ref is not None else (2.8 if tides else 20.0)
        if waveform in NSBH_MODELS:                        # the table must stay in the model's domain
            lim = NSBH_MODELS[waveform]
            if q_min < lim["q_min"] or m_ref / 2 > lim["m2_max"]:
                raise ValueError(f"{waveform} needs q_min >= {lim['q_min']} and m_ref <= "
                                 f"{2 * lim['m2_max']:g} (m2 <= {lim['m2_max']:g} Msun at q = 1)")
        d = delta

        # ---- grids (all logarithmic grids share the spacing d) ----
        self.l_src = np.log(m_tot_range[0]) + d * np.arange(
            int(np.ceil(np.log(m_tot_range[1] / m_tot_range[0]) / d)) + 1)
        xi0, xi1 = np.log1p(z_min), np.log1p(z_max)                     # xi = ln(1+z)
        n_z = int(np.floor((xi1 - xi0) / d)) + 1
        self.xi = xi0 + d * np.arange(n_z)
        wxi = np.full(n_z, d)
        wxi[0] *= 0.5
        wxi[-1] = 0.5 * d + (xi1 - self.xi[-1])       # close the last partial interval
        self.z = np.expm1(self.xi)

        self._lq_min = np.log(q_min)
        self.lq, self.wq = _q_nodes(q_rule, n_q, self._lq_min)
        n_q = len(self.lq)

        n_l, n_d = len(self.l_src), len(self.l_src) + n_z - 1
        self.l_det = self.l_src[0] + self.xi[0] + d * np.arange(n_d)
        lnf0 = np.log(self.freqs.min())
        n_f  = int(np.ceil(np.log(self.freqs.max() / self.freqs.min()) / d)) + 1
        self.lnf = lnf0 + d * np.arange(n_f)
        n_x  = n_d + n_f - 1
        self.ln_x = np.log(T_SUN) + self.l_det[0] + lnf0 + d * np.arange(n_x)

        # ---- the intensity table: columns (q, tide node, spin sample) ----
        # tide nodes: kappa_eff nodes x split nodes (kappa major); a single node without tides
        n_s = n_spin if spins else 1
        kd = (0.01, 300.0, 48) if waveform in NSBH_MODELS else (0.1, 3000.0, 24)
        kappa_min, kappa_max, n_kappa = (v if v is not None else dv
                                         for v, dv in zip((kappa_min, kappa_max, n_kappa), kd))
        n_k = n_kappa if tides else 1
        self.kappa = (np.concatenate([[0.0], np.geomspace(kappa_min, kappa_max, n_k - 1)])
                      if tides else np.zeros(1))
        nrtidal = self.tides and waveform not in NSBH_MODELS
        # split nodes: fractions of w0 (SPLIT_MODELS), None = the single split q^6, () = no split axis
        if n_split not in (None, len(SPLIT_FRACTIONS)) and n_split < 2:
            raise ValueError("n_split must be at least 2")
        fractions = (SPLIT_FRACTIONS if n_split in (None, len(SPLIT_FRACTIONS))
                     else tuple(np.linspace(0.0, 1.0, int(n_split))))
        self.split_fractions = ((fractions if waveform in SPLIT_MODELS else None) if nrtidal else ())
        self._wnodes = _split_nodes(np.exp(self.lq), self.split_fractions) if nrtidal else None
        n_w = len(self.split_fractions) if self.split_fractions else 1
        n_t = n_k * n_w
        self._n_k, self._n_w = n_k, n_w
        self._spin_samples = self._draw_spins(n_q, n_s, seed)              # (n_q, n_s, 6)
        dJ = table_step * d
        lx_hi = min(self.ln_x[-1], 0.0)                                    # every model ends by M f = 1
        self.ln_xJ = np.arange(self.ln_x[0] - 2 * dJ, lx_hi + 2 * dJ, dJ)
        # cache key of the table.  table_version: bump when a code change alters the columns
        # for the same key; lal_version: a LALSuite upgrade can change the waveforms
        tkey = dict(table_version=2, lal_version=_lal_versions(),
                    spins=spins, tides=self.tides, n_s=n_s, seed=seed, a_max=self.a_max,
                    kappa=self.kappa.tolist(), x_ref=x_ref, m_ref=m_ref, lq=self.lq.tolist(),
                    lxJ=[self.ln_xJ[0], dJ, len(self.ln_xJ)], rule=str(orientation_rule),
                    # phi1, lambda_split: keep tables of older layouts from being reused
                    **({"phi1": "uniform"} if spins == "precessing" else {}),
                    **({"lambda_split": "q6" if self.split_fractions is None
                        else {"w0_fractions": list(self.split_fractions)}} if nrtidal else {}),
                    **({"spin_proposal": spin_proposal} if spins and spin_proposal != "uniform" else {}))
        colw = np.sqrt(np.repeat(self.wq, n_t * n_s) / n_s)
        self._cache = (cache_dir, waveform, tkey)

        # ---- truncated SVD of the table (cached) ----
        # The SVD cache holds the row scale, the singular values and the first `rank` singular
        # vectors (~0.1 GB for the largest tables); its key adds the column weights and the
        # truncation settings to the table key.  The decomposition streams the table from
        # disk: ~1 GB of memory for any table.
        svd_file = _cache_path(cache_dir, "svd", waveform,
                               dict(tkey, svd_version=1, wq=self.wq.tolist(), rank=rank, rank_tol=rank_tol))
        if svd_file and os.path.exists(svd_file):
            c = np.load(svd_file)
            sx, sv, B, vt_r = c["sx"], c["sv"], c["B"], c["vt"]
            rank, t_table = B.shape[1], 0.0
            cache_file = self._table_file()
            if verbose:
                print(f"FastOmega: SVD loaded from {svd_file}", flush=True)
        else:
            table, t_table, cache_file, tmp_dir = self._build_table(
                waveform, orientation_rule, m_ref, x_ref, n_proc, cache_dir, tkey, verbose)  # (n_col, n_xJ)
            try:
                t1 = time.time()
                sx, sv, B, vt_r = _streaming_svd(table, colw, rank, rank_tol)
                if verbose:
                    print(f"FastOmega: SVD of {table.shape[0]} columns in {time.time() - t1:.0f} s", flush=True)
            finally:
                del table
                if tmp_dir:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            rank = B.shape[1]
            if svd_file:
                os.makedirs(os.path.dirname(svd_file), exist_ok=True)
                np.savez(svd_file, sx=sx, sv=sv, B=B, vt=vt_r)
        B, vt_r = _fix_signs(B, vt_r)                                       # loaded caches too
        self._sv = sv / sv[0]
        vt_r /= colw[None, :]                                              # C (rank, n_col), in place
        # q-major (n_q, rank * n_t, n_s): the spin average is a batched mat-vec that XLA
        # computes without transposing C (~30% less memory when a jit captures it)
        coef = np.ascontiguousarray(vt_r.reshape(rank, n_q, n_t, n_s).transpose(1, 0, 2, 3)
                                    ).reshape(n_q, rank * n_t, n_s)
        del vt_r
        live = sx > 0
        xe   = self.ln_xJ[live][-1]                                        # last live table node
        spl  = CubicSpline(self.ln_xJ, B, axis=0)
        # the row scale spans tens of decades: cubic spline of ln s on the live nodes
        # (linear interpolation biases Omega by ~-4e-4 at 1 kHz: ln s is curved near merger)
        spl_s = CubicSpline(self.ln_xJ[live], np.log(sx[live]))
        ok   = self.ln_x <= xe
        s_f  = np.zeros(len(self.ln_x))
        s_f[ok] = np.exp(spl_s(self.ln_x[ok]))
        B_f  = spl(self.ln_x) * s_f[:, None]                               # (n_x, rank)

        # ---- constants of the Lambda-dependent part ----
        self._mass_grids = {}
        self._cell = jnp.asarray(d * self.wq)[None, :]                     # dlnM dlnq weights
        dL = cosmo.luminosity_distance(self.z)
        # w(xi) dxi = (1+z) dN/dz / dL^2 dxi and dN/dz = R/(1+z) dVc/dz, R in Gpc^-3/yr
        self._zw  = jnp.asarray(wxi * cosmo.dVc_dz(self.z) / dL ** 2 / (1e9 * YR_S))
        self._e4l = jnp.asarray(np.exp((4 - A_PRE) * self.l_det))[:, None]   # M_det^(4-A)
        self._L1  = _next_fast_len(n_l + n_z - 1)                          # K convolution
        self._L2  = _next_fast_len(n_d + n_x - 1)                          # f correlation
        xa = np.exp(A_PRE * self.ln_x)[:, None]                            # x^A
        self._FJ  = jnp.fft.rfft(jnp.asarray(B_f * xa), n=self._L2, axis=0)
        self._Cq  = jnp.asarray(coef)
        self._rank, self._n_t = rank, n_t
        del coef
        self._B   = B_f                                                    # (n_x, rank), for table_intensity
        self._n_l, self._n_z, self._n_d, self._n_f = n_l, n_z, n_d, n_f
        self._lnf_out = jnp.asarray(np.log(self.freqs))
        self._td_cache = {}
        self.cosmo = cosmo

        if spins:
            s = self._spin_samples
            self._spin_arrays = tuple(jnp.asarray(s[..., i]) for i in range(4))   # a1 a2 c1 c2
            _, dens = _magnitude_proposal(self.spin_proposal)
            # proposal density of (a1, a2, c1, c2); phi1 and phi12 are uniform and cancel
            am1, am2 = self._amax
            self._g_spin = jnp.asarray(dens(s[..., 0] / am1) * dens(s[..., 1] / am2)
                                       / (4.0 * am1 * am2))                       # (n_q, n_s)
        if self.tides:
            M = np.exp(self.l_src)[:, None]
            q = np.exp(self.lq)[None, :]
            self._m1c, self._m2c = jnp.asarray(M / (1 + q)), jnp.asarray(M * q / (1 + q))
            c1, c2 = jp.kappa_eff_coefficients(np.exp(self.lq))
            self._kc1, self._kc2 = jnp.asarray(c1)[None, :], jnp.asarray(c2)[None, :]
            self._lk0, self._dlk = np.log(self.kappa[1]), np.log(self.kappa[2] / self.kappa[1])
            if n_w > 1:
                self._wn = jnp.asarray(self._wnodes)                            # (n_q, n_w)

        self.info = dict(n_lnM=n_l, n_z=n_z, n_q=n_q, n_lnf=n_f, n_x=n_x, n_xJ=len(self.ln_xJ),
                         n_spin=n_s, n_kappa=n_k, n_split=n_w, split_fractions=self.split_fractions,
                         n_columns=n_q * n_t * n_s,
                         rank=rank, delta=d, q_rule=q_rule, waveform=waveform, spins=spins, tides=self.tides,
                         x_ref=x_ref, m_ref=m_ref, table_seconds=t_table, cache_file=cache_file, svd_file=svd_file,
                         setup_seconds=time.time() - t0)
        if verbose:
            print(f"FastOmega: {waveform}, spins={spins}, tides={self.tides}: "
                  f"{self.info['n_columns']} table columns, rank {rank}; "
                  f"grids lnM {n_l}, z {n_z}, lnq {n_q}, lnf {n_f}; "
                  f"setup {self.info['setup_seconds']:.1f} s", flush=True)

    # ------------------------------------------------------------------

    def _draw_spins(self, n_q, n_s, seed):
        """
        (n_q, n_s, 6) = a1, a2, cos t1, cos t2, phi1, phi12; one Sobol scrambling per q node.
        phi1 (azimuth of spin 1's in-plane part relative to the orbital separation at f_ref)
        and phi12 are uniform and cancel against the uniform azimuths of the spin models;
        orbit-averaged precession (IMRPhenomX(P)HM, XP_NRTidalv2) does not depend on phi1,
        other models may.  Magnitudes follow the spin_proposal (inverse CDF of the Sobol
        coordinate); tilts are uniform.
        """
        out = np.zeros((n_q, n_s, 6))
        if not self.spins:
            out[..., 2:4] = 1.0
            return out
        dim = 6 if self.spins == "precessing" else 4
        icdf, _ = _magnitude_proposal(self.spin_proposal)              # a / a_max = icdf(u)
        for k in range(n_q):
            u = qmc.Sobol(dim, scramble=True, seed=np.random.default_rng([seed, k])).random(n_s)
            u = np.clip(u, 1e-9, 1 - 1e-9)
            out[k, :, 0] = self._amax[0] * icdf(u[:, 0])
            out[k, :, 1] = self._amax[1] * icdf(u[:, 1])
            out[k, :, 2] = 2 * u[:, 2] - 1
            out[k, :, 3] = 2 * u[:, 3] - 1
            if dim == 6:
                out[k, :, 4] = 2 * np.pi * u[:, 4]
                out[k, :, 5] = 2 * np.pi * u[:, 5]
        return out

    def _table_file(self):
        """Path of the cached table: the .npy memory map, or a legacy .npz table (n_q, n_tide,
        n_spin, n_xJ), loaded whole, if only that exists; None without a cache."""
        cache_dir, waveform, key = self._cache
        npz = _cache_path(cache_dir, "jtable", waveform, key)
        if npz is None:
            return None
        npy = npz[:-4] + ".npy"
        return npz if os.path.exists(npz) and not os.path.exists(npy) else npy

    def table(self):
        """
        The J table as a read-only array (n_q, n_kappa, n_split, n_spin, n_xJ) on the grid
        self.ln_xJ (n_split = 1 except for SPLIT_MODELS): a memory map, so it costs no memory
        until read.  Needs the cache (cache_dir); for diagnostics and the tutorials.
        """
        path = self.info["cache_file"]
        if not path or not os.path.exists(path):
            raise FileNotFoundError("the table is not cached (cache_dir=None, or the SVD was loaded "
                                    "from a cache whose table was removed)")
        J = np.load(path)["J"] if path.endswith(".npz") else np.load(path, mmap_mode="r")
        n_q, n_s = self._spin_samples.shape[:2]
        return J.reshape(n_q, self._n_k, self._n_w, n_s, len(self.ln_xJ))

    def _build_table(self, waveform, rule, m_ref, x_ref, n_proc, cache_dir, key, verbose):
        """
        (table, seconds, cache_file, tmp_dir): the J table as an (n_columns, n_xJ) array, columns
        ordered (q, tide node, spin sample), tide nodes ordered (kappa, split).  It is written to
        a memory-mapped .npy column by column as the workers return them, so it is never held in
        memory whole; without a cache the file lives in tmp_dir, which the caller removes.
        """
        n_q, n_s = self._spin_samples.shape[:2]
        n_k, n_w = self._n_k, self._n_w
        n_t, n_x = n_k * n_w, len(self.ln_xJ)
        n_col = n_q * n_t * n_s
        path = self._table_file()
        if path and os.path.exists(path):
            if verbose:
                print(f"FastOmega: table loaded from {path}", flush=True)
            J = np.load(path)["J"] if path.endswith(".npz") else np.load(path, mmap_mode="r")
            return J.reshape(n_col, n_x), 0.0, path, None

        t0 = time.time()
        xJ   = np.exp(self.ln_xJ)
        n_live = int(np.sum(xJ < 1.0))                                     # x ascending: a prefix
        f    = xJ[:n_live] / (m_ref * T_SUN)
        fref = x_ref / (m_ref * T_SUN)
        tasks = []
        for k, q in enumerate(np.exp(self.lq)):
            lambdas = None
            if self.tides:
                c1, c2 = jp.kappa_eff_coefficients(q)
                if waveform in NSBH_MODELS:        # kappa_eff = c2 Lambda_NS; Lambda_NS clipped to the domain
                    lmax = NSBH_MODELS[waveform]["lambda_max"]
                    lambdas = [(0.0, min(kap / c2, lmax)) for kap in self.kappa]
                else:                              # NRTidal: share w of the heavier star in kappa_eff
                    lambdas = [(w * kap / c1, (1.0 - w) * kap / c2)
                               for kap in self.kappa for w in self._wnodes[k]]
            for i in range(n_s):
                tasks.append((f, fref, m_ref, q, tuple(self._spin_samples[k, i]),
                              self.spins != "precessing", lambdas))
        if verbose:
            print(f"FastOmega: building the {waveform} table: {len(tasks)} spin/q columns "
                  f"x {n_t} tide nodes ({n_k} kappa x {n_w} split), {len(f)} frequencies each, "
                  f"{8 * n_col * n_x / 1e9:.2f} GB on disk ...", flush=True)
        tmp_dir = None if path else tempfile.mkdtemp(prefix="stochaxtic_")
        target = path or os.path.join(tmp_dir, "jtable.npy")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        part = target[:-4] + ".part.npy"                  # renamed when complete: no torn cache
        mm = np.lib.format.open_memmap(part, mode="w+", dtype=np.float64, shape=(n_col, n_x))
        tide = np.arange(n_t)
        # written pages stay in memory until flushed: flush every ~256 MB, or the whole
        # table would be resident before the end
        per_flush = max(1, 2 ** 28 // (8 * n_t * n_x))

        def store(i, col):                                # task i = (q node, spin sample)
            iq, isp = divmod(i, n_s)
            mm[(iq * n_t + tide) * n_s + isp, :n_live] = col
            if (i + 1) % per_flush == 0:
                mm.flush()

        if n_proc > 1:
            with Pool(n_proc, initializer=_init_table_worker, initargs=(waveform, rule)) as pool:
                for i, col in enumerate(pool.imap(_table_column, tasks,
                                                  chunksize=max(1, len(tasks) // (8 * n_proc)))):
                    store(i, col)
        else:
            _init_table_worker(waveform, rule)
            for i, t in enumerate(tasks):
                store(i, _table_column(t))
        mm.flush()
        del mm
        os.replace(part, target)
        dt = time.time() - t0
        if verbose:
            print(f"FastOmega: table built in {dt:.0f} s" +
                  (f", cached at {path}" if path else ""), flush=True)
        return np.load(target, mmap_mode="r"), dt, path, tmp_dir

    def _mass_grid(self, n_sub):
        """(m1, m2, jacobian) at n_sub points inside each ln M cell [l-d/2, l+d/2]."""
        if n_sub not in self._mass_grids:
            d   = self.info["delta"]
            sub = (np.arange(n_sub) + 0.5) / n_sub - 0.5
            M   = np.exp(self.l_src[:, None] + d * sub[None, :]).reshape(-1)[:, None]
            q   = np.exp(self.lq)[None, :]
            # d(m1, m2) = M/(1+q)^2 dM dq = M^2 q/(1+q)^2 dlnM dlnq
            self._mass_grids[n_sub] = (M / (1 + q), M * q / (1 + q),
                                       M ** 2 * q / (1 + q) ** 2)
        return self._mass_grids[n_sub]

    def _rate(self, rate, params):
        if rate == "madau_dickinson_timedelay":
            if rate not in self._td_cache:          # built eagerly by make(); this FastOmega's cosmology
                self._td_cache[rate] = jp.TimeDelayedMadauDickinson(self.cosmo)
            return self._td_cache[rate](self.z, params)
        return jp.RATE_MODELS[rate](jnp.asarray(self.z), params)

    @property
    def _C(self):
        """The coefficients C_r(q, tide node, spin sample), (rank, n_q, n_t, n_s): a copy, for
        diagnostics and the tutorials (the computation uses the q-major self._Cq)."""
        n_q = self._Cq.shape[0]
        return np.asarray(self._Cq).reshape(n_q, self._rank, self._n_t, -1).transpose(1, 0, 2, 3)

    def _spin_coefficients(self, spin_model, sp):
        """Cbar_r(q, tide node): spin-averaged table coefficients C, self-normalised per q node, (rank, n_q, n_t)."""
        n_q = self._Cq.shape[0]
        if not self.spins:
            Cbar = self._Cq[..., 0]                                            # (n_q, rank * n_t)
        else:
            fn, _ = jp.SPIN_MODELS[spin_model]
            a1, a2, c1, c2 = self._spin_arrays
            w = fn(a1, a2, c1, c2, sp, self.a_max) / self._g_spin             # (n_q, n_s)
            Cbar = jnp.einsum("kxs,ks->kx", self._Cq, w) / jnp.sum(w, axis=1)[:, None]
        return Cbar.reshape(n_q, self._rank, self._n_t).transpose(1, 0, 2)

    def _tide_weights(self, tide_model, tp):
        """
        W(l, q, tide node): interpolation weights of the binary's tidal numbers on the tide
        nodes, (kappa, split) flattened kappa-major: those of kappa_eff(m1, m2) on the kappa
        nodes, times (SPLIT_MODELS) those of the share w = c1 Lambda(m1) / kappa_eff on the
        split nodes.
        """
        fn, _ = jp.TIDE_MODELS[tide_model]
        w_edge = self.info["delta"]
        k1   = self._kc1 * fn(self._m1c, tp, w_edge)
        kap  = k1 + self._kc2 * fn(self._m2c, tp, w_edge)                      # (l, q)
        Wk   = self._kappa_weights(kap)                                        # (l, q, n_k)
        if self._n_w == 1:
            return Wk
        Ww   = self._split_weights(k1 / jnp.where(kap > 0, kap, 1.0))           # (l, q, n_w)
        return (Wk[..., :, None] * Ww[..., None, :]).reshape(kap.shape + (-1,))

    def _split_weights(self, w):
        """
        Lagrange weights in w (l, q) on the split nodes of each q (cubic for the 4 nodes of
        SPLIT_FRACTIONS).  The nodes span the physical range [0, w0] (Lambda decreasing with
        mass), so every EOS is interpolated; w is clipped to [0, 1] only as a guard (w > w0
        would need a heavier star with a larger Lambda, and is then extrapolated).
        """
        wn  = self._wn                                                         # (n_q, n_w)
        n_w = wn.shape[1]
        x = jnp.clip(w, 0.0, 1.0)
        out = []
        for i in range(n_w):
            wi = jnp.ones_like(x)
            for j in range(n_w):
                if j != i:
                    wi = wi * (x - wn[None, :, j]) / (wn[:, i] - wn[:, j])[None, :]
            out.append(wi)
        return jnp.stack(out, axis=-1)

    def _kappa_weights(self, kap):
        """W(l, q, kappa): interpolation weights of kappa_eff (l, q) on the kappa nodes."""
        n_k  = len(self.kappa)
        # nodes: 0, then log-spaced from kappa[1]; the interval is found in ln kappa
        kn   = jnp.asarray(self.kappa)
        kc   = jnp.clip(kap, 0.0, self.kappa[-1])
        lk   = jnp.log(jnp.maximum(kc, self.kappa[1]))
        i0   = jnp.where(kc < self.kappa[1], 0,
                         1 + jnp.floor((lk - self._lk0) / self._dlk).astype(jnp.int32))
        i0   = jnp.clip(i0, 0, n_k - 2)
        # linear in kappa between the two neighbouring nodes (exact for the linear
        # small-kappa response, so it is kept between kappa = 0 and kappa[1])
        fr   = jnp.clip((kc - kn[i0]) / (kn[i0 + 1] - kn[i0]), 0.0, 1.0)
        lin  = ((1.0 - fr)[..., None] * jax.nn.one_hot(i0, n_k)
                + fr[..., None] * jax.nn.one_hot(i0 + 1, n_k))                  # (l, q, kappa)
        if self.kappa_interp == "linear" or n_k < 5:
            return lin
        # cubic: 4-point Lagrange in ln kappa on the log nodes s .. s+3 (one-sided at
        # the ends).  J depends on kappa mainly through the NRTidal taper frequency,
        # a power of kappa, so it is smooth in ln kappa; the error is O(h^4) instead
        # of O(h^2), with the same table.
        s    = jnp.clip(i0 - 1, 1, n_k - 4)
        t    = (lk - self._lk0) / self._dlk - (s - 1)                        # node s at t = 0
        w    = jnp.stack([-(t - 1) * (t - 2) * (t - 3) / 6, t * (t - 2) * (t - 3) / 2,
                          -t * (t - 1) * (t - 3) / 2, t * (t - 1) * (t - 2) / 6], axis=-1)
        cub  = jnp.einsum("lqj,lqjk->lqk", w, jax.nn.one_hot(s[..., None] + jnp.arange(4), n_k))
        return jnp.where((kc < self.kappa[1])[..., None], lin, cub)

    def check_support(self, mass, mass_params=None, n_sub=None, tol=1e-3, warn=True):
        """
        Fraction of the mass distribution (at mass_params, defaults otherwise)
        that lies outside the (M, q) grid, which FastOmega silently drops.
        The density is evaluated on a grid 4x wider in M on both sides and
        reaching q_min/10.  Returns dict(total, below_M, above_M, below_q);
        warns if total > tol.  Call it with the extreme values of your prior
        (e.g. the largest m_max, the smallest m_2_min).
        """
        mfun, mdef = jp.MASS_MODELS[mass]
        n_sub = n_sub or DEFAULT_N_SUB.get(mass, 4)
        d  = self.info["delta"]
        mp = {**mdef, **(mass_params or {}), "_w": d / n_sub}
        lo, hi = self.l_src[0] - d / 2, self.l_src[-1] + d / 2
        l  = np.arange(lo - np.log(4.0), hi + np.log(4.0), d)[:, None]
        lq = np.linspace(self._lq_min - np.log(10.0), 0.0, 800)[None, :]
        M, q = np.exp(l), np.exp(lq)
        P = np.asarray(mfun(jnp.asarray(M / (1 + q)), jnp.asarray(M * q / (1 + q)), mp))
        P = np.nan_to_num(P) * M ** 2 * q / (1 + q) ** 2
        tot = P.sum()
        if not tot > 0:
            raise ValueError(f"the {mass} density vanishes on the extended grid")
        out = dict(below_M=float(P[l[:, 0] < lo].sum() / tot),
                   above_M=float(P[l[:, 0] > hi].sum() / tot),
                   below_q=float(P[:, lq[0] < self._lq_min].sum() / tot))
        out["total"] = float(1.0 - P[(l[:, 0] >= lo) & (l[:, 0] <= hi)][:, lq[0] >= self._lq_min].sum() / tot)
        if warn and out["total"] > tol:
            warnings.warn(
                f"FastOmega: {out['total']:.2%} of the {mass} population lies outside the grid "
                f"(M < {np.exp(lo):.3g}: {out['below_M']:.2%}, M > {np.exp(hi):.3g}: "
                f"{out['above_M']:.2%}, q < {np.exp(self._lq_min):.3g}: {out['below_q']:.2%}) "
                f"and is dropped; widen m_tot_range / q_min", stacklevel=2)
        return out

    def check_spins(self, spin_model=None, spin_params=None, min_ess=10.0, warn=True):
        """
        Effective sample size of the spin importance weights w = p(s | Lambda_s) / g per q
        node, ESS = (sum w)^2 / sum w^2 out of the n_spin samples drawn from the spin
        proposal g (self.spin_proposal).  The samples are fixed, so Omega stays a smooth
        function of Lambda_s, but its error grows as the ESS drops.  Median ESS, aligned
        XAS BBH, 128 samples, defensive_gwtc5_wide: gwtc5_default 91, gwtc3_default 67;
        gwtc5_default with sigma_chi = 0.1: 12, 0.05: 5; zeta = mu_t = 1, sigma_t = 0.1: 1.4.
        The ESS is only a rough guide to the error: with the uniform proposal (spread over
        spin seeds at 1 kHz, 10x smaller at 100 Hz) ESS 41 gave 4e-3 and ESS 3 gave 1e-2.
        Populations much narrower than the proposal have a small ESS: raise n_spin (cost
        and memory grow linearly) or lower a_max to the population's support.  Returns
        dict(ess_median, ess_min, ess (n_q,), n_spin); warns if ess_median < min_ess.  Call
        it with the extreme values of your prior.  spin_model None: self.spin_model.
        """
        if not self.spins:
            raise ValueError("this FastOmega was built without spins")
        spin_model = spin_model or self.spin_model
        fn, dflt = jp.SPIN_MODELS[spin_model]
        a1, a2, c1, c2 = self._spin_arrays
        w = np.asarray(fn(a1, a2, c1, c2, {**dflt, **(spin_params or {})}, self.a_max)
                       / self._g_spin)                                       # (n_q, n_s)
        w = np.nan_to_num(w)
        s1, s2 = w.sum(axis=1), (w ** 2).sum(axis=1)
        ess = np.where(s2 > 0, s1 ** 2 / np.where(s2 > 0, s2, 1.0), 0.0)
        out = dict(ess_median=float(np.median(ess)), ess_min=float(ess.min()), ess=ess,
                   n_spin=w.shape[1])
        if warn and out["ess_median"] < min_ess:
            warnings.warn(
                f"FastOmega: the {spin_model} spin population has a median effective sample "
                f"size of {out['ess_median']:.1f} (min {out['ess_min']:.1f}) out of "
                f"{out['n_spin']} spin samples per q node; Omega may be off by percents at high "
                f"frequency.  Raise n_spin or lower a_max (now {self.a_max})", stacklevel=2)
        return out

    def check_tides(self, mass, mass_params=None, tide_model="eos_lnpoly", tide_params=None,
                    n_sub=None, tol=1e-3, warn=True):
        """
        NSBH waveforms only: fraction of the mass distribution whose neutron star
        has Lambda(m2) above the model's limit (IMRPhenomNSBH: 5000).  The table
        columns are clipped there, so these binaries are evaluated as if Lambda
        were 5000.  Returns dict(frac_above, lambda_max); warns if frac_above > tol.
        Call it with the extreme values of your prior (large Lambda_14, light NSs).
        """
        lim = NSBH_MODELS.get(self.info["waveform"])
        if not (self.tides and lim):
            raise ValueError("check_tides needs tides=True and an NSBH waveform")
        n_sub = n_sub or DEFAULT_N_SUB.get(mass, 4)
        mfun, mdef = jp.MASS_MODELS[mass]
        fn, tdef = jp.TIDE_MODELS[tide_model]
        m1, m2, jac = self._mass_grid(n_sub)
        P = np.nan_to_num(np.asarray(mfun(m1, m2, {**mdef, **(mass_params or {}),
                                                    "_w": self.info["delta"] / n_sub}) * jac))
        lam = np.asarray(fn(m2, {**tdef, **(tide_params or {})}, self.info["delta"]))
        out = dict(frac_above=float(P[lam > lim["lambda_max"]].sum() / P.sum()),
                   lambda_max=lim["lambda_max"])
        if warn and out["frac_above"] > tol:
            warnings.warn(
                f"FastOmega: {out['frac_above']:.2%} of the {mass} population has Lambda_NS > "
                f"{lim['lambda_max']:g}, outside {self.info['waveform']}; it is evaluated at "
                f"Lambda_NS = {lim['lambda_max']:g}", stacklevel=2)
        return out

    def table_intensity(self, spin_params=None, spin_model=None, kappa_index=0,
                        split_index=None):
        """
        Orientation- and spin-averaged J(x; q) reconstructed from the table,
        shape (len(self.ln_x), n_q), at the kappa node `kappa_index` and the split
        node `split_index` (SPLIT_MODELS; default: interpolated to the split
        Lambda1 / Lambda2 = q^6).  For diagnostics: compare with a direct average of
        OrientationAverager.  spin_model None: self.spin_model.
        """
        spin_model = spin_model or self.spin_model
        Cbar = np.asarray(self._spin_coefficients(
            spin_model, {**jp.SPIN_MODELS[spin_model][1], **(spin_params or {})}))
        Cbar = Cbar[:, :, kappa_index * self._n_w:(kappa_index + 1) * self._n_w]   # (rank, n_q, n_w)
        if self._n_w == 1:
            return self._B @ Cbar[:, :, 0]
        if split_index is not None:
            return self._B @ Cbar[:, :, split_index]
        Ww = np.asarray(self._split_weights(jnp.asarray(_split_nodes(np.exp(self.lq)).T)))[0]
        return self._B @ np.einsum("rqj,qj->rq", Cbar, Ww)

    # ------------------------------------------------------------------

    def omega(self, mass, rate, mass_params=None, rate_params=None, spin_params=None,
              tide_params=None, n_sub=None, spin_model=None, tide_model="eos_lnpoly"):
        """Omega_GW at self.freqs (not jitted; see make()); >= 0, and 0 below OMEGA_REL_CUT of the peak.
        spin_model None: self.spin_model."""
        spin_model = spin_model or self.spin_model
        n_sub = n_sub or DEFAULT_N_SUB.get(mass, 4)
        m1, m2, jac = self._mass_grid(n_sub)
        mfun, mdef = jp.MASS_MODELS[mass]
        # hard mass edges become logistic steps one ln M sample wide (gradients)
        mp = {**mdef, **(mass_params or {}), "_w": self.info["delta"] / n_sub}
        rp = {**jp.RATE_DEFAULTS[rate], **(rate_params or {})}

        # population density in (ln M, ln q), normalised on the grid
        P = mfun(m1, m2, mp) * jac
        P = P.reshape(self._n_l, n_sub, -1).mean(axis=1)                   # cell averages
        P = P / jnp.sum(P * self._cell)
        P = P * self._cell

        # project onto the table modes, averaging over spins and tides
        Cbar = self._spin_coefficients(
            spin_model, {**jp.SPIN_MODELS[spin_model][1], **(spin_params or {})})     # (r, q, kappa)
        if self.tides:
            W = self._tide_weights(
                tide_model, {**jp.TIDE_MODELS[tide_model][1], **(tide_params or {})})
            P = jnp.einsum("lq,lqk,rqk->lr", P, W, Cbar)
        else:
            P = P @ Cbar[:, :, 0].T                                            # (n_l, rank)

        # K(l_det) = sum_xi w(xi) P(l_det - xi)
        w  = self._zw * self._rate(rate, rp)
        K  = jnp.fft.irfft(jnp.fft.rfft(P, n=self._L1, axis=0)
                           * jnp.fft.rfft(w, n=self._L1)[:, None],
                           n=self._L1, axis=0)[:self._n_d]
        B  = K * self._e4l

        # Omega(f_k) ∝ sum_r sum_i B[i, r] U[i + k, r]  (correlation via reversed B)
        FB = jnp.fft.rfft(B[::-1], n=self._L2, axis=0)
        c  = jnp.fft.irfft(jnp.sum(FB * self._FJ, axis=1), n=self._L2)
        c  = jnp.where(c > OMEGA_REL_CUT * jnp.max(c), c, 0.0)               # round-off -> 0
        S  = c[self._n_d - 1:self._n_d - 1 + self._n_f]
        # S (flattened by the preconditioning) is smooth in ln f: interpolate
        # it on the uniform ln f grid, then restore x^-A and 4 pi^2 f^3/(3 H0^2)
        S = jp.interp_uniform(self._lnf_out, self.lnf[0], self.info["delta"], S)
        lnf = self._lnf_out
        return (S * jnp.exp(-A_PRE * (np.log(T_SUN) + lnf))
                * 4 * np.pi ** 2 * jnp.exp(3 * lnf) / (3 * H0_SI ** 2))

    def make(self, mass, rate, spin_model=None, tide_model="eos_lnpoly", n_sub=None):
        """
        Jitted function
            omega(mass_params, rate_params, spin_params=None, tide_params=None)
        returning Omega_GW(freqs).  Parameters not given keep the model
        defaults (jax_populations.MASS_MODELS, RATE_DEFAULTS, SPIN_MODELS,
        TIDE_MODELS), so pass only the ones you sample.  spin_params and
        tide_params are ignored unless the table was built with spins/tides.
        spin_model None: self.spin_model (gwtc5_default for black holes, a_max >= 0.5;
        uniform_isotropic for neutron stars).

        Omega is >= 0, and exactly 0 below 1e-12 of its peak (OMEGA_REL_CUT), i.e.
        above the last ringdown, where the FFTs leave only round-off.  A likelihood
        in log Omega must keep its band below that (or add a floor of its own).

        n_sub: sub-samples per ln M cell for the population density (cell
        averages keep sharp mass cut-offs accurate).  Default 1 for smooth
        models (bpl_double_gaussian), 4 otherwise.

        The default population is checked against the grid (check_support)
        and, with spins, the default spin population for degenerate importance
        weights (check_spins), with NSBH tides Lambda_NS against the waveform's
        limit (check_tides); check the extremes of your prior yourself.
        """
        spin_model = spin_model or self.spin_model
        if mass not in jp.MASS_MODELS:
            raise ValueError(f"unknown mass model '{mass}'; available: {list(jp.MASS_MODELS)}")
        if rate not in jp.RATE_DEFAULTS:
            raise ValueError(f"unknown rate model '{rate}'; available: {list(jp.RATE_DEFAULTS)}")
        if spin_model not in jp.SPIN_MODELS:
            raise ValueError(f"unknown spin model '{spin_model}'; available: {list(jp.SPIN_MODELS)}")
        if tide_model not in jp.TIDE_MODELS:
            raise ValueError(f"unknown tide model '{tide_model}'; available: {list(jp.TIDE_MODELS)}")
        n_sub = n_sub or DEFAULT_N_SUB.get(mass, 4)
        self._mass_grid(n_sub)
        self.check_support(mass, None, n_sub)                 # warns if the defaults leak out
        if self.spins:
            self.check_spins(spin_model)                      # warns if the spin weights degenerate
        if self.tides and self.info["waveform"] in NSBH_MODELS:
            self.check_tides(mass, None, tide_model, None, n_sub)   # warns if Lambda_NS > model limit
        if rate == "madau_dickinson_timedelay" and rate not in self._td_cache:
            self._td_cache[rate] = jp.TimeDelayedMadauDickinson(self.cosmo)

        @jax.jit
        def omega(mass_params=None, rate_params=None, spin_params=None, tide_params=None):
            return self.omega(mass, rate, mass_params, rate_params, spin_params, tide_params,
                              n_sub, spin_model, tide_model)
        return omega
