# fast_omega.py
# Differentiable, jit-able Omega_GW(f; Lambda) for MCMC with numpyro, with
# optional spins (aligned or precessing) and tides.
#
#   fo    = FastOmega(freqs, waveform="IMRPhenomXPHM", spins="precessing")
#   omega = fo.make("bpl_double_gaussian", "madau_dickinson")
#   omega(mass_params, rate_params, spin_params)                  # ~ms, (Nf,)
#
# Method (see notes/fast_omega_note.pdf).  In GR the intensity of a source
# with dimensionless intrinsic parameters theta (q, spins at a fixed
# dimensionless reference frequency, tidal deformabilities) scales as
#     I(f; M, theta) = M^4 J(M f; theta)          (exact, checked to 1e-13).
# J is tabulated once per waveform model as "columns": q nodes x kappa2T
# nodes (tides) x spin samples (Sobol, per q node), each column averaged
# exactly over the observer direction (orientation.py), and compressed by an
# SVD, J(x; col) ≈ sum_r B_r(x) C_r(col).  Per call only weights change:
#   spins: self-normalised importance weights p(s | Lambda_s) / g(s) per q node
#   tides: Lambda(m) from an EOS model on the source-frame mass grid (one
#          deterministic EOS), kappa2T linearly interpolated between nodes
# giving P_r(l) = sum_{q,kappa} P(l, q) W(l, q, kappa) V_r(q, kappa).  The
# redshift integral is then an FFT convolution in ln M and the frequency
# dependence an FFT cross-correlation in ln f, exactly as without spins.
#
# Requirements: float64 in jax; the (M, q) grid must contain the population
# support for every Lambda of the prior; spin magnitudes live in [0, a_max]
# and kappa2T in [0, kappa_max].

import hashlib
import os
import time
import warnings
from multiprocessing import Pool

import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import CubicSpline
from scipy.stats import qmc

from .constants import H0_SI, YR_S, Z_MIN_DEFAULT, Z_MAX_DEFAULT
from .cosmology import COSMO
from .orientation import OrientationAverager, TIDAL_MODELS, PRECESSING_MODELS
from . import jax_populations as jp

T_SUN = 4.925490947641267e-06        # G Msun / c^3 in s

# Preconditioning exponent for the frequency correlation.  With
# ln x = ln T_sun + ln M_det + ln f, a factor x^A can be moved exactly from
# J onto M_det and f.  A = 7/3 flattens the inspiral J ∝ x^(-7/3), so the FFT
# no longer mixes terms spanning ~20 decades.
A_PRE = 7.0 / 3.0

# sub-samples per ln M cell: 1 suffices for smooth (tapered) mass models;
# models with hard edges need cell averages
DEFAULT_N_SUB = {"bpl_double_gaussian": 1}



def _next_fast_len(n):
    return int(2 ** np.ceil(np.log2(n)))


# ----------------------------------------------------------------------
# table columns (module level so that they can run in a process pool)
# ----------------------------------------------------------------------

_AVG = None


def _init_table_worker(waveform, rule):
    global _AVG
    _AVG = OrientationAverager(waveform, rule)


def _table_column(task):
    """task = (f, f_ref, m_ref, q, spin 5-vector, aligned, lambdas or None) -> (n_kappa, n_f)."""
    f, f_ref, m_ref, q, s, aligned, lambdas = task
    a1, a2, c1, c2, p12 = s
    # IMRPhenomXP/XPHM jump at exactly m1 == m2 (+10% at merger, x18 in the
    # ringdown tail, while XAS/XHM are continuous): evaluate the q -> 1 limit
    q = min(q, 1.0 - 1e-6)
    m1, m2 = m_ref / (1 + q), m_ref * q / (1 + q)
    # aligned: keep only the projections a cos t on L (in-plane components dropped)
    r1, r2 = ((0.0, 0.0) if aligned else
              (np.sqrt(max(0.0, 1 - c1 * c1)), np.sqrt(max(0.0, 1 - c2 * c2))))
    s1 = (a1 * r1, 0.0, a1 * c1)
    s2 = (a2 * r2 * np.cos(p12), a2 * r2 * np.sin(p12), a2 * c2)
    out = []
    for lam in (lambdas if lambdas is not None else [None]):
        out.append(_AVG(f, m1, m2, s1, s2, f_ref, lam) / m_ref ** 4)
    return np.array(out)


class FastOmega:
    """
    freqs          frequencies of the output (default 5..1000 Hz, 1 Hz)
    waveform       LAL approximant.  Precessing spins need a precessing model
                   (IMRPhenomXPHM, IMRPhenomXP, IMRPhenomXP_NRTidalv2), tides
                   an NRTidal model (IMRPhenomXP_NRTidalv2, IMRPhenomXAS_NRTidalv2)
    spins          None, "aligned" or "precessing"
    tides          include tidal deformabilities (EOS model Lambda(m))
    m_tot_range    source-frame total mass range (Msun) of the grid; must
                   contain the support of every population you evaluate
    q_min          smallest mass ratio of the grid (same requirement)
    z_min, z_max   redshift range of the integral
    delta          spacing of the ln M, ln(1+z) and ln f grids
    n_q            points of the ln q grid (odd: Simpson weights)
    n_spin         Sobol spin samples per q node (spins != None)
    a_max          spin magnitudes live in [0, a_max] (use ~0.05 for BNS)
    x_ref          spins are defined at f_ref = x_ref / (M T_sun), i.e. at a
                   fixed M f (default 0.005, i.e. 20 Hz for M ~ 50 Msun; with
                   tides 3e-4, i.e. 20 Hz for M ~ 3 Msun)
    kappa_min, kappa_max, n_kappa   kappa2T grid (tides): 0 plus n_kappa-1 nodes
                   log-spaced on [kappa_min, kappa_max]; a binary's kappa2T is
                   shared linearly (in kappa) between its two neighbouring
                   nodes.  NRTidal tapers the waveform at a kappa-dependent
                   merger frequency, so J is very non-linear in kappa near 0 and
                   a uniform grid overestimates the high-frequency power.
                   kappa2T = 3/16 Lambda-tilde: ~10-400 for BNS, O(1) for NSBH
    table_step     the J table is computed on an ln x grid table_step * delta
                   apart and its SVD basis interpolated to the delta grid
    rank           SVD rank of the table; None picks the smallest rank with
                   sigma_r / sigma_1 < rank_tol (12..96; rank_tol 1e-4 keeps the
                   truncation error of Omega at ~1e-5)
    orientation_rule  see orientation.OrientationAverager (None: exact default)
    m_ref          total mass at which the table is generated (default 20,
                   2.8 with tides)
    n_proc         processes for building the table
    cache_dir      the table is cached here (None disables caching)
    seed           seed of the spin samples
    """

    def __init__(self, freqs=None, waveform="IMRPhenomXAS", spins=None, tides=False,
                 waveform_kwargs=None, m_tot_range=(1.5, 700.0), q_min=0.01,
                 z_min=Z_MIN_DEFAULT, z_max=Z_MAX_DEFAULT, delta=0.005, n_q=121,
                 n_spin=128, a_max=1.0, x_ref=None, kappa_min=0.1, kappa_max=3000.0,
                 n_kappa=24,
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
        self.spins, self.tides, self.a_max = spins, bool(tides), float(a_max)
        x_ref  = x_ref if x_ref is not None else (3e-4 if tides else 0.005)
        m_ref  = m_ref if m_ref is not None else (2.8 if tides else 20.0)
        d = delta

        # ---- grids (all logarithmic grids share the spacing d) ----
        self.l_src = np.log(m_tot_range[0]) + d * np.arange(
            int(np.ceil(np.log(m_tot_range[1] / m_tot_range[0]) / d)) + 1)
        x0, x1 = np.log1p(z_min), np.log1p(z_max)
        n_z = int(np.floor((x1 - x0) / d)) + 1
        self.x = x0 + d * np.arange(n_z)
        wx = np.full(n_z, d)
        wx[0] *= 0.5
        wx[-1] = 0.5 * d + (x1 - self.x[-1])          # close the last partial interval
        self.z = np.expm1(self.x)

        if n_q % 2 == 0:
            n_q += 1
        self.lq = np.linspace(np.log(q_min), 0.0, n_q)
        hq = self.lq[1] - self.lq[0]
        self.wq = np.full(n_q, 2.0)
        self.wq[1::2] = 4.0
        self.wq[[0, -1]] = 1.0
        self.wq *= hq / 3.0                                              # Simpson

        n_l, n_d = len(self.l_src), len(self.l_src) + n_z - 1
        self.l_det = self.l_src[0] + self.x[0] + d * np.arange(n_d)
        lnf0 = np.log(self.freqs.min())
        n_f  = int(np.ceil(np.log(self.freqs.max() / self.freqs.min()) / d)) + 1
        self.lnf = lnf0 + d * np.arange(n_f)
        n_x  = n_d + n_f - 1
        self.ln_x = np.log(T_SUN) + self.l_det[0] + lnf0 + d * np.arange(n_x)

        # ---- the intensity table: columns (q, kappa, spin sample) ----
        n_s = n_spin if spins else 1
        n_k = n_kappa if tides else 1
        self.kappa = (np.concatenate([[0.0], np.geomspace(kappa_min, kappa_max, n_k - 1)])
                      if tides else np.zeros(1))
        self._spin_samples = self._draw_spins(n_q, n_s, seed)              # (n_q, n_s, 5)
        dJ = table_step * d
        lx_hi = min(self.ln_x[-1], 0.0)                                    # every model ends by M f = 1
        self.ln_xJ = np.arange(self.ln_x[0] - 2 * dJ, lx_hi + 2 * dJ, dJ)
        Jcols, t_table, cache_file = self._build_table(
            waveform, orientation_rule, m_ref, x_ref, n_proc, cache_dir,
            dict(table_version=2, spins=spins, tides=self.tides, n_s=n_s, seed=seed, a_max=self.a_max,
                 kappa=self.kappa.tolist(), x_ref=x_ref, m_ref=m_ref, lq=self.lq.tolist(),
                 lxJ=[self.ln_xJ[0], dJ, len(self.ln_xJ)], rule=str(orientation_rule)),
            verbose)                                                       # (n_q, n_k, n_s, n_xJ)

        # ---- SVD of the table and interpolation of the basis to the fine grid ----
        A   = Jcols.reshape(-1, len(self.ln_xJ)).T                         # (n_xJ, n_col)
        sx  = np.sqrt(np.mean(A ** 2, axis=1))
        An  = np.where(sx[:, None] > 0, A / np.where(sx > 0, sx, 1.0)[:, None], 0.0)
        colw = np.sqrt(np.repeat(self.wq, n_k * n_s) / n_s)
        u, sv, vt = np.linalg.svd(An * colw[None, :], full_matrices=False)
        if rank is None:
            # Omega's truncation error is ~0.03-0.1 x rank_tol (measured); at least 12
            rank = int(min(96, max(12, np.sum(sv / sv[0] > rank_tol))))
        self._sv = sv / sv[0]
        coef = (vt[:rank] / colw[None, :]).reshape(rank, n_q, n_k, n_s)
        B    = u[:, :rank] * sv[:rank]                                     # normalised basis
        live = sx > 0
        xe   = self.ln_xJ[live][-1]                                        # last live table node
        spl  = CubicSpline(self.ln_xJ, B, axis=0)
        ls   = np.log(np.where(live, sx, sx[live].min()))
        s_f  = np.exp(np.interp(self.ln_x, self.ln_xJ, ls))
        s_f[self.ln_x > xe] = 0.0
        B_f  = spl(self.ln_x) * s_f[:, None]                               # (n_x, rank)

        # ---- constants of the Lambda-dependent part ----
        self._mass_grids = {}
        self._cell = jnp.asarray(d * self.wq)[None, :]                     # dlnM dlnq weights
        dL = cosmo.luminosity_distance(self.z)
        # w(x) dx = (1+z) dN/dz / dL^2 dx and dN/dz = R/(1+z) dVc/dz, R in Gpc^-3/yr
        self._zw  = jnp.asarray(wx * cosmo.dVc_dz(self.z) / dL ** 2 / (1e9 * YR_S))
        self._e4l = jnp.asarray(np.exp((4 - A_PRE) * self.l_det))[:, None]   # M_det^(4-A)
        self._L1  = _next_fast_len(n_l + n_z - 1)                          # K convolution
        self._L2  = _next_fast_len(n_d + n_x - 1)                          # f correlation
        xa = np.exp(A_PRE * self.ln_x)[:, None]                            # x^A
        self._FJ  = jnp.fft.rfft(jnp.asarray(B_f * xa), n=self._L2, axis=0)
        self._C   = jnp.asarray(coef)
        self._B   = B_f                                                    # (n_x, rank), for table_intensity
        self._n_l, self._n_z, self._n_d, self._n_f = n_l, n_z, n_d, n_f
        self._lnf_out = jnp.asarray(np.log(self.freqs))
        self._td_cache = {}
        self.cosmo = cosmo

        if spins:
            s = self._spin_samples
            self._spin_arrays = tuple(jnp.asarray(s[..., i]) for i in range(4))   # a1 a2 c1 c2
            self._g_spin = 1.0 / (4.0 * self.a_max ** 2)                   # proposal density
        if self.tides:
            M = np.exp(self.l_src)[:, None]
            q = np.exp(self.lq)[None, :]
            self._m1c, self._m2c = jnp.asarray(M / (1 + q)), jnp.asarray(M * q / (1 + q))
            c1, c2 = jp.kappa2T_coefficients(np.exp(self.lq))
            self._kc1, self._kc2 = jnp.asarray(c1)[None, :], jnp.asarray(c2)[None, :]
            self._lk0, self._dlk = np.log(self.kappa[1]), np.log(self.kappa[2] / self.kappa[1])

        self.info = dict(n_lnM=n_l, n_z=n_z, n_q=n_q, n_lnf=n_f, n_x=n_x, n_xJ=len(self.ln_xJ),
                         n_spin=n_s, n_kappa=n_k, n_columns=Jcols.shape[0] * n_k * n_s,
                         rank=rank, delta=d, waveform=waveform, spins=spins, tides=self.tides,
                         x_ref=x_ref, m_ref=m_ref, table_seconds=t_table, cache_file=cache_file,
                         setup_seconds=time.time() - t0)
        if verbose:
            print(f"FastOmega: {waveform}, spins={spins}, tides={self.tides}: "
                  f"{self.info['n_columns']} table columns, rank {rank}; "
                  f"grids lnM {n_l}, z {n_z}, lnq {n_q}, lnf {n_f}; "
                  f"setup {self.info['setup_seconds']:.1f} s", flush=True)

    # ------------------------------------------------------------------

    def _draw_spins(self, n_q, n_s, seed):
        """(n_q, n_s, 5) = a1, a2, cos t1, cos t2, phi12; one Sobol scrambling per q node."""
        out = np.zeros((n_q, n_s, 5))
        if not self.spins:
            out[..., 2:4] = 1.0
            return out
        dim = 5 if self.spins == "precessing" else 4
        for k in range(n_q):
            u = qmc.Sobol(dim, scramble=True, seed=np.random.default_rng([seed, k])).random(n_s)
            u = np.clip(u, 1e-9, 1 - 1e-9)
            out[k, :, 0] = self.a_max * u[:, 0]
            out[k, :, 1] = self.a_max * u[:, 1]
            out[k, :, 2] = 2 * u[:, 2] - 1
            out[k, :, 3] = 2 * u[:, 3] - 1
            out[k, :, 4] = 2 * np.pi * u[:, 4] if dim == 5 else 0.0
        return out

    def _build_table(self, waveform, rule, m_ref, x_ref, n_proc, cache_dir, key, verbose):
        n_q, n_s = self._spin_samples.shape[:2]
        n_k = len(self.kappa)
        cache_file = None
        if cache_dir:
            cache_dir = os.path.expanduser(cache_dir)
            tag = repr(sorted(key.items())) + waveform
            h = hashlib.sha1(tag.encode()).hexdigest()[:16]
            cache_file = os.path.join(cache_dir, f"jtable_{waveform}_{h}.npz")
            if os.path.exists(cache_file):
                if verbose:
                    print(f"FastOmega: table loaded from {cache_file}", flush=True)
                return np.load(cache_file)["J"], 0.0, cache_file

        t0 = time.time()
        xJ   = np.exp(self.ln_xJ)
        live = xJ < 1.0
        f    = xJ[live] / (m_ref * T_SUN)
        fref = x_ref / (m_ref * T_SUN)
        tasks = []
        for k, q in enumerate(np.exp(self.lq)):
            lambdas = None
            if self.tides:
                c1, c2 = jp.kappa2T_coefficients(q)
                lambdas = [(kap / (c1 + c2), kap / (c1 + c2)) for kap in self.kappa]
            for i in range(n_s):
                tasks.append((f, fref, m_ref, q, tuple(self._spin_samples[k, i]),
                              self.spins != "precessing", lambdas))
        if verbose:
            print(f"FastOmega: building the {waveform} table: {len(tasks)} spin/q columns "
                  f"x {n_k} kappa nodes, {len(f)} frequencies each ...", flush=True)
        if n_proc > 1:
            with Pool(n_proc, initializer=_init_table_worker, initargs=(waveform, rule)) as pool:
                cols = pool.map(_table_column, tasks, chunksize=max(1, len(tasks) // (8 * n_proc)))
        else:
            _init_table_worker(waveform, rule)
            cols = [_table_column(t) for t in tasks]
        J = np.zeros((n_q, n_k, n_s, len(xJ)))
        J[..., live] = np.array(cols).reshape(n_q, n_s, n_k, -1).transpose(0, 2, 1, 3)
        dt = time.time() - t0
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.savez(cache_file, J=J)
        if verbose:
            print(f"FastOmega: table built in {dt:.0f} s" +
                  (f", cached at {cache_file}" if cache_file else ""), flush=True)
        return J, dt, cache_file

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
            if rate not in self._td_cache:          # built eagerly by make()
                self._td_cache[rate] = jp.TimeDelayedMadauDickinson(self.z, self.cosmo)
            return self._td_cache[rate](self.z, params)
        return jp.RATE_MODELS[rate](jnp.asarray(self.z), params)

    def _spin_coefficients(self, spin_model, sp):
        """V_r(q, kappa): spin-averaged table coefficients, self-normalised per q node."""
        if not self.spins:
            return self._C[..., 0]
        fn, _ = jp.SPIN_MODELS[spin_model]
        a1, a2, c1, c2 = self._spin_arrays
        w = fn(a1, a2, c1, c2, sp, self.a_max) / self._g_spin                 # (n_q, n_s)
        return jnp.einsum("rkts,ks->rkt", self._C, w) / jnp.sum(w, axis=1)[None, :, None]

    def _tide_weights(self, tide_model, tp):
        """W(l, q, kappa): hat-function weights of kappa2T(m1, m2) on the kappa nodes."""
        fn, _ = jp.TIDE_MODELS[tide_model]
        w_edge = self.info["delta"]
        kap  = (self._kc1 * fn(self._m1c, tp, w_edge)
                + self._kc2 * fn(self._m2c, tp, w_edge))                       # (l, q)
        n_k  = len(self.kappa)
        # nodes are 0 and log-spaced; the interval is found in ln kappa, and the
        # weights are linear in kappa inside it (exact for the linear small-kappa
        # response, dense nodes resolve the non-linear taper near kappa = 0)
        kn   = jnp.asarray(self.kappa)
        kc   = jnp.clip(kap, 0.0, self.kappa[-1])
        lk   = jnp.log(jnp.maximum(kc, self.kappa[1]))
        i0   = jnp.where(kc < self.kappa[1], 0,
                         1 + jnp.floor((lk - self._lk0) / self._dlk).astype(jnp.int32))
        i0   = jnp.clip(i0, 0, n_k - 2)
        fr   = jnp.clip((kc - kn[i0]) / (kn[i0 + 1] - kn[i0]), 0.0, 1.0)
        return ((1.0 - fr)[..., None] * jax.nn.one_hot(i0, n_k)
                + fr[..., None] * jax.nn.one_hot(i0 + 1, n_k))                  # (l, q, kappa)

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
        lq = np.linspace(self.lq[0] - np.log(10.0), 0.0, 800)[None, :]
        M, q = np.exp(l), np.exp(lq)
        P = np.asarray(mfun(jnp.asarray(M / (1 + q)), jnp.asarray(M * q / (1 + q)), mp))
        P = np.nan_to_num(P) * M ** 2 * q / (1 + q) ** 2
        tot = P.sum()
        if not tot > 0:
            raise ValueError(f"the {mass} density vanishes on the extended grid")
        out = dict(below_M=float(P[l[:, 0] < lo].sum() / tot),
                   above_M=float(P[l[:, 0] > hi].sum() / tot),
                   below_q=float(P[:, lq[0] < self.lq[0]].sum() / tot))
        out["total"] = float(1.0 - P[(l[:, 0] >= lo) & (l[:, 0] <= hi)][:, lq[0] >= self.lq[0]].sum() / tot)
        if warn and out["total"] > tol:
            warnings.warn(
                f"FastOmega: {out['total']:.2%} of the {mass} population lies outside the grid "
                f"(M < {np.exp(lo):.3g}: {out['below_M']:.2%}, M > {np.exp(hi):.3g}: "
                f"{out['above_M']:.2%}, q < {np.exp(self.lq[0]):.3g}: {out['below_q']:.2%}) "
                f"and is dropped; widen m_tot_range / q_min", stacklevel=2)
        return out

    def table_intensity(self, spin_params=None, spin_model="lvk_default", kappa_index=0):
        """
        Orientation- and spin-averaged J(x; q) reconstructed from the table,
        shape (len(self.ln_x), n_q), at the kappa node `kappa_index`.  For
        diagnostics: compare with a direct average of OrientationAverager.
        """
        V = self._spin_coefficients(
            spin_model, {**jp.SPIN_MODELS[spin_model][1], **(spin_params or {})})
        return self._B @ np.asarray(V[:, :, kappa_index])

    # ------------------------------------------------------------------

    def omega(self, mass, rate, mass_params=None, rate_params=None, spin_params=None,
              tide_params=None, n_sub=None, spin_model="lvk_default", tide_model="eos_lnpoly"):
        """Omega_GW at self.freqs (not jitted; see make())."""
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
        V = self._spin_coefficients(
            spin_model, {**jp.SPIN_MODELS[spin_model][1], **(spin_params or {})})     # (r, q, kappa)
        if self.tides:
            W = self._tide_weights(
                tide_model, {**jp.TIDE_MODELS[tide_model][1], **(tide_params or {})})
            P = jnp.einsum("lq,lqk,rqk->lr", P, W, V)
        else:
            P = P @ V[:, :, 0].T                                               # (n_l, rank)

        # K(l_det) = sum_x w(x) P(l_det - x)
        w  = self._zw * self._rate(rate, rp)
        K  = jnp.fft.irfft(jnp.fft.rfft(P, n=self._L1, axis=0)
                           * jnp.fft.rfft(w, n=self._L1)[:, None],
                           n=self._L1, axis=0)[:self._n_d]
        B  = K * self._e4l

        # Omega(f_k) ∝ sum_r sum_i B[i, r] U[i + k, r]  (correlation via reversed B)
        FB = jnp.fft.rfft(B[::-1], n=self._L2, axis=0)
        c  = jnp.fft.irfft(jnp.sum(FB * self._FJ, axis=1), n=self._L2)
        S  = c[self._n_d - 1:self._n_d - 1 + self._n_f]
        # S (flattened by the preconditioning) is smooth in ln f: interpolate
        # it on the uniform ln f grid, then restore x^-A and 4 pi^2 f^3/(3 H0^2)
        S = jp.interp_uniform(self._lnf_out, self.lnf[0], self.info["delta"], S)
        lnf = self._lnf_out
        return (S * jnp.exp(-A_PRE * (np.log(T_SUN) + lnf))
                * 4 * np.pi ** 2 * jnp.exp(3 * lnf) / (3 * H0_SI ** 2))

    def make(self, mass, rate, spin_model="lvk_default", tide_model="eos_lnpoly", n_sub=None):
        """
        Jitted function
            omega(mass_params, rate_params, spin_params=None, tide_params=None)
        returning Omega_GW(freqs).  Parameters not given keep the model
        defaults (jax_populations.MASS_MODELS, RATE_DEFAULTS, SPIN_MODELS,
        TIDE_MODELS), so pass only the ones you sample.  spin_params and
        tide_params are ignored unless the table was built with spins/tides.

        n_sub: sub-samples per ln M cell for the population density (cell
        averages keep sharp mass cut-offs accurate).  Default 1 for smooth
        models (bpl_double_gaussian), 4 otherwise.

        The default population is checked against the grid (check_support);
        check the extremes of your prior yourself.
        """
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
        if rate == "madau_dickinson_timedelay" and rate not in self._td_cache:
            self._td_cache[rate] = jp.TimeDelayedMadauDickinson(self.z, self.cosmo)

        @jax.jit
        def omega(mass_params=None, rate_params=None, spin_params=None, tide_params=None):
            return self.omega(mass, rate, mass_params, rate_params, spin_params, tide_params,
                              n_sub, spin_model, tide_model)
        return omega
