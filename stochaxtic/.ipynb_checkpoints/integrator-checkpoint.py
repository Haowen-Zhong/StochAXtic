# Monte Carlo Omega_GW(f) of a compact-binary population, all frequencies at once:
#
#   Omega_GW(f) = 4 pi^2 f^3 / (3 H0^2)
#                 * int dz dN/(dz dt) < |h+(f)|^2 + |hx(f)|^2 >_{m1, m2, iota, phi}
#
# with the waveform at dL(z) and detector-frame masses m(1+z).  Omega_GW is a property
# of the population alone (no overlap reduction function).
#
# 1) The waveform depends on z only through Mc_det = Mc (1+z) and 1/dL, so with
#    I(f) = I_1Mpc(f; Mc_det, q, iota, phi) / dL^2 the z integral is a deterministic
#    kernel
#        K(Mc_det, q) = int dz dN/(dz dt) p(Mc_det/(1+z), q) / ((1+z) dL^2)
#    evaluated by the trapezoid rule on a grid uniform in ln(1+z).
# 2) theta = (ln Mc_det, ln q, cos iota, phi) is integrated by randomised
#    quasi Monte Carlo (scrambled Sobol) with an importance proposal g built
#    from K itself.  One waveform call returns every frequency.
#        Omega(f) = 4 pi^2 f^3/(3 H0^2) E_g[ Mc_det q K / (2 g) I_1Mpc(f) ]
# 3) Independent scramblings (replicas) give the error bar.
#
# Unbiased for any proposal that covers the integrand (the defensive uniform part
# eps > 0 guarantees it); the proposal only affects the variance.  No waveform
# symmetry is assumed.

import time
from dataclasses import dataclass, field
from multiprocessing import Pool

import numpy as np
from scipy.stats import qmc

from .constants import H0_SI, S_REF, Z_MIN_DEFAULT, Z_MAX_DEFAULT
from .cosmology import COSMO, get_rate_model
from .mass_models import get_mass_model
from .waveforms import get_waveform_model


# ==========================================================
# Redshift kernel
# ==========================================================

class RedshiftKernel:
    """
    K(Mc_det, q) = int dz dN/(dz dt)(z) p(Mc_det/(1+z), q) / ((1+z) dL(z)^2)

    in s^-1 Mpc^-2 per unit (Mc_det, q), dL in Mpc.  With xi = ln(1+z),
    dz = (1+z) dxi cancels the 1/(1+z) from dMc = dMc_det/(1+z).
    """

    def __init__(self, mass_model, rate_model, z_min, z_max, n_z, cosmo=COSMO):
        self.mass = mass_model
        xi  = np.linspace(np.log1p(z_min), np.log1p(z_max), n_z)
        z   = np.expm1(xi)
        wxi = np.full(n_z, xi[1] - xi[0])
        wxi[[0, -1]] *= 0.5
        dNdz = rate_model.event_rate_density(z, cosmo)                # s^-1
        self.z      = z
        self.weight = wxi * dNdz / cosmo.luminosity_distance(z) ** 2   # (n_z,)
        self.N_tot  = float(np.sum(wxi * (1 + z) * dNdz))              # events / s

    def __call__(self, Mc_det, q, chunk=2048):
        """K for arrays Mc_det, q of shape (c,)."""
        out = np.empty(len(Mc_det))
        for s in range(0, len(Mc_det), chunk):
            mc = Mc_det[s:s + chunk, None] / (1.0 + self.z[None, :])
            out[s:s + chunk] = self.mass.density_mc_q(mc, q[s:s + chunk, None]) @ self.weight
        return out


# ==========================================================
# Proposal over (ln Mc_det, ln q, cos iota, phi)
# ==========================================================

class Proposal:
    """
    Piecewise-constant density in the measure d ln Mc_det d ln q d cos(iota),
    sampled by inverse CDF, so the density of every drawn point is exact.

    (ln Mc_det, ln q): n_mc x n_q cells,
        g2 = eps / area + (1-eps) mean_t[ b Mc_det^t / norm_t ],
        b  = Mc_det q K(Mc_det, q) at the cell centres.
        t > 0 favours heavy systems (low f), t < 0 light ones (high f); the
        uniform part keeps every cell of the rectangle sampled.
    cos(iota): g_u ∝ A(u)^iota_power, A(u) = ((1+u^2)/2)^2 + u^2, the
        quadrupole inclination factor of I.  Symmetric in u.
    phi: uniform on [0, 2 pi).

    Defaults tuned for Omega_GW (bpl_double_gaussian + madau_dickinson, XAS,
    5-1000 Hz, 16 replicas x 2^11 points): median / max relative error
    0.05% / 0.13%.  Antithetic pairs in cos(iota) only double the cost (I is
    even in cos(iota) for quadrupole-dominated signals).
    """

    def __init__(self, kernel, mass_model, z_min, z_max, n_mc=200, n_q=60,
                 eps=0.1, tilts=(-5.0, -2.0, 1.0, 4.0), iota_power=1.0, n_u=2000):
        self.mc_edges = np.linspace(np.log(mass_model.mc_min * (1 + z_min)),
                                    np.log(mass_model.mc_max * (1 + z_max)), n_mc + 1)
        self.q_edges  = np.linspace(np.log(mass_model.q_min), 0.0, n_q + 1)
        self.dmc = self.mc_edges[1] - self.mc_edges[0]
        self.dq  = self.q_edges[1] - self.q_edges[0]

        c_mc = 0.5 * (self.mc_edges[1:] + self.mc_edges[:-1])
        c_q  = 0.5 * (self.q_edges[1:] + self.q_edges[:-1])
        LM, LQ = np.meshgrid(c_mc, c_q, indexing="ij")
        Mc, Q  = np.exp(LM).ravel(), np.exp(LQ).ravel()
        base   = (Mc * Q * kernel(Mc, Q)).reshape(n_mc, n_q)
        if not base.sum() > 0:
            raise ValueError("the population has no support on the (Mc_det, q) grid")

        area = self.dmc * self.dq
        dens = np.full((n_mc, n_q), eps / (n_mc * n_q * area))
        for t in tilts:
            score = base * np.exp(LM) ** t
            dens += (1 - eps) / len(tilts) * score / (score.sum() * area)
        self.dens2  = dens
        cell_p      = dens * area
        self.p_mc   = cell_p.sum(axis=1)
        self.cdf_mc = np.concatenate([[0.0], np.cumsum(self.p_mc)])
        self.cdf_q  = np.concatenate([np.zeros((n_mc, 1)), np.cumsum(cell_p, axis=1)],
                                     axis=1) / self.p_mc[:, None]

        self.u_edges = np.linspace(-1.0, 1.0, n_u + 1)
        u_c = 0.5 * (self.u_edges[1:] + self.u_edges[:-1])
        pu  = (((1 + u_c ** 2) / 2) ** 2 + u_c ** 2) ** iota_power
        pu /= pu.sum()
        self.dens_u = pu / np.diff(self.u_edges)
        self.cdf_u  = np.concatenate([[0.0], np.cumsum(pu)])

    @staticmethod
    def _inverse(cdf, x):
        i    = np.clip(np.searchsorted(cdf, x, side="right") - 1, 0, len(cdf) - 2)
        frac = (x - cdf[i]) / np.maximum(cdf[i + 1] - cdf[i], 1e-300)
        return i, np.clip(frac, 0.0, 1.0)

    def sample(self, x):
        """x: (N, 4) uniforms -> Mc_det, q, u, phi and g (measure d ln Mc d ln q du)."""
        i, fi = self._inverse(self.cdf_mc, x[:, 0])
        j  = np.empty(len(x), dtype=int)
        fj = np.empty(len(x))
        for cell in np.unique(i):
            sel = i == cell
            j[sel], fj[sel] = self._inverse(self.cdf_q[cell], x[sel, 1])
        k, fk = self._inverse(self.cdf_u, x[:, 2])

        Mc_det = np.exp(self.mc_edges[i] + fi * self.dmc)
        q      = np.exp(self.q_edges[j] + fj * self.dq)
        u      = self.u_edges[k] + fk * (self.u_edges[k + 1] - self.u_edges[k])
        phi    = 2.0 * np.pi * x[:, 3]
        return Mc_det, q, u, phi, self.dens2[i, j] * self.dens_u[k]


# ==========================================================
# Waveform evaluation (optionally in a process pool)
# ==========================================================

_WF, _FREQS = None, None


def _init_worker(wf_name, wf_kwargs, freqs):
    global _WF, _FREQS
    _WF, _FREQS = get_waveform_model(wf_name, **wf_kwargs), freqs


def _intensity_one(args):
    m1, m2, iota, phi, s1, s2, f_ref = args
    try:
        return _WF.intensity(_FREQS, m1, m2, iota, phi, s1, s2, f_ref) / S_REF, True
    except Exception:
        return np.zeros(len(_FREQS)), False


# ==========================================================
# Result container
# ==========================================================

@dataclass
class OmegaResult:
    """
    freqs          (Nf,)    Hz
    omega          (Nf,)    Omega_GW(f), mean over replicas
    omega_err      (Nf,)    std over replicas / sqrt(R)   (the error to quote)
    omega_err_mc   (Nf,)    plain-MC error from within-replica variance
                            (conservative for QMC; ~10x larger)
    replicas       (R, Nf)  per-replica estimates
    ess            (R, Nf)  effective sample size / N per replica and f;
                            values << 0.01 flag an inadequate proposal
    meta           dict     settings and bookkeeping
    """
    freqs:        np.ndarray
    omega:        np.ndarray
    omega_err:    np.ndarray
    omega_err_mc: np.ndarray
    replicas:     np.ndarray
    ess:          np.ndarray
    meta:         dict = field(default_factory=dict)

    @classmethod
    def from_replicas(cls, freqs, replicas, var_of_mean, ess, meta):
        replicas = np.atleast_2d(replicas)
        R = len(replicas)
        err_mc = np.sqrt(np.sum(np.atleast_2d(var_of_mean), axis=0)) / R
        err = replicas.std(axis=0, ddof=1) / np.sqrt(R) if R > 1 else err_mc
        return cls(freqs, replicas.mean(axis=0), err, err_mc, replicas,
                   np.atleast_2d(ess), dict(meta, var_of_mean=np.atleast_2d(var_of_mean)))

    def save(self, path):
        meta = {f"meta_{k}": (np.asarray(v) if not isinstance(v, dict) else repr(v))
                for k, v in self.meta.items()}
        np.savez(path, freqs=self.freqs, omega=self.omega, omega_err=self.omega_err,
                 omega_err_mc=self.omega_err_mc, replicas=self.replicas,
                 ess=self.ess, **meta)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        meta = {k[5:]: d[k][()] if d[k].ndim == 0 else d[k]
                for k in d.files if k.startswith("meta_")}
        return cls(d["freqs"], d["omega"], d["omega_err"], d["omega_err_mc"],
                   d["replicas"], d["ess"], meta)

    @classmethod
    def merge(cls, results):
        """Combine results with disjoint replica sets and identical settings."""
        r0 = results[0]
        for r in results[1:]:
            if not np.array_equal(r.freqs, r0.freqs):
                raise ValueError("frequency grids differ")
            for k in ("mass_model", "rate_model", "waveform", "seed", "settings"):
                if str(r.meta.get(k)) != str(r0.meta.get(k)):
                    raise ValueError(f"results disagree on {k}")
        reps = np.concatenate([np.atleast_1d(r.meta["rep_ids"]) for r in results])
        if len(set(reps.tolist())) != len(reps):
            raise ValueError(f"duplicate replicas: {sorted(reps.tolist())}")
        meta = dict(r0.meta, rep_ids=reps,
                    n_failed=int(sum(int(r.meta["n_failed"]) for r in results)))
        meta.pop("var_of_mean", None)
        return cls.from_replicas(
            r0.freqs, np.concatenate([r.replicas for r in results]),
            np.concatenate([np.atleast_2d(r.meta["var_of_mean"]) for r in results]),
            np.concatenate([r.ess for r in results]), meta)

    def summary(self):
        rel = np.abs(self.omega_err / np.where(self.omega != 0, self.omega, np.nan))
        return (f"{len(self.replicas)} replicas, {len(self.freqs)} frequencies "
                f"{self.freqs[0]:g}-{self.freqs[-1]:g} Hz: rel. err median "
                f"{np.nanmedian(rel):.2%}, max {np.nanmax(rel):.2%} at "
                f"{self.freqs[np.nanargmax(rel)]:g} Hz; min ESS/N "
                f"{np.nanmin(self.ess):.3f}; {int(self.meta.get('n_failed', 0))} "
                f"failed waveform calls")


# ==========================================================
# Main class
# ==========================================================

class OmegaIntegrator:
    """
    Build once per population, then call run().

        oi  = OmegaIntegrator("bpl_double_gaussian", "madau_dickinson",
                              rate_kwargs={"R0": 15.9})
        res = oi.run(n_rep=8)
        res.freqs, res.omega, res.omega_err

    mass_model, rate_model, waveform: a registered name or an instance.
    freqs: default 5..1000 Hz in 1 Hz steps.
    Proposal settings (eps, tilts, iota_power, n_mc, n_q, n_u) go in
    proposal_kwargs; the defaults are tuned for Omega_GW over 5-1000 Hz.

    spins: None, "aligned" or "precessing".  Spins are drawn uniformly
    (a in [0, a_max], cos t in [-1, 1], phi1 and phi12 in [0, 2 pi)) and weighted by
    the spin model of spin_models.py; they are defined at f_ref =
    x_ref / (M_det T_sun), the same convention as FastOmega.  "aligned"
    keeps only a cos t.  f_ref (Hz) instead fixes the detector-frame reference
    frequency for all sources (the LVK convention, f_ref = 20 Hz); it only
    matters for precessing spins.
    """

    def __init__(self, mass_model="bpl_double_gaussian", rate_model="madau_dickinson",
                 waveform="IMRPhenomXAS", freqs=None, mass_kwargs=None,
                 rate_kwargs=None, waveform_kwargs=None,
                 z_min=Z_MIN_DEFAULT, z_max=Z_MAX_DEFAULT, n_z=1000,
                 n_z_proposal=200, proposal_kwargs=None, spins=None,
                 spin_model="gwtc5_default", spin_params=None, a_max=1.0, x_ref=0.005,
                 f_ref=None, n_proc=1, verbose=True):
        self.mass = (get_mass_model(mass_model, **(mass_kwargs or {}))
                     if isinstance(mass_model, str) else mass_model)
        self.rate = (get_rate_model(rate_model, **(rate_kwargs or {}))
                     if isinstance(rate_model, str) else rate_model)
        self.freqs = (np.arange(5.0, 1001.0, 1.0) if freqs is None
                      else np.asarray(freqs, dtype=float))
        if not isinstance(waveform, str):
            raise TypeError("waveform must be a name (it is rebuilt in worker processes)")
        self.wf_name, self.wf_kwargs = waveform, dict(waveform_kwargs or {})
        self.n_proc, self.verbose = n_proc, verbose
        self.z_min, self.z_max = z_min, z_max
        if spins not in (None, "aligned", "precessing"):
            raise ValueError("spins must be None, 'aligned' or 'precessing'")
        self.spins, self.a_max, self.x_ref, self.f_ref = spins, a_max, x_ref, f_ref
        if spins:
            from .spin_models import SPIN_MODELS
            fn, dflt = SPIN_MODELS[spin_model]
            sp = {**dflt, **(spin_params or {})}
            self._spin_density = lambda a1, a2, c1, c2: fn(a1, a2, c1, c2, sp, a_max)
            self._spin_label = f"{spin_model} {sp}"

        t0 = time.time()
        self.kernel = RedshiftKernel(self.mass, self.rate, z_min, z_max, n_z)
        # the proposal only needs the shape of K, so a coarser z grid suffices
        k_prop = (self.kernel if n_z_proposal in (None, n_z) else
                  RedshiftKernel(self.mass, self.rate, z_min, z_max, n_z_proposal))
        self.proposal_kwargs = dict(proposal_kwargs or {})
        self.proposal = Proposal(k_prop, self.mass, z_min, z_max, **self.proposal_kwargs)
        self.settings = dict(z_min=z_min, z_max=z_max, n_z=n_z, n_z_proposal=n_z_proposal,
                             **{f"proposal_{k}": v for k, v in self.proposal_kwargs.items()},
                             **{f"wf_{k}": v for k, v in self.wf_kwargs.items()},
                             spins=spins, **({"spin_model": self._spin_label, "a_max": a_max,
                                              **({"f_ref": f_ref} if f_ref else {"x_ref": x_ref})}
                                             if spins else {}))
        self._log(f"setup: kernel + proposal in {time.time() - t0:.1f} s, "
                  f"R_tot = {self.kernel.N_tot * 3.15576e7:.4g} events/yr "
                  f"in z = [{z_min}, {z_max}]")

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def prefactor(self):
        """4 pi^2 f^3 / (3 H0^2), times S_REF to undo the internal units."""
        return 4.0 * np.pi ** 2 * self.freqs ** 3 / (3.0 * H0_SI ** 2) * S_REF

    def run(self, n_rep=8, log2_points=12, seed=20261126, rep_start=0,
            antithetic=False, chunk=512):
        """
        Run replicas rep_start .. rep_start + n_rep - 1, each with 2^log2_points
        scrambled-Sobol points (x2 waveform calls with antithetic=True).
        Replica r uses the seed (seed, r), so results for different replica
        ranges can be merged with OmegaResult.merge.
        """
        _init_worker(self.wf_name, self.wf_kwargs, self.freqs)
        pool = (Pool(self.n_proc, initializer=_init_worker,
                     initargs=(self.wf_name, self.wf_kwargs, self.freqs))
                if self.n_proc > 1 else None)
        pref = self.prefactor()
        reps = list(range(rep_start, rep_start + n_rep))
        vals, varm, esss, n_fail, n_wf = [], [], [], 0, 0
        try:
            for r in reps:
                t0 = time.time()
                m, v, e, nf, nw = self._replica(r, log2_points, seed, antithetic, chunk, pool)
                vals.append(m * pref)
                varm.append(v * pref ** 2)
                esss.append(e)
                n_fail += nf
                n_wf   += nw
                self._log(f"  replica {r}: {nw} waveforms, {nf} failed, "
                          f"{time.time() - t0:.1f} s")
        finally:
            if pool:
                pool.close()
                pool.join()

        if n_fail:
            print(f"WARNING: {n_fail}/{n_wf} waveform calls failed and were set "
                  f"to zero -> biased result")
        meta = dict(mass_model=type(self.mass).__name__, rate_model=type(self.rate).__name__,
                    mass_params=_params(self.mass), rate_params=_params(self.rate),
                    waveform=self.wf_name, seed=seed, rep_ids=np.array(reps),
                    n_points=2 ** log2_points, antithetic=antithetic,
                    n_failed=n_fail, n_waveforms=n_wf, settings=repr(self.settings))
        res = OmegaResult.from_replicas(self.freqs, np.array(vals), np.array(varm),
                                        np.array(esss), meta)
        self._log(res.summary())
        return res

    def _replica(self, rep, log2_points, seed, antithetic, chunk, pool):
        dim = 4 + {None: 0, "aligned": 5, "precessing": 6}[self.spins]
        sobol = qmc.Sobol(d=dim, scramble=True, seed=np.random.default_rng([seed, rep]))
        x = sobol.random_base2(log2_points)
        Mc_det, q, u, phi, g = self.proposal.sample(x[:, :4])
        N, Nf = len(x), len(self.freqs)
        if self.spins:
            xs = np.clip(x[:, 4:], 1e-9, 1 - 1e-9)
            a1, a2 = self.a_max * xs[:, 0], self.a_max * xs[:, 1]
            c1, c2 = 2 * xs[:, 2] - 1, 2 * xs[:, 3] - 1
            p12    = 2 * np.pi * xs[:, 4]
            # phi1: azimuth of spin 1's in-plane part relative to the orbital separation
            p1     = 2 * np.pi * xs[:, 5] if self.spins == "precessing" else np.zeros(N)
            # proposal density of (a1, a2, c1, c2) is 1/(4 a_max^2); phi1 and phi12 cancel
            w_spin = self._spin_density(a1, a2, c1, c2) * 4 * self.a_max ** 2
            if self.spins == "aligned":
                r1 = r2 = np.zeros(N)
            else:
                r1, r2 = np.sqrt(1 - c1 ** 2), np.sqrt(1 - c2 ** 2)
            S1 = np.stack([a1 * r1 * np.cos(p1), a1 * r1 * np.sin(p1), a1 * c1], axis=1)
            S2 = np.stack([a2 * r2 * np.cos(p1 + p12), a2 * r2 * np.sin(p1 + p12), a2 * c2], axis=1)
        else:
            w_spin = np.ones(N)
            S1 = S2 = np.zeros((N, 3))
        s1, s2 = np.zeros(Nf), np.zeros(Nf)
        n_wf = n_fail = 0
        signs = (1.0, -1.0) if antithetic else (1.0,)

        for s in range(0, N, chunk):
            sl = slice(s, min(s + chunk, N))
            # measure: dMc_det dq du/2 dphi/2pi = Mc_det q dlnMc dlnq du/2 dphi/2pi
            W = (Mc_det[sl] * q[sl] * self.kernel(Mc_det[sl], q[sl]) / (2.0 * g[sl])
                 * w_spin[sl])
            alive = np.nonzero(W > 0)[0]
            if len(alive) == 0:
                continue
            mc_a, q_a = Mc_det[sl][alive], q[sl][alive]
            m1 = mc_a * (1 + q_a) ** 0.2 / q_a ** 0.6
            m2 = q_a * m1
            I = np.zeros((len(alive), Nf))
            if not self.spins:
                fref = [None] * len(alive)
            elif self.f_ref:
                fref = [float(self.f_ref)] * len(alive)
            else:
                fref = self.x_ref / ((m1 + m2) * 4.925490947641267e-06)
            s1a, s2a = [tuple(v) for v in S1[sl][alive]], [tuple(v) for v in S2[sl][alive]]
            for sgn in signs:
                jobs = list(zip(m1, m2, np.arccos(sgn * u[sl][alive]), phi[sl][alive],
                                s1a, s2a, fref))
                res  = (pool.map(_intensity_one, jobs, chunksize=8) if pool
                        else [_intensity_one(j) for j in jobs])
                I     += np.array([r[0] for r in res]) / len(signs)
                n_wf  += len(res)
                n_fail += sum(not r[1] for r in res)
            v = W[alive, None] * I
            s1 += v.sum(axis=0)
            s2 += (v ** 2).sum(axis=0)

        mean = s1 / N
        var_of_mean = (s2 / N - mean ** 2) / (N - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ess = s1 ** 2 / (N * s2)
        return mean, var_of_mean, ess, n_fail, n_wf


def _params(model):
    """repr of a model's hyperparameters (public attributes plus a `_p` dict)."""
    p = {k: v for k, v in vars(model).items() if not k.startswith("_")}
    p.update(getattr(model, "_p", {}))
    return repr(p)


def compute_omega(mass_model="bpl_double_gaussian", rate_model="madau_dickinson",
                  waveform="IMRPhenomXAS", n_rep=8, log2_points=12, seed=20261126,
                  **kwargs):
    """One-call convenience wrapper around OmegaIntegrator(...).run(...)."""
    oi = OmegaIntegrator(mass_model, rate_model, waveform, **kwargs)
    return oi.run(n_rep=n_rep, log2_points=log2_points, seed=seed)
