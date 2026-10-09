# orientation.py
# Orientation-averaged intensity of one source, computed exactly.
#
#   <I>(f) = < |h+(f)|^2 + |hx(f)|^2 >  over observer directions (iota, phi_ref)
#
# Two exact methods:
#
# 1) Modes (IMRPhenomXPHM).  With h = h+ - i hx = sum_lm h_lm Y^{-2}_lm(n)
#    and the orthonormality of the spin-weighted harmonics,
#        <I>(f) = 1/(8 pi) sum_lm ( |h_lm(f)|^2 + |h_lm(-f)|^2 ),
#    using the inertial-frame modes.  Precession angles drop out of the
#    average; one call per mode, no sampling of the orientation.
#    SEOBNRv5HM_ROM returns its seven m < 0 modes in one call; for aligned
#    spins h_{l,m}(-f) = (-1)^l h*_{l,-m}(f), so <I> = 1/(4 pi) sum_{m<0} |h_lm|^2.
#
# 2) Product quadrature for the polarisation interface.  For a model whose
#    inertial-frame modes have l <= L, I(iota, phi) contains e^{i(m-m')phi}
#    with |m-m'| <= 2L and, after the phi average, polynomials of degree
#    <= 2L in cos(iota).  So n_phi = 2L+1 uniform phases and a Gauss-Legendre
#    rule exact to degree 2L in cos(iota) average exactly.  Non-precessing
#    22-only models need no phi nodes.
#
# (iota, phi_ref) parametrise the observer direction in the source frame,
# so the average over them is the average over directions.
#
# Spins are passed in the L-frame at the reference frequency f_ref; the
# FastOmega tables use f_ref = x_ref / (M T_sun), i.e. spins defined at a
# fixed dimensionless frequency, which keeps the scale invariance exact.

import os
import sys

import numpy as np


def ensure_lal_data_path():
    """
    Point LAL_DATA_PATH at <environment>/share/lal_data when it is unset and
    that directory exists (ROM data files such as SEOBNRv5ROM_v1.0.hdf5).
    Jupyter kernels started by an editor do not always load conda's
    environment variables, so this is done here rather than relied upon.
    """
    if not os.environ.get("LAL_DATA_PATH"):
        d = os.path.join(sys.prefix, "share", "lal_data")
        if os.path.isdir(d):
            os.environ["LAL_DATA_PATH"] = d


# (n_u, n_phi) product rules, exact for the modes each model contains.
# "face-on": a non-precessing 22-only signal has I(iota) ∝ A(cos iota),
# A(u) = ((1+u^2)/2)^2 + u^2, <A> = 4/5, A(1) = 2, so <I> = 0.4 I(face-on):
# one call, exact.
PRODUCT_RULES = {
    "IMRPhenomXAS":           "face-on",  # 22 only, aligned spins
    "IMRPhenomD":             "face-on",
    "IMRPhenomXAS_NRTidalv2": "face-on",
    "IMRPhenomXP":            (3, 5),     # 22 only, precessing: l = 2
    "IMRPhenomXP_NRTidalv2":  (3, 5),
    "IMRPhenomXHM":           (5, 9),     # l <= 4, aligned spins
    "SEOBNRv5_ROM":           "face-on",  # 22 only, aligned spins
}
DEFAULT_RULE = (5, 9)                     # exact for any model with l <= 4

MODE_MODELS = {"IMRPhenomXPHM": [(l, m) for l in (2, 3, 4) for m in range(-l, l + 1)],
               "SEOBNRv5HM_ROM": 7}     # (2,2) (3,3) (2,1) (4,4) (5,5) (3,2) (4,3), one call

TIDAL_MODELS = {"IMRPhenomXAS_NRTidalv2", "IMRPhenomXP_NRTidalv2"}

# Reduced-order models only cover M f >= x_min.  Below x_join = 1.1 x_min the
# intensity is continued with a Phenom inspiral of the same mode content,
# rescaled to the ROM over [x_join, 2 x_join] and blended smoothly (the
# models differ by ~1-2% there, spin dependent).  The SEOBNRv5 ROMs fail
# (GSL interpolation error) near their low-frequency edge, and not even
# deterministically: HM below x ~ 2e-4 for q < 0.03, 22 at the 1e-3 level
# of calls below 1e-4.  x_min = 3e-4 was clean in all tests.
ROM_MODELS = {
    "SEOBNRv5_ROM":   dict(x_min=3e-4, extend_with="IMRPhenomXAS"),
    "SEOBNRv5HM_ROM": dict(x_min=3e-4, extend_with="IMRPhenomXHM"),
}
PRECESSING_MODELS = {"IMRPhenomXP", "IMRPhenomXPHM", "IMRPhenomXP_NRTidalv2",
                     "IMRPhenomXO4a", "IMRPhenomPv2", "IMRPhenomPv3HM"}


class OrientationAverager:
    """
    <I>(f; m1, m2, spin1, spin2, Lambda1, Lambda2) at 1 Mpc for one approximant.

    rule: None picks the modes method for IMRPhenomXPHM and the product
    rule of PRODUCT_RULES (default (5, 9)) otherwise; or pass "modes" or an
    (n_u, n_phi) tuple.
    """

    def __init__(self, approximant, rule=None):
        ensure_lal_data_path()
        import lal
        import lalsimulation as lalsim
        lal.ClobberDebugLevel(0)
        self._lal, self._lalsim = lal, lalsim
        self.name = approximant
        self.approximant = lalsim.GetApproximantFromString(approximant)
        if rule is None:
            rule = "modes" if approximant in MODE_MODELS else PRODUCT_RULES.get(approximant, DEFAULT_RULE)
        if rule == "modes" and approximant not in MODE_MODELS:
            raise ValueError(f"no mode interface wired for {approximant}")
        self.rule = rule
        if rule == "face-on":
            self._nodes = [(0.0, 0.0, 0.4)]
        elif rule != "modes":
            u, w = np.polynomial.legendre.leggauss(rule[0])
            self._nodes = [(np.arccos(ui), 2 * np.pi * j / rule[1], 0.5 * wi / rule[1])
                           for ui, wi in zip(u, w) for j in range(rule[1])]
        self.tidal = approximant in TIDAL_MODELS
        self.precessing = approximant in PRECESSING_MODELS

    def _params(self, lambdas):
        if lambdas is None:
            return None
        p = self._lal.CreateDict()
        self._lalsim.SimInspiralWaveformParamsInsertTidalLambda1(p, float(lambdas[0]))
        self._lalsim.SimInspiralWaveformParamsInsertTidalLambda2(p, float(lambdas[1]))
        return p

    def __call__(self, freqs, m1, m2, spin1=(0, 0, 0), spin2=(0, 0, 0), f_ref=20.0,
                 lambdas=None):
        freqs = np.asarray(freqs, dtype=float)
        s1, s2 = tuple(map(float, spin1)), tuple(map(float, spin2))
        if not self.precessing:
            s1, s2 = (0.0, 0.0, s1[2]), (0.0, 0.0, s2[2])
        params = self._params(lambdas)
        if self.name in ROM_MODELS:
            return self._rom_extended(freqs, m1, m2, s1, s2, f_ref, params)
        return self._average(freqs, m1, m2, s1, s2, f_ref, params)

    def _average(self, freqs, m1, m2, s1, s2, f_ref, params):
        lal, lalsim = self._lal, self._lalsim
        dist = 1e6 * lal.PC_SI

        if self.rule == "modes":
            fv = lal.CreateREAL8Vector(len(freqs))
            fv.data[:] = freqs
            if self.name == "SEOBNRv5HM_ROM":
                h = lalsim.SimIMRSEOBNRv5HMROMFrequencySequence_Modes(
                    fv, 0.0, f_ref, dist, 0.0, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
                    s1[2], s2[2], -1, MODE_MODELS[self.name], params, lalsim.NoNRT_V)
                tot = np.zeros(len(freqs))
                while h is not None:
                    tot += np.abs(h.mode.data.data) ** 2
                    h = h.next
                return tot / (4 * np.pi)
            tot = np.zeros(len(freqs))
            for l, m in MODE_MODELS[self.name]:
                hp, hn = lalsim.SimIMRPhenomXPHMFrequencySequenceOneMode(
                    fv, l, m, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, *s1, *s2,
                    dist, 0.0, 0.0, f_ref, params)
                tot += np.abs(hp.data.data) ** 2 + np.abs(hn.data.data) ** 2
            return tot / (8 * np.pi)

        seq = lal.CreateREAL8Sequence(len(freqs))
        seq.data[:] = freqs
        tot = np.zeros(len(freqs))
        for iota, phi, w in self._nodes:
            hp, hc = lalsim.SimInspiralChooseFDWaveformSequence(
                phi, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, *s1, *s2, f_ref, dist, iota,
                params, self.approximant, seq)
            tot += w * (np.abs(hp.data.data) ** 2 + np.abs(hc.data.data) ** 2)
        return tot

    def _rom_extended(self, freqs, m1, m2, s1, s2, f_ref, params):
        """ROM for M f >= x_join, Phenom inspiral (rescaled) below, smooth blend in between."""
        info = ROM_MODELS[self.name]
        if info["x_min"] is None:
            return self._average(freqs, m1, m2, s1, s2, f_ref, params)
        if not hasattr(self, "_ref"):
            self._ref = OrientationAverager(info["extend_with"])
        fj  = 1.1 * info["x_min"] / ((m1 + m2) * self._lal.MTSUN_SI)    # join frequency
        out = np.zeros(len(freqs))
        hi  = freqs >= fj
        if hi.any():
            out[hi] = self._average(freqs[hi], m1, m2, s1, s2, f_ref, params)
        lo = freqs < 2 * fj
        if lo.any():
            fo = np.geomspace(fj, 2 * fj, 12)                           # overlap
            r  = (self._average(fo, m1, m2, s1, s2, f_ref, params)
                  / self._ref._average(fo, m1, m2, s1, s2, f_ref, params))
            scale = np.exp(np.mean(np.log(r)))
            I_ref = self._ref._average(freqs[lo], m1, m2, s1, s2, f_ref, params) * scale
            t = np.clip(np.log(freqs[lo] / fj) / np.log(2.0), 0.0, 1.0)
            w = t * t * (3 - 2 * t)                                     # 0 at fj, 1 at 2 fj
            out[lo] = (1 - w) * I_ref + w * out[lo]
        return out
