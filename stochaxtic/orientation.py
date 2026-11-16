# Orientation-averaged intensity of one source, computed exactly:
#
#   <I>(f) = < |h+(f)|^2 + |hx(f)|^2 >  over observer directions (iota, phi_ref)
#
# (iota, phi_ref) is the observer direction in the source frame.  Spins are given in
# the L-frame at f_ref; FastOmega uses f_ref = x_ref / (M T_sun), which keeps the scale
# invariance exact.
#
# 1) Modes, rule "modes" (IMRPhenomXPHM, SEOBNRv5HM_ROM).  With h = h+ - i hx =
#    sum_lm h_lm Y^{-2}_lm(n) and the orthonormality of the spin-weighted harmonics,
#        <I>(f) = 1/(8 pi) sum_lm ( |h_lm(f)|^2 + |h_lm(-f)|^2 )
#    for the inertial-frame modes; the precession angles drop out.  SEOBNRv5HM_ROM
#    returns its seven m < 0 modes in one call; for aligned spins
#    h_{l,m}(-f) = (-1)^l h*_{l,-m}(f), so <I> = 1/(4 pi) sum_{m<0} |h_lm|^2.
#
# 2) Product rule (n_u, n_phi) for the polarisation interface.  If the inertial-frame
#    modes have l <= L and do not depend on the observer, I(iota, phi) contains
#    e^{i(m-m')phi} with |m-m'| <= 2L and, after the phi average, polynomials of degree
#    <= 2L in u = cos(iota): n_phi = 2L+1 phases and Gauss-Legendre exact to degree 2L
#    in u are exact.  "face-on": non-precessing 22-only models, one call.
#
# 3) Rule "trig" (polarisation-level IMRPhenomPv3HM at fixed spins).  Pv3HM uses phi_ref
#    both as the observer azimuth and as the orbital phase of its co-precessing modes,
#    so cross terms with odd m - m' leave odd powers of sin(iota) after the phi average:
#        <I>_phi(u) = P(u) + sqrt(1 - u^2) Q(u),  deg P <= 8, deg Q <= 7,
#    which Gauss-Legendre in u does not integrate exactly ((5, 9) is off by up to 5e-3).
#    TRIG_RULE: 10 nodes in u, exact for u^j and sqrt(1 - u^2) u^j (j <= 9), x 9 phases.
#    It is also exact for every model of 2) with l <= 4, at twice the cost of (5, 9).
#
# 4) Co-precessing modes, rule "coprec" (default for IMRPhenomPv3, IMRPhenomPv3HM).  Both
#    rotate co-precessing PhenomHM modes into the inertial frame with the MSA angles.
#    The rotation is unitary for each l, so the angles drop out of the sky-integrated
#    power (CC_moments_related/notes/coprecessing_frame_note.tex):
#        <I>(f) = amp0^2 / (4 pi) sum_{l, m > 0} |h'_lm(f)|^2,   amp0 = M^2 MRSUN MTSUN / D,
#    with h'_lm from SimIMRPhenomHMGethlmModes: (2,2) for Pv3; (2,2) (2,1) (3,3) (3,2)
#    (4,4) (4,3) for Pv3HM.  One call, no MSA angles, so none of their refusals (rules 2
#    and 3 fail for ~0.07% of isotropic spins).  Exact for Pv3 at any spins.  Pv3HM
#    rotates its modes at different times, so cross terms between modes survive at fixed
#    spins; they carry e^{-i(m - m') phi1} (phi1: azimuth of the in-plane spins) and
#    cancel in the average over phi1, which is uniform for an isotropic population, as
#    FastOmega and OmegaIntegrator assume.  At fixed spins the rule-3 value is not
#    physical on its own (phi_ref is also the co-precessing phase) and differs from it
#    by up to tens of percent.

import os
import sys

import numpy as np


def ensure_lal_data_path():
    """
    Point LAL_DATA_PATH at <environment>/share/lal_data (ROM data files such as
    SEOBNRv5ROM_v1.0.hdf5) when it is unset and that directory exists: Jupyter
    kernels started by an editor do not always load conda's environment variables.
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
    "IMRPhenomXAS_NRTidalv3": "face-on",
    "IMRPhenomNSBH":          "face-on",  # 22 only, aligned BH spin
    "IMRPhenomXP":            (3, 5),     # 22 only, precessing: l = 2
    "IMRPhenomXP_NRTidalv2":  (3, 5),
    "IMRPhenomXP_NRTidalv3":  (3, 5),
    "IMRPhenomXHM":           (5, 9),     # l <= 4, aligned spins
    "SEOBNRv5_ROM":           "face-on",  # 22 only, aligned spins
    "SEOBNRv5_ROM_NRTidalv3": "face-on",  # 22 only, aligned spins, NRTidalv3 tides
}
# Polarisation-level rules for the COPREC_MODELS, used only when passed explicitly
# (they call the MSA angles): (3, 5) is exact for Pv3, "trig" for Pv3HM at fixed spins.
DEFAULT_RULE = (5, 9)                     # exact for any model with l <= 4

# Rule "trig": nodes u = cos(iota) and weights (sum 1, measure du/2 on [-1, 1]), exact
# for u^j and sqrt(1 - u^2) u^j, j <= 9.  Symmetric pairs +-u, positive weights; solved
# with mpmath Newton from Gauss-Legendre in iota (moment residuals 1e-39).
TRIG_RULE_U = np.array([0.99458284765129843, 0.94677375570525160, 0.80552327704128391,
                        0.54757363817943817, 0.19463059531411349])
TRIG_RULE_W = np.array([0.0088255848371810768, 0.043631500160449105, 0.099693016097141072,
                        0.15636798992909751, 0.19148190897613123])
TRIG_RULE_U = np.concatenate((TRIG_RULE_U, -TRIG_RULE_U[::-1]))
TRIG_RULE_W = np.concatenate((TRIG_RULE_W, TRIG_RULE_W[::-1]))
TRIG_RULE_NPHI = 9

# Approximants that SimInspiralChooseFDWaveformSequence does not offer.  Their own LAL
# functions take a frequency vector and reproduce SimInspiralChooseFDWaveform bit for
# bit.  Both refuse (XLAL_EDOM, "Omegaz5 ... larger than expected") ~0.07% of isotropic
# spins, mostly with a strongly anti-aligned primary, where their MSA angles are not
# defined; aligned spins never fail.
DIRECT_MODELS = {"IMRPhenomPv3":   "SimIMRPhenomPv3",
                 "IMRPhenomPv3HM": "SimIMRPhenomPv3HMGetHplusHcross"}

# Rule "coprec": the co-precessing modes of each model, m > 0
COPREC_MODELS = {"IMRPhenomPv3":   [(2, 2)],
                 "IMRPhenomPv3HM": [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]}

MODE_MODELS = {"IMRPhenomXPHM": [(l, m) for l in (2, 3, 4) for m in range(-l, l + 1)],
               "SEOBNRv5HM_ROM": 7}     # (2,2) (3,3) (2,1) (4,4) (5,5) (3,2) (4,3), one call

TIDAL_MODELS = {"IMRPhenomXAS_NRTidalv2", "IMRPhenomXP_NRTidalv2",
                "IMRPhenomXAS_NRTidalv3", "IMRPhenomXP_NRTidalv3", "SEOBNRv5_ROM_NRTidalv3",
                "IMRPhenomNSBH"}
# NSBH models: only the secondary (the neutron star) has a tidal deformability;
# IMRPhenomNSBH ignores Lambda1 (the aligned spins of both bodies enter), needs m1 >= m2,
# m2 <= 3 Msun, m1/m2 <= 100 and Lambda2 <= 5000
NSBH_MODELS = {"IMRPhenomNSBH": dict(lambda_max=5000.0, m2_max=3.0, q_min=0.01)}

# Reduced-order models cover only M f >= x_min.  Below x_join = 1.1 x_min the intensity
# is continued with a Phenom model of the same mode content (and tides), rescaled to the
# ROM over [x_join, 2 x_join] and blended smoothly (they differ by ~1-2% there).  The
# SEOBNRv5 ROMs fail intermittently (GSL interpolation error) near their low-frequency
# edge (HM below x ~ 2e-4 for q < 0.03, 22 rarely below 1e-4, NRTidalv3 for some long
# sequences from 1e-5); x_min = 3e-4 was clean in all tests.
ROM_MODELS = {
    "SEOBNRv5_ROM":           dict(x_min=3e-4, extend_with="IMRPhenomXAS"),
    "SEOBNRv5HM_ROM":         dict(x_min=3e-4, extend_with="IMRPhenomXHM"),
    "SEOBNRv5_ROM_NRTidalv3": dict(x_min=3e-4, extend_with="IMRPhenomXAS_NRTidalv3"),
}
PRECESSING_MODELS = {"IMRPhenomXP", "IMRPhenomXPHM", "IMRPhenomXP_NRTidalv2", "IMRPhenomXP_NRTidalv3",
                     "IMRPhenomPv2", "IMRPhenomPv3", "IMRPhenomPv3HM"}
# Refused on purpose: an unregistered approximant would be treated as aligned-spin
# (in-plane spins dropped) with the default rule, without a warning.
UNSUPPORTED_MODELS = {
    "IMRPhenomXO4a": "its antisymmetric (2,2) contribution is not a pointwise function of frequency "
                     "in LALSuite (the value at f depends on the other requested frequencies) and its "
                     "polarizations disagree with its own modes, so <I> is not well defined",
}


def direct_fd_waveform(approximant, freqs, m1_si, m2_si, s1, s2, dist, iota, phi_ref, f_ref,
                       lal_params=None):
    """(h+, hx) of a DIRECT_MODELS approximant at `freqs` (any order, repeats allowed)."""
    import lal
    import lalsimulation as lalsim
    fu, inv = np.unique(np.asarray(freqs, dtype=float), return_inverse=True)   # LAL: increasing
    n = len(fu)
    if n == 2:                     # LAL reads a 2-vector as the bounds of a uniform grid
        fu = np.append(fu, 1.01 * fu[-1])
    v = lal.CreateREAL8Vector(len(fu))
    v.data[:] = fu
    try:
        hp, hc = getattr(lalsim, DIRECT_MODELS[approximant])(
            v, m1_si, m2_si, *s1, *s2, dist, iota, phi_ref, 0.0, f_ref, lal_params)
    except RuntimeError as e:
        raise RuntimeError(f"{approximant} failed for m1 = {m1_si / lal.MSUN_SI:.6g}, m2 = "
                           f"{m2_si / lal.MSUN_SI:.6g} Msun, spin1 = {tuple(s1)}, spin2 = {tuple(s2)}, "
                           f"f_ref = {f_ref:.6g} Hz: {e}") from e
    return hp.data.data[:n][inv], hc.data.data[:n][inv]


class OrientationAverager:
    """
    <I>(f; m1, m2, spin1, spin2, Lambda1, Lambda2) at 1 Mpc for one approximant.

    rule: None picks "modes" for IMRPhenomXPHM and SEOBNRv5HM_ROM, "coprec" for
    IMRPhenomPv3/Pv3HM and PRODUCT_RULES (default (5, 9)) otherwise; or pass "modes",
    "coprec", "trig" (10 nodes in iota x 9 phases) or an (n_u, n_phi) tuple
    (Gauss-Legendre in cos(iota) x n_phi phases).
    """
    def __init__(self, approximant, rule=None):
        ensure_lal_data_path()
        import lal
        import lalsimulation as lalsim
        lal.ClobberDebugLevel(0)
        self._lal, self._lalsim = lal, lalsim
        if approximant in UNSUPPORTED_MODELS:
            raise ValueError(f"{approximant} is not supported: {UNSUPPORTED_MODELS[approximant]}")
        self.name = approximant
        self.approximant = lalsim.GetApproximantFromString(approximant)
        if rule is None:
            rule = ("modes" if approximant in MODE_MODELS else
                    "coprec" if approximant in COPREC_MODELS else
                    PRODUCT_RULES.get(approximant, DEFAULT_RULE))
        if rule == "modes" and approximant not in MODE_MODELS:
            raise ValueError(f"no mode interface wired for {approximant}")
        if rule == "coprec" and approximant not in COPREC_MODELS:
            raise ValueError(f"rule 'coprec' is only wired for {sorted(COPREC_MODELS)}, not {approximant}")
        self.rule = rule
        if rule == "face-on":
            self._nodes = [(0.0, 0.0, 0.4)]
        elif rule == "trig":
            n = TRIG_RULE_NPHI
            self._nodes = [(np.arccos(ui), 2 * np.pi * j / n, wi / n)
                           for ui, wi in zip(TRIG_RULE_U, TRIG_RULE_W) for j in range(n)]
        elif rule not in ("modes", "coprec"):
            u, w = np.polynomial.legendre.leggauss(rule[0])
            self._nodes = [(np.arccos(ui), 2 * np.pi * j / rule[1], 0.5 * wi / rule[1])
                           for ui, wi in zip(u, w) for j in range(rule[1])]
        self.tidal = approximant in TIDAL_MODELS
        self.precessing = approximant in PRECESSING_MODELS

    def _params(self, lambdas):
        # a fresh dict for every LAL call: IMRPhenomNSBH writes testGR parameters
        # into the dict it is given, and LAL then rejects that dict on the next call
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
        if self.name in ROM_MODELS:
            return self._rom_extended(freqs, m1, m2, s1, s2, f_ref, lambdas)
        return self._average(freqs, m1, m2, s1, s2, f_ref, lambdas)

    def _average(self, freqs, m1, m2, s1, s2, f_ref, lambdas):
        lal, lalsim = self._lal, self._lalsim
        dist = 1e6 * lal.PC_SI

        if self.rule == "coprec":
            return self._coprec(freqs, m1, m2, s1, s2, f_ref, dist)

        if self.rule == "modes":
            fv = lal.CreateREAL8Vector(len(freqs))
            fv.data[:] = freqs
            if self.name == "SEOBNRv5HM_ROM":
                h = lalsim.SimIMRSEOBNRv5HMROMFrequencySequence_Modes(
                    fv, 0.0, f_ref, dist, 0.0, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
                    s1[2], s2[2], -1, MODE_MODELS[self.name], self._params(lambdas), lalsim.NoNRT_V)
                tot = np.zeros(len(freqs))
                while h is not None:
                    tot += np.abs(h.mode.data.data) ** 2
                    h = h.next
                return tot / (4 * np.pi)
            tot = np.zeros(len(freqs))
            for l, m in MODE_MODELS[self.name]:
                hp, hn = lalsim.SimIMRPhenomXPHMFrequencySequenceOneMode(
                    fv, l, m, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, *s1, *s2,
                    dist, 0.0, 0.0, f_ref, self._params(lambdas))
                tot += np.abs(hp.data.data) ** 2 + np.abs(hn.data.data) ** 2
            return tot / (8 * np.pi)

        direct = self.name in DIRECT_MODELS
        if not direct:
            seq = lal.CreateREAL8Sequence(len(freqs))
            seq.data[:] = freqs
        tot = np.zeros(len(freqs))
        for iota, phi, w in self._nodes:
            if direct:
                hp, hc = direct_fd_waveform(self.name, freqs, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
                                            s1, s2, dist, iota, phi, f_ref, self._params(lambdas))
            else:
                hp, hc = lalsim.SimInspiralChooseFDWaveformSequence(
                    phi, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, *s1, *s2, f_ref, dist, iota,
                    self._params(lambdas), self.approximant, seq)
                hp, hc = hp.data.data, hc.data.data
            tot += w * (np.abs(hp) ** 2 + np.abs(hc) ** 2)
        return tot

    def _coprec(self, freqs, m1, m2, s1, s2, f_ref, dist):
        """Rule "coprec": amp0^2 / (4 pi) sum_{l, m>0} |h'_lm|^2 of the co-precessing modes."""
        lal, lalsim = self._lal, self._lalsim
        fu, inv = np.unique(freqs, return_inverse=True)      # LAL: increasing, no repeats
        v = lal.CreateREAL8Vector(len(fu))
        v.data[:] = fu
        p = lal.CreateDict()
        modes = lalsim.SimInspiralCreateModeArray()
        for l, m in COPREC_MODELS[self.name]:
            lalsim.SimInspiralModeArrayActivateMode(modes, l, m)
        lalsim.SimInspiralWaveformParamsInsertModeArray(p, modes)   # inserts a copy
        try:
            h = lalsim.SimIMRPhenomHMGethlmModes(v, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, *s1, *s2,
                                                 0.0, 0.0, f_ref, p)
        except RuntimeError as e:
            raise RuntimeError(f"{self.name} co-precessing modes failed for m1 = {m1:.6g}, m2 = {m2:.6g} "
                               f"Msun, spin1 = {s1}, spin2 = {s2}, f_ref = {f_ref:.6g} Hz: {e}") from e
        tot = np.zeros(len(fu))
        while h is not None:
            tot += np.abs(h.mode.data.data[:len(fu)]) ** 2
            h = h.next
        M = m1 + m2
        amp0 = M * lal.MRSUN_SI * M * lal.MTSUN_SI / dist
        return amp0 ** 2 / (4 * np.pi) * tot[inv]

    def _rom_extended(self, freqs, m1, m2, s1, s2, f_ref, lambdas):
        """ROM for M f >= x_join, Phenom inspiral (rescaled) below, smooth blend in between."""
        info = ROM_MODELS[self.name]
        if info["x_min"] is None:
            return self._average(freqs, m1, m2, s1, s2, f_ref, lambdas)
        if not hasattr(self, "_ref"):
            self._ref = OrientationAverager(info["extend_with"])
        fj  = 1.1 * info["x_min"] / ((m1 + m2) * self._lal.MTSUN_SI)    # join frequency
        out = np.zeros(len(freqs))
        hi  = freqs >= fj
        if hi.any():
            out[hi] = self._average(freqs[hi], m1, m2, s1, s2, f_ref, lambdas)
        lo = freqs < 2 * fj
        if lo.any():
            fo = np.geomspace(fj, 2 * fj, 12)                           # overlap
            r  = (self._average(fo, m1, m2, s1, s2, f_ref, lambdas)
                  / self._ref._average(fo, m1, m2, s1, s2, f_ref, lambdas))
            scale = np.exp(np.mean(np.log(r)))
            I_ref = self._ref._average(freqs[lo], m1, m2, s1, s2, f_ref, lambdas) * scale
            t = np.clip(np.log(freqs[lo] / fj) / np.log(2.0), 0.0, 1.0)
            w = t * t * (3 - 2 * t)                                     # 0 at fj, 1 at 2 fj
            out[lo] = (1 - w) * I_ref + w * out[lo]
        return out
