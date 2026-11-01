# waveforms.py
# Waveform models.  Only the intensity I(f) = |h+|^2 + |hx|^2 of one source
# at D_REF_MPC = 1 Mpc is needed for Omega_GW.
#
# Public interface:
#   get_waveform_model(name, **kw)  ->  a WaveformModelBase instance
#
# Any name that is not in _REGISTRY is treated as a LAL approximant string,
# e.g. get_waveform_model("IMRPhenomXPHM").  To add a non-LAL model
# (gwsurrogate, ripple, ...), subclass WaveformModelBase, implement hp_hc()
# and register it in _REGISTRY.

import numpy as np

from .constants import D_REF_MPC


class WaveformModelBase:
    name = "base"

    def hp_hc(self, freqs, m1_det, m2_det, iota, phi_ref, spin1, spin2, f_ref=None):
        """
        Complex frequency-domain polarisations at D_REF_MPC, each (Nf,).
        m1_det, m2_det are detector-frame masses in Msun; spins are defined
        at f_ref (None: the model's default rule).
        """
        raise NotImplementedError

    def intensity(self, freqs, m1_det, m2_det, iota, phi_ref=0.0,
                  spin1=(0.0, 0.0, 0.0), spin2=(0.0, 0.0, 0.0), f_ref=None):
        """I(f) = |h+|^2 + |hx|^2 at D_REF_MPC, shape (Nf,)."""
        hp, hc = self.hp_hc(freqs, m1_det, m2_det, iota, phi_ref, spin1, spin2, f_ref)
        return hp.real ** 2 + hp.imag ** 2 + hc.real ** 2 + hc.imag ** 2


class LALWaveform(WaveformModelBase):
    """
    Any lalsimulation approximant.

    mode = "sequence": SimInspiralChooseFDWaveformSequence evaluated exactly
                       on `freqs`.  Fastest; IMRPhenom family and SEOBNR ROMs.
    mode = "fd":       SimInspiralFD on a uniform grid, then picked at
                       `freqs`.  Also handles time-domain models (NRSur7dq4,
                       SEOBNRv4PHM, ...); LAL's conditioning tapers the
                       lowest few Hz above f_min.  `freqs` must be a
                       subset of a 2^-k Hz grid.

    isco_cut: if set, zero the waveform at f > isco_cut * 4400/M_det Hz
    (isco_cut=3 reproduces the MOGWAS per-frequency code; it removes most of
    the merger-ringdown).  Default None keeps the full waveform.

    f_ref is lowered to half the rough ISCO frequency for very heavy systems
    so that it stays below merger.  With zero spins this only shifts the
    phase and leaves I(f) unchanged.
    """

    def __init__(self, approximant, mode="sequence", f_ref=20.0,
                 delta_f=0.0, lal_params=None, isco_cut=None):
        from .orientation import ensure_lal_data_path
        ensure_lal_data_path()                     # ROM data (SEOBNRv5*_ROM)
        import lal
        import lalsimulation as lalsim
        lal.ClobberDebugLevel(0)            # silence cosmetic LAL warnings
        if mode not in ("sequence", "fd"):
            raise ValueError(f"unknown mode '{mode}' (use 'sequence' or 'fd')")
        self._lal, self._lalsim = lal, lalsim
        self.name        = approximant
        self.approximant = lalsim.GetApproximantFromString(approximant)
        self.mode        = mode
        self.f_ref       = f_ref
        self.delta_f     = delta_f
        self.lal_params  = lal_params
        self.isco_cut    = isco_cut
        self._seq_cache  = (None, None)

    def _freq_sequence(self, freqs):
        key = freqs.tobytes()
        if self._seq_cache[0] != key:
            seq = self._lal.CreateREAL8Sequence(len(freqs))
            seq.data[:] = freqs
            self._seq_cache = (key, seq)
        return self._seq_cache[1]

    def hp_hc(self, freqs, m1_det, m2_det, iota, phi_ref, spin1, spin2, f_ref=None):
        lal, lalsim = self._lal, self._lalsim
        freqs  = np.asarray(freqs, dtype=float)
        m_tot  = m1_det + m2_det
        f_isco = 4400.0 / m_tot                         # rough ISCO estimate

        # whole band far beyond the end of any IMR model (M f > 1; PhenomX
        # stops at M f = 0.3).  A "few x f_isco" cut here would remove real
        # merger-ringdown signal.
        if freqs[0] * m_tot * lal.MTSUN_SI > 1.0:
            z = np.zeros(len(freqs), dtype=complex)
            return z, z.copy()

        if f_ref is None:
            f_ref = max(min(self.f_ref, 0.5 * f_isco), freqs[0])
        dist  = D_REF_MPC * 1e6 * lal.PC_SI

        if self.mode == "sequence":
            def call(fs):
                hp, hc = lalsim.SimInspiralChooseFDWaveformSequence(
                    phi_ref, m1_det * lal.MSUN_SI, m2_det * lal.MSUN_SI,
                    *spin1, *spin2, f_ref, dist, iota,
                    self.lal_params, self.approximant, self._freq_sequence(fs))
                return hp.data.data.copy(), hc.data.data.copy()
            try:
                hp, hc = call(freqs)
            except RuntimeError:
                # Some models refuse a sequence whose first point (taken as
                # f_min) lies beyond their end frequency: IMRPhenomXPHM raises
                # for M f > 0.3, while XAS returns zeros and XHM a negligible
                # tail.  The sequence interface evaluates each frequency
                # independently, so prepend M f = 0.05 to pass LAL's check and
                # drop it: the requested values are LAL's own.
                if freqs[0] * m_tot * lal.MTSUN_SI < 0.1:
                    raise
                hp, hc = call(np.concatenate(([0.05 / (m_tot * lal.MTSUN_SI)], freqs)))
                hp, hc = hp[1:], hc[1:]
        else:
            hp, hc = self._hp_hc_fd(freqs, m1_det, m2_det, iota, phi_ref,
                                    spin1, spin2, f_ref, dist)

        if self.isco_cut is not None:
            cut = freqs > self.isco_cut * f_isco
            hp[cut] = 0.0
            hc[cut] = 0.0
        return hp, hc

    def _hp_hc_fd(self, freqs, m1_det, m2_det, iota, phi_ref, spin1, spin2, f_ref, dist):
        lal, lalsim = self._lal, self._lalsim

        def call(delta_f):
            return lalsim.SimInspiralFD(
                m1_det * lal.MSUN_SI, m2_det * lal.MSUN_SI, *spin1, *spin2,
                dist, iota, phi_ref, 0.0, 0.0, 0.0,
                delta_f, freqs[0], freqs[-1] + 1.0, f_ref,
                self.lal_params, self.approximant)

        hp, hc = call(self.delta_f)
        if hp.deltaF > np.min(np.diff(freqs), initial=1.0):
            hp, hc = call(np.min(np.diff(freqs), initial=1.0))
        df  = hp.deltaF
        idx = np.rint(freqs / df).astype(int)
        if not np.allclose(idx * df, freqs):
            raise ValueError(f"frequency grid (deltaF={df}) is not aligned with `freqs`")
        out = []
        for h in (hp, hc):
            vals = np.zeros(len(freqs), dtype=complex)
            ok   = idx < len(h.data.data)
            vals[ok] = h.data.data[idx[ok]]
            out.append(vals)
        return out[0], out[1]


_REGISTRY = {}


def get_waveform_model(name="IMRPhenomXAS", **kwargs):
    """
    Usage:
        get_waveform_model("IMRPhenomXAS")
        get_waveform_model("IMRPhenomXPHM", f_ref=20.0)
        get_waveform_model("NRSur7dq4", mode="fd")
    """
    if name in _REGISTRY:
        return _REGISTRY[name](**kwargs)
    return LALWaveform(name, **kwargs)
