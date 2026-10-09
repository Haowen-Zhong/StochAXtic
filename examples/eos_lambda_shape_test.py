# How much does the shape of Lambda(m) matter for Omega_GW?  The true Lambda(m) of a LAL EOS
# (TOV, interpolated as a JAX tide model) is compared with eos_lnpoly fits of degree 1 and 2 and
# with the old default (b1=-6, b2=0).  Uniform BNS on [1, 2] Msun, IMRPhenomXAS_NRTidalv2.
# Result (2026-10-02): below 300 Hz all agree to <=0.3%; degree 2 is <=0.6% up to 1 kHz;
# b2=0 is off by 3-7% at 1 kHz.  This motivated the default b2 = -3.  Run with the sgwb python.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import lal, lalsimulation as ls
from stochaxtic import jax_populations as jp
from stochaxtic.fast_omega import FastOmega
lal.ClobberDebugLevel(0)

def lam_table(name):
    fam = ls.CreateSimNeutronStarFamily(ls.SimNeutronStarEOSByName(name))
    Mmax = ls.SimNeutronStarMaximumMass(fam) / lal.MSUN_SI
    m = np.linspace(0.9, Mmax - 0.005, 200)
    L = np.array([2/3*ls.SimNeutronStarLoveNumberK2(mi*lal.MSUN_SI, fam) /
                  (mi*lal.MRSUN_SI/ls.SimNeutronStarRadius(mi*lal.MSUN_SI, fam))**5 for mi in m])
    return m, L, Mmax

def main():
    F = np.array([100., 200., 300., 500., 700., 1000., 1500.])
    fo = FastOmega(F, waveform="IMRPhenomXAS_NRTidalv2", tides=True, m_tot_range=(1.8, 6.0), q_min=0.3,
                   n_q=41, n_proc=8, verbose=False)
    for name in ["SLY", "MPA1", "H4"]:
        m, L, Mmax = lam_table(name)
        lm, lL = jnp.asarray(np.log(m)), jnp.asarray(np.log(L))
        def tab(mm, p, w=0.0, lm=lm, lL=lL, Mmax=Mmax):
            return jnp.exp(jnp.interp(jnp.log(mm), lm, lL)) * jp._between(mm, hi=Mmax, w=w)
        jp.TIDE_MODELS[f"tab_{name}"] = (tab, dict())
        y = np.log(m / 1.4); sel = (m >= 1.0) & (m <= 2.0)
        fits = {d: np.polyfit(y[sel], np.log(L[sel]), d)[::-1] for d in (1, 2, 3)}
        mp = {"m1_min": 1.0, "m2_min": 1.0, "m_max": 2.0}          # uniform BNS on [1, 2] Msun
        res = {}
        res["true"] = np.asarray(fo.make("uniform", "madau_dickinson", tide_model=f"tab_{name}")(mp, None, None, {}))
        om_poly = fo.make("uniform", "madau_dickinson", tide_model="eos_lnpoly")
        base = dict(m_NS_max=Mmax)
        for d, c in fits.items():
            res[f"deg{d}"] = np.asarray(om_poly(mp, None, None, {**base, "Lambda_14": np.exp(c[0]), "b1": c[1],
                                                                    "b2": c[2] if d >= 2 else 0.0}))
            if d == 3:   # eos_lnpoly has no b3: report deg-2 truncation of the cubic instead
                res.pop("deg3")
        res["default b1=-6,b2=0"] = np.asarray(om_poly(mp, None, None, {**base, "Lambda_14": np.exp(fits[1][0]), "b1": -6.0, "b2": 0.0}))
        res["Lambda=0"] = np.asarray(om_poly(mp, None, None, {**base, "Lambda_14": 1e-8}))
        print(f"\n{name}: Lambda1.4={np.exp(fits[1][0]):.0f}, Mmax={Mmax:.2f}.  Omega / Omega(true Lambda(m)) - 1  [%]")
        print("   f [Hz]        " + "  ".join(f"{f:7.0f}" for f in F))
        for k, v in res.items():
            if k == "true": continue
            print(f"   {k:19s}" + "  ".join(f"{100*(v[i]/res['true'][i]-1):+7.3f}" for i in range(len(F))))

if __name__ == "__main__":
    main()
