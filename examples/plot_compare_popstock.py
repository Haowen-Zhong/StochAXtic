# Plot the FastOmega vs popstock comparison written by compare_popstock.py.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_popstock")
WAVEFORMS = ["IMRPhenomD", "IMRPhenomXAS", "IMRPhenomXHM"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6})

r = np.load(os.path.join(HERE, "results.npz"))
f = r["freqs"]
n = len(r["samples_z"])

fig, ax = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
for wf, c in zip(WAVEFORMS, SERIES):
    fast, pop, sig = r[f"fast_popstock_mixed_{wf}"], r[f"pop_{wf}"], r[f"popsig_{wf}"]
    ax[0].loglog(f, fast, color=c, label=f"{wf} FastOmega")
    ax[0].errorbar(f[::6], pop[::6], yerr=sig[::6], fmt="o", ms=3, color=c, mec="white", mew=0.6,
                   label=f"{wf} popstock")
    rel, rs = fast / pop - 1, fast / pop * sig / pop
    ax[1].semilogx(f, 100 * rel, color=c, label=wf)
    ax[1].fill_between(f, -200 * rs, 200 * rs, color=c, alpha=0.12, lw=0)
    if wf != "IMRPhenomD":
        fr = fast / r["fast_popstock_mixed_IMRPhenomD"]
        pr, ps = r[f"popratio_{wf}"], r[f"popratiosig_{wf}"]
        k = f <= 1100
        ax[2].semilogx(f[k], 100 * (fr[k] - 1), color=c, label=f"{wf} / D  FastOmega")
        ax[2].fill_between(f[k], 100 * (pr[k] - 1 - 2 * ps[k]), 100 * (pr[k] - 1 + 2 * ps[k]), color=c, alpha=0.25, lw=0,
                           label=f"{wf} / D  popstock ±2σ")
ax[0].set(xlabel="$f$ [Hz]", ylabel=r"$\Omega_{\rm GW}(f)$", title="FastOmega (lines) vs popstock (points)")
ax[1].axhline(0, color="0.3", lw=0.8)
ax[1].set(xlabel="$f$ [Hz]", ylabel="FastOmega / popstock − 1  [%]", ylim=(-6, 6),
          title=f"FastOmega / popstock − 1 (band ±2σ, N={n:,})")
ax[2].axhline(0, color="0.3", lw=0.8)
ax[2].set(xlabel="$f$ [Hz]", ylabel="ratio to IMRPhenomD − 1  [%]",
          title="ratio to PhenomD (popstock band ±2σ)")
for a in ax:
    a.legend(fontsize=7, frameon=False)
fig.savefig(os.path.join(HERE, "fastomega_vs_popstock.pdf"))
fig.savefig(os.path.join(HERE, "fastomega_vs_popstock.png"), dpi=130)

print(f"N = {n}, z in [{r['samples_z'].min():.3f}, {r['samples_z'].max():.2f}]")
sel = [np.argmin(abs(f - x)) for x in (20, 50, 100, 200, 500, 1000, 2000)]
for wf in WAVEFORMS:
    fast, pop, sig = r[f"fast_popstock_mixed_{wf}"], r[f"pop_{wf}"], r[f"popsig_{wf}"]
    pull = (fast - pop) / sig
    print(f"\n{wf}: popstock {float(r[f'time_pop_{wf}']):.0f} s, FastOmega call {float(r[f'time_fast_{wf}']):.1f} ms;"
          f"  chi2/dof = {np.mean(pull ** 2):.2f}, max |pull| = {np.abs(pull).max():.1f}")
    print("   f [Hz]   FO/pop-1 [%]   sigma [%]   pull   (mixed-cosmo vs consistent: [%])")
    for i in sel:
        cc = r[f"fast_popstock_mixed_{wf}"][i] / r[f"fast_consistent_{wf}"][i] - 1
        print(f"   {f[i]:6.0f}   {100*(fast[i]/pop[i]-1):+8.2f}     {100*sig[i]/pop[i]:6.2f}   {pull[i]:+5.1f}"
              f"   {100*cc:+.2f}")
    if wf != "IMRPhenomD":
        fr = fast / r["fast_popstock_mixed_IMRPhenomD"]
        pr, ps = r[f"popratio_{wf}"], r[f"popratiosig_{wf}"]
        rp = (fr - pr) / ps
        for i in sel[:-1]:
            print(f"      {f[i]:6.0f} Hz  {wf}/D-1: FastOmega {100*(fr[i]-1):+7.2f}%   popstock {100*(pr[i]-1):+7.2f} ± {100*ps[i]:.2f}%")
        print(f"   ratio to D: chi2/dof = {np.mean(rp ** 2):.2f}, max |pull| = {np.abs(rp).max():.1f}")
