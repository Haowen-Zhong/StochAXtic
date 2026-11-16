"""Generate the StochAXtic icon and wordmark logos as self-contained SVGs."""
import sys
import numpy as np
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

out = sys.argv[1]

# ---------- spectra: individual sources and their sum, in log-log ----------
def source(lf, lf_merge, amp):
    """log10 of f*dE/df: slope 2/3 inspiral, 5/3 merger, Lorentzian ringdown."""
    f = 10.0 ** (lf - lf_merge)          # f / f_merge
    f2, sig = 2.2, 0.55                  # ringdown frequency and width (units of f_merge)
    insp = f ** (2 / 3)
    merg = f ** (5 / 3)
    ring = f2 ** (5 / 3) * f / f2 / (1 + ((f - f2) / (sig / 2)) ** 2) / 1.0
    ring *= 1.0 / (1 + 0)                # continuous at f2
    y = np.where(f < 1, insp, np.where(f < f2, merg, ring))
    return amp * y

lf = np.linspace(-0.6, 3.6, 1200)
def smooth(y, sig=0.045):
    """Gaussian blur in log f: rounds the piecewise kinks."""
    k = np.exp(-0.5 * (np.arange(-60, 61) * (lf[1] - lf[0]) / sig) ** 2)
    return np.convolve(y, k / k.sum(), mode="same")

# population: Gaussian in log10 f_merge (heavier / more redshifted sources peak lower)
MU, SIG = 1.62, 0.36
grid = np.linspace(MU - 3 * SIG, MU + 3 * SIG, 240)
wts = np.exp(-0.5 * ((grid - MU) / SIG) ** 2)
total = sum(w * source(lf, g, 1.0) for g, w in zip(grid, wts)) / wts.sum()
# a few representative sources drawn under the sum
reps = [1.02, 1.32, 1.62, 1.92, 2.22]
ys = [smooth(0.27 * np.exp(-0.5 * ((g - MU) / SIG) ** 2 * 0.6) * source(lf, g, 1.0)) for g in reps]
sel = (lf >= 0.0) & (lf <= 3.0)
lf, total, ys = lf[sel], total[sel], [y[sel] for y in ys]

# map to icon coordinates (512 box)
X0, X1, YB, YT = 64, 448, 400, 118
lo, hi = np.log10(total.max()) - 1.6, np.log10(total.max())

def to_xy(y):
    ly = np.log10(np.maximum(y, 1e-30))
    x = X0 + (lf - lf[0]) / (lf[-1] - lf[0]) * (X1 - X0)
    yy = YB - (ly - lo) / (hi - lo) * (YB - YT)
    return x, yy

def path(y, shift=0.0):
    x, yy = to_xy(y)
    yy = yy + shift
    keep = yy < YB + 30
    pts = [f"{a:.1f},{b:.1f}" for a, b in zip(x[keep], yy[keep])]
    return "M" + " L".join(pts)

thin = "\n".join(
    f'    <path d="{path(y, 0)}" stroke="#7dd3fc" stroke-opacity="0.42" '
    f'stroke-width="3.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
    for i, y in enumerate(ys))
main = path(total)

ICON_DEFS = """
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#141a3a"/>
      <stop offset="1" stop-color="#070a1c"/>
    </linearGradient>
    <linearGradient id="curve" gradientUnits="userSpaceOnUse" x1="64" y1="0" x2="448" y2="0">
      <stop offset="0" stop-color="#fbbf24"/>
      <stop offset="0.55" stop-color="#fb7185"/>
      <stop offset="1" stop-color="#a78bfa"/>
    </linearGradient>
    <clipPath id="clip"><rect width="512" height="512" rx="112"/></clipPath>"""

ICON_BODY = f"""
  <g clip-path="url(#clip)">
    <rect width="512" height="512" fill="url(#bg)"/>
{thin}
    <path d="{main}" stroke="url(#curve)" stroke-opacity="0.22" stroke-width="34" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="{main}" stroke="url(#curve)" stroke-width="15" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  </g>
  <rect x="1.5" y="1.5" width="509" height="509" rx="110.5" fill="none" stroke="#ffffff" stroke-opacity="0.12" stroke-width="3"/>"""

icon = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512" role="img" aria-label="StochAXtic">
  <defs>{ICON_DEFS}
  </defs>{ICON_BODY}
</svg>
"""
open(f"{out}/icon.svg", "w").write(icon)

# ---------- wordmark, converted to outlines ----------
def text_d(s, size, x0, baseline, weight="bold"):
    tp = TextPath((0, 0), s, size=size, prop=FontProperties(family="DejaVu Sans", weight=weight))
    v, c = tp.vertices, tp.codes
    d, i = [], 0
    P = lambda p: f"{x0 + p[0]:.2f},{baseline - p[1]:.2f}"
    while i < len(c):
        k = c[i]
        if k == 1:   d.append("M" + P(v[i])); i += 1
        elif k == 2: d.append("L" + P(v[i])); i += 1
        elif k == 3: d.append("Q" + P(v[i]) + " " + P(v[i + 1])); i += 2
        elif k == 4: d.append("C" + P(v[i]) + " " + P(v[i + 1]) + " " + P(v[i + 2])); i += 3
        else:        d.append("Z"); i += 1
    return "".join(d), tp.get_extents().x1

SIZE, BASE, TX = 150, 310, 600
parts, x = [], TX
for s, kind in (("Stoch", "ink"), ("AX", "hot"), ("tic", "ink")):
    d, adv = text_d(s, SIZE, x, BASE)
    parts.append((d, kind))
    x += adv + SIZE * 0.035
W = int(x + 40)

for name, ink in (("logo-light", "#111827"), ("logo-dark", "#f3f4f6")):
    word = "\n".join(
        f'  <path d="{d}" fill="{ink if kind == "ink" else "url(#hot)"}"/>' for d, kind in parts)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 512" width="{W}" height="512" role="img" aria-label="StochAXtic">
  <defs>{ICON_DEFS}
    <linearGradient id="hot" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#f59e0b"/>
      <stop offset="1" stop-color="#f43f5e"/>
    </linearGradient>
  </defs>{ICON_BODY}
{word}
</svg>
"""
    open(f"{out}/{name}.svg", "w").write(svg)
print("wrote", W)
