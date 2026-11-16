# numpy / scipy version shims, imported first by the package __init__:
# np.trapz -> np.trapezoid (numpy 2.0), scipy.integrate.cumtrapz ->
# cumulative_trapezoid (scipy 1.6; cumtrapz removed in 1.14).  A missing name is
# aliased to the one that exists.

import numpy as np
import scipy.integrate as _si

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

if not hasattr(_si, "cumulative_trapezoid"):
    _si.cumulative_trapezoid = _si.cumtrapz
