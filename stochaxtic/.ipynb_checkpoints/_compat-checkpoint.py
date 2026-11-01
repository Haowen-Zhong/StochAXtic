# _compat.py
# numpy / scipy version shims, imported by the package __init__ before
# anything else.
#
# numpy 2.0 renamed np.trapz -> np.trapezoid, and scipy 1.6 renamed
# scipy.integrate.cumtrapz -> cumulative_trapezoid (cumtrapz removed in 1.14).
# Whichever name is missing is aliased to the one that exists.

import numpy as np
import scipy.integrate as _si

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

if not hasattr(_si, "cumulative_trapezoid"):
    _si.cumulative_trapezoid = _si.cumtrapz
