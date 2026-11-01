# constants.py
# Physical constants and defaults of StochAXtic.

from astropy.cosmology import Planck18

C_LIGHT_KMS = 299792.458                 # km/s (exact)
H0_KMS_MPC  = Planck18.H0.value          # km/s/Mpc
H0_SI       = Planck18.H0.si.value       # s^-1
H0_GYR      = H0_KMS_MPC * 1.0227121651e-3   # Gyr^-1
HUBBLE_TIME = Planck18.hubble_time.value     # Gyr
YR_S        = 365.25 * 86400.0           # Julian year in s

# Stokes intensities are accumulated in units of S_REF to keep squares
# (variances) far from underflow
S_REF = 1e-46

# default integration range and frequency grid
Z_MIN_DEFAULT = 0.2
Z_MAX_DEFAULT = 10.0
D_REF_MPC     = 1.0                      # waveforms are generated at 1 Mpc
