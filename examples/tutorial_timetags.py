"""Tutorial: from raw time tags to a single-emitter verdict with
uncertainties.

A time-tagging card records every click of the two HBT detectors. This
script makes such a record with the package's photon-by-photon
simulator (an illustrative emitter, detector jitter and dead time, a
fixed random seed), writes it to a PicoQuant PTU file and reads it back
as a lab would, builds the coincidence histogram, and answers "is this
a single emitter?" three ways: the exact interval and test for the
window g2, its Bayesian probability, and a model-based interval for
g2(0) itself.

Run it with `python examples/tutorial_timetags.py`. It printed
(sparq 0.13.0, Python 3.13, NumPy 2.5.3, SciPy 1.18.1):

399568 + 398752 clicks in 2 s (399.2 kcps counted)
window g2 = 0.133, exact 95 % interval [0.069, 0.233]; p-value of g2 >= 0.5: 6.6e-09
Bayesian P[window g2 < 0.5] = 1.000000
model g2(0) = 0.113, 95 % interval [0.068, 0.162] (true 0.1536)
"""
import os
import tempfile
import warnings

import numpy as np

from sparq import (DetectorImpairments, EmitterSite, HBTConfig, bayesian_g2,
                   correlate, load_ptu_timetags, profile_likelihood_ci,
                   save_ptu_t2, simulate_photon_stream, window_g2_ci)

# 1. Two seconds of clicks from an illustrative emitter: 15 ns
#    antibunching time, 250 ns shelving time, 400 kcps detected, 92 % of
#    the light from the emitter; detectors with 0.35 ns jitter and 22 ns
#    dead time.
site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=400.0,
                        rho=0.92, blinking=False), n_emitters=1)
T = 2.0
imp = DetectorImpairments(dead_time_ns=22.0, afterpulse_p=0.0,
                          sigma_irf_ns=0.35)
t_a, t_b = simulate_photon_stream(site, T, np.random.default_rng(7), imp)

# 2. The tags as a PTU file (HydraHarp, T2 mode, inputs 1 and 2, 1 ps
#    ticks), read back with the package's reader.
ch = np.r_[np.ones(t_a.size, int), np.full(t_b.size, 2)]
t = np.r_[t_a, t_b]
order = np.argsort(t, kind="stable")
path = os.path.join(tempfile.mkdtemp(), "hbt.ptu")
save_ptu_t2(path, ch[order], t[order], 1e-12, "HydraHarp2T2")
a, b = load_ptu_timetags(path, channel_a=1, channel_b=2)
r_a, r_b = a.size / T, b.size / T                 # counted rates (1/s)
print(f"{a.size} + {b.size} clicks in {T:.0f} s "
      f"({(r_a + r_b) / 1e3:.1f} kcps counted)")

# 3. The coincidence histogram: 121 bins of 1 ns around zero delay.
cfg = HBTConfig()
hist = correlate(a, b, cfg)

# 4. The window g2 (zero-delay bin against the far bins): exact 95 %
#    interval, and the exact test of "g2 >= 0.5" (no emitter model and no
#    count rate needed).
w = window_g2_ci(hist, cfg)
print(f"window g2 = {w['g2']:.3f}, exact 95 % interval [{w['lo']:.3f}, "
      f"{w['hi']:.3f}]; p-value of g2 >= 0.5: {w['p_value']:.1e}")
print(f"Bayesian P[window g2 < 0.5] = {bayesian_g2(hist, cfg).prob_below():.6f}")

# 5. g2(0) itself, from the three-level model with the detectors'
#    jitter, the flat level pinned by the counted rates to 1 %.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)  # tau2 search limit
    prof = profile_likelihood_ci(hist, T, 2.0 * np.sqrt(r_a * r_b), cfg,
                                 c0_prior=(1.0, 0.01))
print(f"model g2(0) = {prof['g2_hat']:.3f}, 95 % interval "
      f"[{prof['lo']:.3f}, {prof['hi']:.3f}] (true {site.g2_0:.4f})")
