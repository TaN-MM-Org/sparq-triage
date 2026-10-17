"""Tutorial: the saturation curve of an emitter, from counts to R_inf
and P_sat with 95 % intervals.

A saturation measurement records the detected counts at a series of
excitation powers. This script makes such a data set (illustrative
numbers, a fixed random seed), fits it with `sparq.fit_saturation`, and
shows what happens when the powers stay far below saturation.

Run it with `python examples/tutorial_saturation.py`. It printed
(sparq 0.13.0, Python 3.13, NumPy 2.5.3, SciPy 1.18.1):

R_inf = 849.1 kcps, 95 % interval [843.2, 855.1] (true 850)
P_sat = 419.0 uW, 95 % interval [415.3, 422.7] (true 420)
background 89.9 cps/uW, dark 233 cps; fit p-value 0.95
signal fraction at 400 uW: 0.920; at 2400 uW: 0.770
powers up to 12 uW only: saturated False, P_sat upper end nan, warning issued: True
"""
import warnings

import numpy as np

from sparq import fit_saturation, saturation_curve

# 1. A synthetic measurement. Illustrative emitter: 850 kcps at full
#    saturation, P_sat = 420 uW, background light 90 cps per uW from
#    the sample, 250 cps of dark counts; 15 powers (the first with the
#    laser blocked, for the dark counts), 1 s of counting each.
truth = dict(R_inf=850e3, P_sat=420.0, bg_slope=90.0, dark=250.0)
power = np.array([0, 25, 50, 75, 100, 150, 200, 300, 400, 550, 700, 900,
                  1200, 1700, 2400.0])                      # uW
t_s = 1.0
rng = np.random.default_rng(2026)
counts = rng.poisson(saturation_curve(power, **truth) * t_s)

# 2. The fit: Poisson likelihood, profile-likelihood intervals (95 %).
fit = fit_saturation(power, counts, t_s)
print(f"R_inf = {fit['R_inf'] / 1e3:.1f} kcps, 95 % interval "
      f"[{fit['R_inf_lo'] / 1e3:.1f}, {fit['R_inf_hi'] / 1e3:.1f}] "
      f"(true {truth['R_inf'] / 1e3:.0f})")
print(f"P_sat = {fit['P_sat']:.1f} uW, 95 % interval "
      f"[{fit['P_sat_lo']:.1f}, {fit['P_sat_hi']:.1f}] "
      f"(true {truth['P_sat']:.0f})")
print(f"background {fit['bg_slope']:.1f} cps/uW, dark {fit['dark']:.0f} cps;"
      f" fit p-value {fit['p_value']:.2f}")

# 3. The emitter's share of the detected light at each power: it falls
#    as the background keeps growing after the emitter saturates.
i = int(np.argmin(np.abs(power - 400.0)))
print(f"signal fraction at {power[i]:.0f} uW: {fit['rho'][i]:.3f}; "
      f"at {power[-1]:.0f} uW: {fit['rho'][-1]:.3f}")

# 4. The same emitter measured only from 2 to 12 uW, 0.1 s per point:
#    the curve has hardly begun to bend, so the data cannot bound the
#    saturation from above, and the fit says so instead of guessing.
weak = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
weak_counts = rng.poisson(saturation_curve(weak, **truth) * 0.1)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    low = fit_saturation(weak, weak_counts, 0.1)
print(f"powers up to 12 uW only: saturated {low['saturated']}, P_sat upper "
      f"end {low['P_sat_hi']}, warning issued: {len(caught) == 1}")
