# SPARQ

[![Tests](https://github.com/TaN-MM-Org/sparq-triage/actions/workflows/tests.yml/badge.svg)](https://github.com/TaN-MM-Org/sparq-triage/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/sparq-triage?label=PyPI&color=blue)](https://pypi.org/project/sparq-triage/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22278040-blue)](https://doi.org/10.5281/zenodo.22278040)

`sparq` (installed as `sparq-triage`) is a Python package for one
question that comes up again and again in quantum-optics labs: **does
this light source give out one photon at a time?** A source that does
is a *single-photon emitter*, the building block of many quantum
communication and computing schemes. The standard test is to split the
light onto two detectors and record how often both fire at nearly the
same moment. A true single emitter almost never makes both fire
together, so the record shows a dip at zero delay.

The package works from that record (a histogram, or the raw detector
time stamps) and answers:

- How deep is the dip, and how sure can I be about it?
- Is the dip deep enough to call the site a single emitter, and with
  what probability?
- How long must I measure before the answer is reliable, and can a
  longer run ever make it reliable?
- Can I stop measuring early because the data already suffice?
- How much do background light and detector dead time change the
  numbers?
- For a heralded pair source, what purity does a measured
  coincidence-to-accidental ratio allow?

The analysis core needs only NumPy and SciPy. An optional
machine-learning layer (PyTorch) holds the neural estimators and the
automated triage agent of the manuscript *"Closed-loop, event-driven
machine learning for autonomous triage of single-photon emitters"*;
the
[companion repository](https://github.com/Tanvir-Mahmud-Mahim/a-spiking-RL-triage-of-solid-state-single-photon-emitters)
reproduces the paper itself.

## Contents

- [A short guide to the words used here](#a-short-guide-to-the-words-used-here)
- [Install, requirements and units](#install-requirements-and-units)
- [Examples](#examples) (each with the output it prints)
- [What is in the package](#what-is-in-the-package)
- [When it refuses, and why](#when-it-refuses-and-why)
- [How the results are checked](#how-the-results-are-checked)
- [Corrections in earlier versions](#corrections-in-earlier-versions)
- [Limits](#limits)
- [Where it comes from](#where-it-comes-from)
- [Citing, support and license](#citing-support-and-license)

## A short guide to the words used here

- **HBT experiment** (Hanbury Brown and Twiss) -- the light is split
  onto two detectors, A and B, and every pair of clicks is sorted by
  the time between them (the **delay**, in nanoseconds).
- **Coincidence histogram** -- the count of click pairs in each small
  delay interval (**bin**). By default the package uses 121 bins of
  1 ns, from -60.5 ns to +60.5 ns (`HBTConfig`).
- **Flat level, accidentals** -- two detectors that see unrelated
  light still fire together by chance. This chance rate sets the
  **flat level** of the histogram far from zero delay.
- **g2(tau)** -- the histogram divided by that flat level. It is 1 for
  unrelated clicks. **g2(0)**, its value at zero delay, is the key
  number: 0 for a perfect single emitter, 1 - 1/N for N equal
  independent emitters (0.5 for two). The usual rule, used throughout
  this package, is that g2(0) below 0.5 means "single emitter".
- **Antibunching time** `tau1` -- how fast the dip closes around zero
  delay. **Shelving (bunching) time** `tau2` and **amplitude** `a` --
  many emitters sometimes park in a dark state; this lifts g2 above 1
  (a "shoulder") at delays around `tau2`. The model used is
  `g2(tau) = 1 - (1 + a) exp(-|tau|/tau1) + a exp(-|tau|/tau2)`.
- **Signal fraction** `rho` -- the share of detected light that comes
  from the emitter, S/(S+B). Unrelated background light makes the dip
  shallower: `g2_meas = 1 + rho^2 (g2_true - 1)`.
- **Instrument response (IRF)** -- detector timing jitter. It blurs
  the dip, so the raw histogram shows a shallower dip than the
  emitter really has. The fits include it (`sigma_irf`, 0.35 ns per
  detector by default).
- **Time tags** -- the raw list of (detector channel, click time)
  pairs that a time-tagging card records.
- **Site** -- one candidate spot on a sample, described in code by an
  `EmitterSite`.
- **Twin** -- the package's simulator of the HBT experiment. It is
  used to test the analysis against a known truth.
- **Confidence interval, credible interval** -- a range that should
  hold the true value with a stated probability. **Bootstrap**,
  **profile likelihood** and **Bayesian posterior** are three standard
  ways to get one; the package offers all three. A **posterior
  probability** such as P[g2 < 0.5] is the probability, given the data,
  that the value lies below 0.5.
- **SPRT** (sequential probability ratio test) -- a test that looks at
  the data as it arrives and stops as soon as the evidence is strong
  enough.
- **Heralded source** -- a source that makes photons in pairs; detecting
  one photon announces ("heralds") the other. **CAR**
  (coincidence-to-accidental ratio) is its usual quality number.
- **Poisson counts, standard error** -- counts of random, independent
  clicks follow the Poisson distribution: a bin that expects N counts
  scatters by about sqrt(N) from run to run. The standard error of an
  average is how much that average itself scatters.
- **Reduced chi-square** -- the average of (deviation / expected
  scatter)^2 over all bins; it is close to 1 when the scatter of the
  data matches the stated error bars.
- **Emitter types** named in the code: **NV** (nitrogen-vacancy centre
  in diamond), **SiV** (silicon-vacancy centre in diamond), and defects
  in **hBN** (hexagonal boron nitride) and **GaN** (gallium nitride).

## Install, requirements and units

```
pip install sparq-triage        # analysis core: NumPy and SciPy only
pip install sparq-triage[ml]    # adds PyTorch for the machine-learning layer
```

It needs Python 3.10 or newer, NumPy 1.24 or newer and SciPy 1.10 or
newer. The `ml` extra adds PyTorch 2.0 or newer. `import sparq` and
everything listed under "Analysis core" below work without PyTorch.

Units:

- delays, lifetimes (`tau1`, `tau2`), bin widths, IRF widths and dead
  times: nanoseconds (ns);
- acquisition times (`T_s`): seconds;
- count rates: `rate_kcps` in an `EmitterSite` is kilocounts per
  second, summed over both detectors; `r_hat` and `rate_cps` in
  results are counts per second, also summed over both detectors;
  `deadtime_corrected_rate` takes one detector's rate in counts per
  second;
- g2, `a`, `rho`, CAR: no unit.

## Examples

Each example below runs as written, and the output shown is what it
printed with sparq-triage 0.9.1. All emitter numbers are illustrative
values, not measurements. The data are simulated with a fixed random
seed, so you get the same output.

### 1. Analyze a measured histogram

```python
import os, tempfile
import numpy as np
from sparq import (EmitterSite, HBTConfig, expected_histogram,
                   save_hbt_csv, load_hbt_csv, analyze_histogram)

# Illustrative emitter: 15 ns antibunching time, 250 ns shelving time,
# 150 kcps detected, 95 % of the light from the emitter itself.
site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                        rho=0.95, blinking=False), n_emitters=1)
print(f"true g2(0) of this site: {site.g2_0:.4f}")

# A simulated 30 s measurement on a 0.5 ns grid whose zero is shifted
# by 7 ns, as a correlator with a cable delay would record it.
grid = HBTConfig(tau_max=90.0, n_bins=360)
rng = np.random.default_rng(0)
counts = rng.poisson(expected_histogram(site, 30.0, grid))

path = os.path.join(tempfile.mkdtemp(), "my_hbt.csv")
save_hbt_csv(path, grid.bin_centers + 7.0, counts)   # header: delay_ns,counts
delay, counts = load_hbt_csv(path)

res = analyze_histogram(delay, counts, T_s=30.0, n_bootstrap=30, seed=1)
print(f"dip found at {res['center']:.2f} ns")
print(f"g2(0) = {res['g2_0']:.3f}, 68 % interval "
      f"[{res['g2_0_low']:.3f}, {res['g2_0_high']:.3f}]")
print("below 0.5:", res["single_emitter"],
      "| whole interval below 0.5:", res["single_emitter_confident"])
```

```
true g2(0) of this site: 0.0975
dip found at 8.25 ns
g2(0) = 0.094, 68 % interval [0.077, 0.150]
below 0.5: True | whole interval below 0.5: True
```

`analyze_histogram` finds the dip, re-bins the data onto the 1 ns
analysis grid centred on it, estimates the flat level from the far
bins, and fits the emitter model with the instrument response
included. The interval comes from a **bootstrap**: the counts are
redrawn at random (Poisson) `n_bootstrap` times and the whole analysis
is repeated on each copy. A real analysis should use more redraws than
the 30 used here to keep the example fast (the default is 200). The
dip position is only needed to centre the grid; here it is 1.25 ns
from the true 7 ns shift. `single_emitter_confident` is True when the
whole interval lies on one side of 0.5 (here: below it, as
`single_emitter` is True). Your own file needs the header
`delay_ns,counts` and one row per bin. For a pulsed-laser measurement
(a comb of peaks), use `analyze_pulsed` instead.

### 2. The probability that a site is a single emitter

```python
import numpy as np
from sparq import EmitterSite, HBTConfig, expected_histogram, bayesian_g2

params = dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0, rho=0.95,
              blinking=False)                       # illustrative values
cfg = HBTConfig()                                   # 121 bins of 1 ns
rng = np.random.default_rng(2)
for n in (1, 2):                                    # one emitter, then two
    site = EmitterSite(dict(params), n_emitters=n)
    hist = rng.poisson(expected_histogram(site, 30.0, cfg)).astype(float)
    post = bayesian_g2(hist, cfg)       # zero-delay bin against the far bins
    lo, hi = post.credible_interval(0.95)
    print(f"{n} emitter(s): true g2(0) {site.g2_0:.3f}; window g2 in "
          f"[{lo:.3f}, {hi:.3f}] (95 %); P[g2 < 0.5] = {post.prob_below(0.5):.4f}")
```

```
1 emitter(s): true g2(0) 0.098; window g2 in [0.084, 0.185] (95 %); P[g2 < 0.5] = 1.0000
2 emitter(s): true g2(0) 0.549; window g2 in [0.442, 0.657] (95 %); P[g2 < 0.5] = 0.2149
```

`bayesian_g2` compares the counts in the zero-delay bin with the
counts in the far bins (|delay| of at least 0.65 of the window edge).
For Poisson counts this comparison has an exact closed-form answer
(a scaled beta-prime distribution), so the interval and the
probability need no simulation and no fitting. What it estimates is
the *raw* g2 averaged over the central bin, with no model of the dip
shape. When the dip is wider than the bin, as here, the bin also
covers the walls of the dip, so the value is an upper bound on the
true g2(0): a cautious number.

### 3. A model-based interval, and why the singles rate helps

```python
import numpy as np
from sparq import EmitterSite, HBTConfig, expected_histogram, profile_likelihood_ci

site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                        rho=0.95, blinking=False), n_emitters=1)
cfg = HBTConfig()
hist = np.random.default_rng(0).poisson(
    expected_histogram(site, 60.0, cfg)).astype(float)

free = profile_likelihood_ci(hist, 60.0, 150e3, cfg)
pinned = profile_likelihood_ci(hist, 60.0, 150e3, cfg, c0_prior=(1.0, 0.01))
print(f"true g2(0): {site.g2_0:.4f}")
print(f"flat level free:        [{free['lo']:.3f}, {free['hi']:.3f}]")
print(f"flat level pinned (1%): [{pinned['lo']:.3f}, {pinned['hi']:.3f}]")
```

```
true g2(0): 0.0975
flat level free:        [0.067, 0.178]
flat level pinned (1%): [0.084, 0.126]
```

`profile_likelihood_ci` gives a 95 % interval for g2(0) itself, using
the full emitter model and the exact Poisson likelihood. With the
flat level left free, a stronger, slower shoulder can trade against a
lower flat level, and the histogram alone cannot tell them apart, so
the interval is wide. The detectors' own singles rates fix the flat
level independently. Pass the rate as `r_hat` (here 150e3 counts per
second) and its relative uncertainty as `c0_prior=(1.0, sd)`, and the
interval narrows.

### 4. Plan the measurement before running it

```python
from sparq import site_from_numbers, required_acquisition_time

site = site_from_numbers(tau1_ns=12.0, tau2_ns=200.0, a=0.3,
                         rate_kcps=120.0, rho=0.97)        # illustrative
T, report = required_acquisition_time(site, confidence=0.95)
print(f"typical-data time: {T:.2f} s")
print(f"P[g2 < 0.5] at that time: {report['prob_below']:.4f}")
print(f"window g2 the data settle on: {report['g2_window']:.3f}")

crowded = site_from_numbers(tau1_ns=12.0, tau2_ns=200.0, a=0.3,
                            rate_kcps=120.0, rho=0.97, n_emitters=4)
try:
    required_acquisition_time(crowded)
except ValueError as err:
    print("refused:", err)
```

```
typical-data time: 1.38 s
P[g2 < 0.5] at that time: 0.9500
window g2 the data settle on: 0.082
refused: the site's window-averaged g2 is 0.738 >= the threshold 0.5: the window ratio does not change with acquisition time, so no run length certifies this site below the threshold
```

The planner applies the Bayesian verdict of example 2 to the
*average* histogram the site would give, and searches for the
shortest time at which it reaches the confidence. Real runs scatter
around that average, so plan a margin: in the test suite, 4 times the
planned time certified more than 90 % of 200 simulated runs (for the
first site above). The refusal is exact, not a timeout: both the
zero-delay counts and the far counts grow in proportion to the
measuring time, so their ratio never changes, and a site whose ratio
is not below the threshold can never be certified.

### 5. Stop as soon as the data suffice

```python
import numpy as np
from sparq import EmitterSite, HBTConfig, expected_histogram, SPRTCertifier

params = dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0, rho=0.95,
              blinking=False)                       # illustrative values
one = EmitterSite(dict(params), n_emitters=1)       # "accept" hypothesis
two = EmitterSite(dict(params), n_emitters=2)       # "reject" hypothesis
cfg = HBTConfig()

cert = SPRTCertifier(one, two, cfg, alpha=0.05, beta=0.05)
t_acc, t_rej = cert.expected_times()
print(f"Wald's expected time to decide: {t_acc:.2f} s (one emitter), "
      f"{t_rej:.2f} s (two)")

rng = np.random.default_rng(1)
step = 0.05                                         # 50 ms per increment
mu = expected_histogram(one, step, cfg)             # the truth: one emitter
while cert.decision == "continue":
    cert.update(rng.poisson(mu), step)
print(f"decision: {cert.decision} after {cert.T_total:.2f} s")
```

```
Wald's expected time to decide: 0.22 s (one emitter), 0.17 s (two)
decision: accept after 0.30 s
```

"Wald's expected time" is the standard approximate formula for the
average time such a test needs (it ignores the small overshoot past
the stopping limit, so real runs take a little longer).
`SPRTCertifier` compares two fully specified possibilities (here: the
same emitter alone, or as a pair) and adds up the evidence from each
new piece of data. It stops when the evidence passes limits set by the
error rates you choose (`alpha`: wrongly accepting a pair, `beta`:
wrongly rejecting a single emitter). The emitter parameters in the two
possibilities must come from somewhere, usually a calibration fit, so
the error rates hold only approximately; the tests measure them on
simulated runs (see below).

### 6. Background and dead-time corrections

```python
from sparq import background_corrected_g2, signal_fraction, deadtime_corrected_rate

rho = signal_fraction(signal_rate=90e3, background_rate=10e3)
out = background_corrected_g2(0.28, rho, ci=(0.22, 0.34))   # illustrative
print(f"signal fraction {rho:.2f}")
print(f"background-corrected g2(0) = {out['g2_corrected']:.4f}, "
      f"interval ({out['ci'][0]:.4f}, {out['ci'][1]:.4f})")

print(f"true rate behind 1 Mcps measured with 45 ns dead time: "
      f"{deadtime_corrected_rate(1e6, 45.0):.0f} cps")
try:
    deadtime_corrected_rate(25e6, 45.0)
except ValueError as err:
    print("refused:", err)
```

```
signal fraction 0.90
background-corrected g2(0) = 0.1111, interval (0.0370, 0.1852)
true rate behind 1 Mcps measured with 45 ns dead time: 1047120 cps
refused: measured rate 2.5e+07 cps is at or above the saturation rate 1/tau_d = 2.22e+07 cps of the non-paralyzable model; the correction has no solution there
```

The background correction undoes `g2_meas = 1 + rho^2 (g2_true - 1)`
(Brouri, Beveratos, Poizat and Grangier, Opt. Lett. 25, 1294 (2000)),
the same relation the package's simulator uses. A corrected value
below 0 is reported as 0, and the uncorrected result is kept in
`g2_uncorrected_inverse`. The dead-time correction undoes
`r_meas = r / (1 + r tau_d)` for a detector that is blind for `tau_d`
after each click; a measured rate at or above `1/tau_d` has no
solution and is refused.

### 7. From raw time tags to g2

```python
import numpy as np
from sparq import (EmitterSite, HBTConfig, simulate_photon_stream, g2_measured,
                   correlate, correlate_start_stop, normalize_g2)

site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                        rho=0.95, blinking=False), n_emitters=1)
cfg = HBTConfig()
rng = np.random.default_rng(2)
t_a, t_b = simulate_photon_stream(site, 2.0, rng)   # 2 s of photon tags (ns)

hist = correlate(t_a, t_b, cfg)                     # every pair counts
out = normalize_g2(hist, len(t_a), len(t_b), 2.0, cfg)
c = cfg.n_bins // 2                                 # the zero-delay bin
print(f"{len(t_a)} + {len(t_b)} tags, {out['accidentals_per_bin']:.1f} "
      "accidental coincidences per bin")
print(f"g2 at zero delay: {out['g2'][c]:.2f} +- {out['sigma'][c]:.2f}")
print(f"g2 far out (mean of 30 edge bins): "
      f"{np.r_[out['g2'][:15], out['g2'][-15:]].mean():.3f}")
edges = np.r_[cfg.bin_centers[:15], cfg.bin_centers[-15:]]
print(f"model g2 at the same delays: "
      f"{g2_measured(edges, 15.0, 250.0, 0.3, 1, 0.95).mean():.3f}")
ss = correlate_start_stop(t_a, t_b, cfg)            # first stop per start only
print(f"pairs: all-pairs {hist.sum():.0f}, start-stop {ss.sum():.0f}")
```

```
149933 + 149856 tags, 11.2 accidental coincidences per bin
g2 at zero delay: 0.00 +- 0.09
g2 far out (mean of 30 edge bins): 1.225
model g2 at the same delays: 1.183
pairs: all-pairs 1310, start-stop 1308
```

`normalize_g2` divides the coincidences by the level two unrelated
detectors would give, `N1 N2 w / T` per bin (N1, N2 the click counts,
w the bin width, T the measuring time), and gives Poisson one-sigma
error bars. The far bins sit above 1 here because of this emitter's
shelving shoulder, as the model line shows; they are not an error.
`correlate` counts every pair of clicks (as a software correlator
does). `correlate_start_stop` copies an older kind of timing card that
records only the first stop in each start's window; at high rates it loses
pairs, so fit the `correlate` histogram. Real tag files can be read
with `load_timetags_csv` (two columns: channel, time in ns).

### 8. Heralded pair sources

```python
from sparq import heralded_g2_limit, car_for_purity

for car in (10.0, 100.0):
    print(f"CAR {car:5.0f}: g2_h(0) at least "
          f"{heralded_g2_limit(car):.4f} (laser-pumped), "
          f"{heralded_g2_limit(car, 'thermal'):.4f} (thermal)")
print(f"CAR needed for g2_h(0) = 0.01: {car_for_purity(0.01):.1f} (laser-pumped), "
      f"{car_for_purity(0.01, 'thermal'):.1f} (thermal)")
```

```
CAR    10: g2_h(0) at least 0.1900 (laser-pumped), 0.3471 (thermal)
CAR   100: g2_h(0) at least 0.0199 (laser-pumped), 0.0394 (thermal)
CAR needed for g2_h(0) = 0.01: 199.5 (laser-pumped), 398.5 (thermal)
```

For a heralded source the closed forms are
`g2_h(0) = (2 CAR - 1)/CAR^2` for Poissonian (laser-pumped) pair
statistics and `(4 CAR + 2)/(CAR + 1)^2` for thermal statistics
(H. Wang et al., arXiv:2404.03236), and `car_for_purity` inverts them.
They are the lowest values the pair statistics allow. A real source
sits at or above them, because background and detector effects only
add to g2_h. They do not apply to single emitters.

## What is in the package

Everything below except the last group is importable straight from
`sparq`. Each function's docstring (`help(sparq.bayesian_g2)`, for
example) gives its inputs, units and conventions.

**Analysis core: measured data** (NumPy and SciPy only)

- `analyze_histogram` -- g2(0) of a CW (continuous laser) histogram
  with a bootstrap interval (example 1). `analyze_pulsed` -- the same
  for a pulsed-laser histogram, from peak areas.
- `fit_g2_histogram` -- the model fit that `analyze_histogram` runs
  once per copy; returns `(g2_0, ok)`.
- `t1_bounds` / `t2_bounds` (arguments of `analyze_histogram`,
  `fit_g2_histogram`, `profile_likelihood_ci`) -- the ranges, in ns,
  searched for `tau1` and `tau2`. The defaults (0.3 to 80 ns and 50 to
  800 ns) suit emitters on the timescales of NV centres in diamond. A much faster emitter needs
  its own ranges; a range that excludes the true value makes the fit
  quietly wrong (the tests show this).
- `profile_likelihood_ci` -- model-based interval for g2(0), with
  optional `c0_prior` (example 3).
- `bayesian_g2`, `G2Posterior` -- the exact posterior of the raw
  central-window g2: `pdf`, `cdf`, `ppf`, `mean`, `median`, `mode`,
  `credible_interval`, `prob_below` (example 2).
- `SPRTCertifier` -- sequential test with `update`, `kl_rates`,
  `expected_times` (example 5).
- `load_hbt_csv`, `save_hbt_csv` -- the `delay_ns,counts` file format.
  `rebin_real` -- re-bins a histogram onto the analysis grid, centred
  on the dip. `robust_flat_rate` -- the singles rate implied by the
  median of the far bins.
- `background_corrected_g2`, `signal_fraction`,
  `deadtime_corrected_rate` -- closed-form corrections (example 6).
- `load_timetags_csv`, `save_timetags_csv`, `correlate`,
  `correlate_start_stop`, `normalize_g2` -- time tags to g2
  (example 7).
- `site_from_numbers`, `expected_posterior`,
  `required_acquisition_time` -- measurement planning (example 4).
- `heralded_g2_limit`, `car_for_purity` -- heralded sources
  (example 8).

**Analysis core: models and simulators**

- `g2_three_level` -- the ideal model above. `g2_measured` -- the same
  with several emitters, background and instrument response.
  `g2_zero` -- g2(0) of a site with no instrument response.
- `HBTConfig` -- the histogram grid (`tau_max`, `n_bins`,
  `sigma_irf`; `bin_width`, `bin_centers`).
- `EmitterSite` -- one site (`params` dict and `n_emitters`; `g2_0`,
  and `is_good` = g2(0) < 0.5, brighter than 60 kcps and not
  blinking).
- `Platform`, `PLATFORMS`, `register_platform`, `sample_site` --
  parameter ranges per emitter type. Four are built in: NV, hBN, GaN
  and SiV. Their ranges are described in the source as anchored to
  published photophysics, with the citations in the manuscript; the
  package itself does not list them. `register_platform` adds your own.
- `expected_histogram`, `sample_histogram`, `sample_event_stream` --
  the fast simulator: the mean histogram, Poisson draws from it, and
  the same split into time slices.
- `simulate_photon_stream`, `DetectorImpairments` -- a slower
  photon-by-photon simulator with optional detector jitter, dead time
  and afterpulsing, and blinking.
- `liouvillian`, `steady_state`, `g2_exact`, `effective_params`,
  `rates_from_site` -- an exact three-level rate-equation model of the
  emitter, used to check the two-exponential formula.
- `expected_hist_pulsed`, `peak_shape`, `calibrate_comb`,
  `g2_peak_area` -- the pulsed-laser simulator and the peak-area
  analysis behind `analyze_pulsed` (default repetition period
  12.5 ns, i.e. 80 MHz).
- `__version__`.

**Machine-learning layer** (needs the `ml` extra; import the modules
explicitly)

- `sparq.estimators` -- a 1-D convolutional network (a standard
  pattern-recognizing neural network) on histograms (`HistCNN`,
  `TriageCNN`), a spiking neural network (whose model neurons pass
  on-off pulses, suited to data arriving over time) on time-sliced
  data (`SpikingG2Net`), their training loops and `evaluate`.
- `sparq.twin_torch` -- a differentiable version of the simulator, as
  a function of laser power and window width.
- `sparq.rl_env` -- a simulated field of sites to triage (`TriageEnv`):
  at each site the agent measures a little longer (three dwell
  times), rejects the site or certifies it. Also two simple fixed
  strategies to compare against. `sparq.sac_per` -- a soft
  actor-critic agent, a standard reinforcement-learning method that
  learns such decisions by trial and error (Haarnoja et al., ICML
  2018; discrete-action variant: Christodoulou, arXiv:1910.07207),
  with prioritized replay, which re-uses past experience and replays
  the most surprising steps more often (Schaul et al., ICLR 2016).
- `sparq.gnn` -- a graph encoder: a neural network that turns an
  emitter's energy-level diagram into a list of numbers (Gilmer et
  al., 2017).
- `sparq.datasets` -- batch generators for training (NumPy only) and
  `load_fisequr`, a loader for a public quantum-dot data set (see
  [Where it comes from](#where-it-comes-from)).

## When it refuses, and why

`sparq` raises a `ValueError` instead of guessing when:

- delays and counts differ in length or are not 1-D, or counts are
  negative (`analyze_histogram`, `analyze_pulsed`);
- the data's bins are wider than the analysis grid's bins, which would
  leave grid bins empty (`rebin_real`, and so `analyze_histogram`;
  new in 0.9.1 -- pass a coarser `cfg`);
- a `t1_bounds` / `t2_bounds` range is not `(low, high)` with
  0 < low < high;
- a histogram has too few counts to profile, or `c0_prior` is not a
  pair of positive numbers (`profile_likelihood_ci`);
- `bayesian_g2` gets a histogram of the wrong length, negative or
  non-finite counts, a grid with no bin centred on zero, an even or
  too-wide central window, an empty reference window, windows that
  overlap, or a prior with shape <= 0 or rate < 0; `G2Posterior.mean`
  when the mean does not exist (almost no far counts), and quantile or
  interval levels outside (0, 1);
- `SPRTCertifier` gets error rates outside (0, 1), a hypothesis with
  zero expected counts in some bin, data on a different grid, or a
  non-positive time step;
- `site_from_numbers` gets non-positive or non-finite times or rate,
  a negative `a`, `rho` outside (0, 1], or a non-integer or zero
  emitter count; `expected_posterior` a non-positive time;
- `required_acquisition_time` gets a confidence outside (0, 1) or a
  bad `t_max_s`, a site whose window g2 is not below the threshold (no
  run can certify it; example 4), or a confidence not reached within
  `t_max_s` (the message gives the probability reached);
- `background_corrected_g2` gets `rho` outside (0, 1] or an interval
  with low > high; `signal_fraction` negative rates or a zero total;
  `deadtime_corrected_rate` negative inputs or a measured rate at or
  above `1/tau_d` (example 6);
- `heralded_g2_limit` gets a CAR below 1 or not finite, or an unknown
  `statistics`; `car_for_purity` a target outside (0, 1] (laser-pumped)
  or (0, 1.5] (thermal);
- a histogram file has the wrong header, a row without exactly two
  fields, fewer than 5 bins, non-finite values, negative counts or a
  delay axis that is not strictly increasing (`load_hbt_csv`);
  `save_hbt_csv` gets arrays of unequal length or fewer than 5 bins;
- a time-tag file is empty, does not have two columns, has non-finite
  entries or unknown channels (`load_timetags_csv`; use `channel_a`,
  `channel_b` to map your numbering); `save_timetags_csv` gets arrays
  of different shapes;
- time tags are not sorted in time: the stop tags `t_b` for
  `correlate` (new in 0.9.1), both channels for `correlate_start_stop`;
- `normalize_g2` gets a histogram of the wrong length, negative or
  non-finite counts, click counts that are not positive, or a
  non-positive time;
- `register_platform` gets something that is not a `Platform`, a name
  already in use (unless `overwrite=True`), a range with low > high,
  non-positive bounds (negative ones for `a_rng`), `rho` outside (0, 1]
  or a blinking probability
  outside [0, 1];
- `load_fisequr` is not given an existing data directory.

## How the results are checked

88 automated tests run on every push to `main` and every pull request. In CI they run
on Python 3.10, 3.11, 3.12, 3.13 and 3.14 with CPU PyTorch; once more
on Python 3.14 without PyTorch, where the three test files of the
machine-learning parts (14 tests) are skipped; and once more on Python 3.10 with the oldest versions the
package allows (NumPy 1.24.0, SciPy 1.10.0, PyTorch 2.0.0). Most
checks compare the package with something independent of it: a
closed-form result, a second calculation done another way, or the
known truth of a simulated data set. Tests that use random data use
fixed seeds. The main checks:

**Models and simulator**

- The ideal g2 is 0 at zero delay and 1 at very long delay, both to
  1e-12. `g2_zero` equals `g2_measured` at zero delay without jitter.
- The closed-form instrument-response blur matches a brute-force
  numerical convolution to 1e-4.
- The exact three-level rate model gives the two-exponential formula,
  with the `tau1`, `tau2`, `a` computed from its rates, to 1e-10, on
  at least 25 of 30 random parameter sets (those whose g2 does not
  oscillate; mathematically, whose eigenvalues are real). Its steady
  state (the long-run share of time in each level) leaves the rate
  equations unchanged to 1e-12.
- The simulator's edge bin (-60 ns) sits within 2 % of the accidental
  level and its central bin below 10 % of it; its Poisson draws have the right
  mean (every bin within 5 standard errors over 400 draws) and a
  median variance-to-mean ratio within 0.15 of 1 (Poisson counts have
  a ratio of 1); time slices add up to the same mean.

**Analysis of histograms**

- `analyze_histogram` on simulated one-emitter data with a 7 ns
  offset: dip found within 2 ns, g2(0) within 0.1 of the truth, whole
  interval below 0.5. For two emitters: within 0.12 of the truth.
- The fit on a noise-free histogram lands within 0.05 of the true
  g2(0) for one and three emitters (this test needs PyTorch).
- A fast emitter (`tau1` = 0.15 ns, `tau2` = 5 ns): with its own
  `t1_bounds`/`t2_bounds` the fit is within 0.05 of the truth; with the
  defaults it is off by more than 0.2.
- Where the instrument response matters (the blurred dip is more than
  0.02 above the true g2(0)), the profile interval still contains the
  true g2(0).
- Profile-likelihood intervals with `c0_prior` contain the truth at
  10 s and 60 s, and the 60 s interval is less than half as wide.
  Without the prior the interval still contains the truth and is
  wider. They also separate a single emitter (upper end below 0.5)
  from a pair (lower end above 0.4, containing the pair's truth).
- Pulsed data: `analyze_pulsed` within 0.05 of the true 0.12; the
  peak-area method within 0.03 on noise-free data for 0.05, 0.12 and
  0.45; the comb centre within 2 bins; the peak shape has unit area to
  1e-3.
- Data coarser than the analysis grid is refused (new in 0.9.1).

**Bayesian posterior and planning**

- The hand-built posterior matches SciPy's independent beta-prime
  distribution: density to a relative 1e-12, distribution function to
  1e-12, quantiles to a relative 1e-9.
- The density integrates to 1 and gives the closed-form mean, both to
  1e-8 by numerical integration; it agrees with a direct numerical
  integration of the Poisson likelihood over the unknown flat rate
  (an integral the module never uses) to 1e-9 plus ten times the
  integration routine's own error estimate; and with 200 000 random Gamma-ratio draws to 5e-3.
- 95 % intervals contain the true window ratio in 90 % to 100 % of 200
  simulated histograms. A single emitter gets P[g2 < 0.5] above 0.9, a
  pair below 0.5.
- The window g2 is the same at 1 s and 1000 s to a relative 1e-12.
  The planner equals `bayesian_g2` on `expected_histogram` (probability
  to 1e-15, interval to 1e-12). The planned time reaches the
  confidence and 0.8 of it does not. At 4 times the planned time, more
  than 90 % of 200 simulated runs certify. The four-emitter site and an
  unreachable confidence are refused.

**Sequential test**

- The evidence is the same whether the data arrive in one piece or
  two (relative 1e-12).
- Over 80 simulated single-emitter runs and 80 pair runs, with nominal
  error rates of 0.05, each observed error rate is at most 0.125 and
  every run decides within 120 s.
- The mean decision times (40 runs each) lie between Wald's prediction
  and twice it, and below 2 s.

**Corrections, heralded sources, time tags, files**

- The background correction inverts the simulator's own background
  model to 1e-12 (20 random cases) and maps interval ends to 1e-12;
  values below 0 are cut to 0.
- The dead-time correction reproduces the true rate to a relative
  1e-12; the forward formula matches the photon-by-photon detector
  simulation within 2 %.
- Both heralded formulas satisfy their defining quadratic equations to
  1e-9 times CAR; CAR = 1 gives exactly 1 (laser-pumped) and 1.5
  (thermal); at large CAR they approach 2/CAR and 4/CAR; the
  laser-pumped value falls steadily with CAR; the inversions return
  g2 to a relative 1e-12 and CAR to a relative 1e-9; thermal
  statistics need the higher CAR at each of the three targets tested.
- `correlate` equals a brute-force count of every pair, exactly. On
  sparse data `correlate_start_stop` equals it exactly; on dense data
  it counts fewer pairs in total and no more in any bin, with at most
  one per start.
- Two independent random click streams normalize to g2 = 1: the mean
  within 4 standard errors, reduced chi-square between 0.6 and 1.5.
  Simulated single-emitter tags give a zero-delay bin below 0.7 of the
  far level, with the far level within 0.2 of 1.
- Histogram and time-tag files round-trip exactly. Wrong headers,
  wrong field counts, negative counts, a non-increasing delay axis,
  unknown channels and one-column tag files are refused.

**Other**

- Machine-learning parts: output shapes of the networks, the
  environment and the baselines; finite gradients through the
  differentiable simulator; the replay buffer's bookkeeping and
  sampling in proportion to priority (within 0.05). These are contract
  checks, not checks of how well anything learns.
- The version number matches the installed metadata and CITATION.cff;
  the core modules `sparq.physics`, `sparq.exact` and `sparq.pulsed`
  import with PyTorch blocked.

Not covered by tests: the graph encoder (`sparq.gnn`), the training
loops, the agent's learning update, `fisher_info_g2zero`, and
`load_fisequr` beyond refusing a missing directory.

## Corrections in earlier versions

**0.9.1 (this release) fixed three silent failures.**

- `correlate` used a binary search on the stop tags without checking
  that they were sorted. With unsorted `t_b` it returned a histogram
  with most pairs missing (on the test's data, 9 pairs instead of 376)
  and no warning. It now refuses unsorted `t_b`; the order of `t_a` never
  mattered.
- `analyze_histogram` (through `rebin_real`) accepted data binned more
  coarsely than its 1 ns analysis grid, left some grid bins empty, and
  returned a wrong g2(0) with `ok=True` (in one check, about 0.30 for
  a noise-free histogram with 2 ns bins, whose true g2(0) is 0.0975). This is now refused; pass a coarser `cfg`.
- `load_timetags_csv` read a one-column file with exactly two rows as
  a single tag. It now refuses it like other files without two
  columns.

It also fixed a test that used a NumPy 2.0 function, although the
package allows NumPy 1.24, and added a CI job with the oldest allowed
versions.

**0.6.0 changed the fit model.** Before 0.6.0 the fit and the profile
likelihood used the form `1 - d e1 + a e2` with `d` capped at 1. That
biased g2(0) upward for any emitter with a strong shoulder. Both now
use the physical form (the dip depth multiplies both exponentials),
and both include the instrument response, so they estimate the
jitter-free g2(0). `load_fisequr` stopped defaulting to one machine's
data path.

The full history is in [CHANGELOG.md](CHANGELOG.md).

## Limits

- The analysis assumes the three-level model above. A dip of another
  shape is outside what the fits describe.
- The default `tau1`/`tau2` search ranges suit NV-centre-scale
  emitters. Set them for your emitter; a range that excludes the true
  value gives a wrong answer without an error.
- Re-binning puts each input bin whole into one grid bin. If the grid
  bin width is not a whole multiple of the input bin width, grid bins
  receive unequal numbers of input bins and the histogram gets a
  ripple. Choose `cfg` so the grid bins are a whole multiple of your
  bins.
- `bayesian_g2` estimates the raw window-averaged g2, an upper bound
  on g2(0), not g2(0) itself; use `profile_likelihood_ci` for the
  latter.
- The planner works on the average histogram, so its times are
  typical, not guaranteed.
- The sequential test's error rates are exact only if the two
  hypotheses are known exactly.
- The heralded-source formulas are lower limits from pair statistics
  alone.
- The fast simulator treats blinking as a flat raised level within the
  ±60 ns window and has no detector dead time; the photon-by-photon
  simulator has both.
- No vendor binary time-tag formats are read; export to the two-column
  CSV format.
- The machine-learning layer is tested only for shapes and interfaces.

## Where it comes from

The package is the installable part of the manuscript *"Closed-loop,
event-driven machine learning for autonomous triage of single-photon
emitters"*. The
[companion repository](https://github.com/Tanvir-Mahmud-Mahim/a-spiking-RL-triage-of-solid-state-single-photon-emitters)
reproduces the paper itself.

The experimental quantum-dot HBT measurements used by
`sparq.datasets.load_fisequr` are from the openly licensed
[sps-quality](https://github.com/UTS-CASLab/sps-quality) repository
(Kedziora et al., *Mach. Learn.: Sci. Technol.* **4**, 045042
(2023)); they are not redistributed here, and the loader takes the
dataset directory explicitly.

## Citing, support and license

Please cite the associated paper if you use this code; citation
metadata is in [CITATION.cff](CITATION.cff). Every release is archived
on Zenodo under the concept DOI
[10.5281/zenodo.22278040](https://doi.org/10.5281/zenodo.22278040),
which always resolves to the latest version.

The package is maintained by Tanvir Mahmud Mahim (BRAC University),
who reviews issues and pull requests. Bug reports, questions and pull
requests are welcome through
[GitHub issues](https://github.com/TaN-MM-Org/sparq-triage/issues);
see [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and
the design rules. Tagged releases are published to PyPI by CI.

Licensed under Apache-2.0 (see [LICENSE](LICENSE)).
