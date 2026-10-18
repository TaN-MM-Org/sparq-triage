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

The package works from that record (a histogram, the raw detector
time stamps, or a PicoQuant PTU file) and answers:

- How deep is the dip, and how sure can I be about it?
- Does the usual dip formula describe my data at all, and if not,
  which shape does?
- Is the dip deep enough to call the site a single emitter, and with
  what probability?
- How long must I measure before the answer is reliable, for a typical
  run or for 9 runs out of 10, and can a longer run ever make it
  reliable?
- Can I stop measuring early because the data already suffice, with
  error rates that hold without knowing the emitter?
- Did the source stay the same during the run, so that the snapshots
  may be added up?
- How much do background light, blinking, detector dead time and
  afterpulsing change the numbers, and how sure is the background
  correction when the background itself is measured?
- For a pulsed laser, what is g2(0) when the peaks of the comb
  overlap?
- For a heralded pair source, what purity does a measured
  coincidence-to-accidental ratio allow, with my detectors?
- How bright does the emitter get, and at what excitation power does it
  saturate?
- How long does the emitter stay excited (its lifetime), from the
  photons' delays after the pulses of a pulsed laser?

The analysis core needs only NumPy and SciPy. An optional
machine-learning layer (PyTorch) holds the neural estimators and the
automated triage agent of the paper *"Closed-Loop, Event-Driven
Machine Learning for Autonomous Triage of Single-Photon Emitters"*
(T. M. Mahim, M. N. Islam, M. M. Rahman and A. S. M. Mohsin),
*ACS Photonics* (2026),
[doi:10.1021/acsphotonics.6c01984](https://doi.org/10.1021/acsphotonics.6c01984);
the
[companion repository](https://github.com/Tanvir-Mahmud-Mahim/a-spiking-RL-triage-of-solid-state-single-photon-emitters)
reproduces the paper itself.

## Contents

- [A short guide to the words used here](#a-short-guide-to-the-words-used-here)
- [Install, requirements and units](#install-requirements-and-units)
- [Examples](#examples) (each with the output it prints)
- [Tutorials](#tutorials) (three worked scripts, from data to a result)
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
  (coincidence-to-accidental ratio) is its usual quality number: the
  coincidences between the two detectors in the same time window
  divided by those between different windows (the accidentals). Some
  papers subtract the accidentals first ("net" CAR, one less).
- **Bin average** -- a correlator counts every pair whose delay falls
  anywhere in a bin, so a bin's expected count is g2 averaged over the
  bin, not g2 at the bin's center. The package uses the average.
- **Goodness of fit, p-value** -- the chance that data really drawn
  from the fitted model would fit at least this badly. A p-value near
  0 means the model does not describe the data.
- **PTU file, T2 and T3 mode** -- PicoQuant's time-tag file format. In
  T2 mode every click has an absolute time; in T3 mode it has a laser
  pulse number and a delay after that pulse.
- **Dead time** -- after a click a detector is blind for a while (tens
  of ns for common single-photon detectors).
- **Afterpulse** -- a false click that a detector sometimes gives some
  time after a real one. `afterpulse_p` is the chance of one per click.
- **Pulsed excitation, comb** -- with a pulsed laser the emitter gives
  at most about one photon per pulse, so the histogram is a row of
  peaks, one per laser period (`t_rep`, 12.5 ns at 80 MHz). g2(0) is
  the area of the peak at zero delay divided by that of the other
  (side) peaks.
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

- delays, lifetimes (`tau1`, `tau2`, `tau_e`), bin widths, IRF widths,
  dead times, afterpulse delays and laser periods (`t_rep`):
  nanoseconds (ns);
- acquisition times (`T_s`): seconds;
- count rates: `rate_kcps` in an `EmitterSite` is kilocounts per
  second, summed over both detectors; `r_hat` and `rate_cps` in
  results are counts per second, also summed over both detectors;
  `deadtime_corrected_rate` takes one detector's rate in counts per
  second;
- g2, `a`, `rho`, CAR: no unit.


## Examples

Each example below runs as written, and the output shown is what it
printed with sparq-triage 0.14.0 (Python 3.13, NumPy 2.5.3, SciPy
1.18.1). All emitter numbers are illustrative values, not
measurements. The data are simulated with a fixed random seed, so you
get the same output. With other NumPy or SciPy versions some printed
numbers can differ a little: in example 11, whose sampler follows the
fit it starts from, and in the p-value of the rejected fit of example
10 (2.2e-10 with SciPy 1.10), whose optimizer stops at a slightly
different point.

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
grid = HBTConfig(tau_max=90.25, n_bins=361)
rng = np.random.default_rng(0)
counts = rng.poisson(expected_histogram(site, 30.0, grid))

path = os.path.join(tempfile.mkdtemp(), "my_hbt.csv")
save_hbt_csv(path, grid.bin_centers + 7.0, counts)   # header: delay_ns,counts
delay, counts = load_hbt_csv(path)

# the detectors' count rates, as read from the counters (counts/s)
res = analyze_histogram(delay, counts, T_s=30.0, singles_cps=(75e3, 75e3),
                        n_bootstrap=30, seed=1)
print(f"dip found at {res['center']:.2f} ns")
print(f"g2(0) = {res['g2_0']:.3f}, 68 % interval "
      f"[{res['g2_0_low']:.3f}, {res['g2_0_high']:.3f}]")
print("below 0.5:", res["single_emitter"],
      "| whole interval below 0.5:", res["single_emitter_confident"])
print(f"fit p-value {res['fit_p_value']:.2f}; limits that matter: "
      f"{res['bound_matters']}")
```

```
true g2(0) of this site: 0.0975
dip found at 6.90 ns
g2(0) = 0.096, 68 % interval [0.075, 0.114]
below 0.5: True | whole interval below 0.5: True
fit p-value 0.69; limits that matter: False
```

`analyze_histogram` finds the dip, re-bins the data onto the 1 ns
analysis grid centered on it, and fits the emitter model with the
instrument response included. `singles_cps` are the two detectors'
count rates as the counters show them. They fix the flat level
(the height the histogram would have if the clicks were unrelated),
and without them the fit is much less certain (see below). The
interval comes from a **bootstrap**: the counts are redrawn at random
(Poisson) `n_bootstrap` times and the whole analysis is repeated on
each copy. A real analysis should use more redraws than the 30 used
here to keep the example fast (the default is 200).
`single_emitter_confident` is True when the whole interval lies on one
side of 0.5. The **fit p-value** says whether the model describes the
data (a value near 0 means it does not; see example 10; this one is
approximate and errs on the large side), and
**limits that matter** says whether the answer depends on where the
fit's search ranges end (example 9). Your own file needs the header
`delay_ns,counts` and one row per bin. For a pulsed-laser measurement
(a comb of peaks), use `analyze_pulsed` instead (example 17).

Why the singles rates matter. With the flat level left to the fit, a
stronger, slower shoulder can trade against a lower flat level, and a
histogram of ±60 ns cannot tell the two apart. Over 15 simulated runs
of this site (in the tests), the histogram-only fit missed the true
g2(0) by up to 0.16 (and a pair's by up to 0.24); with the singles
rates every run was within 0.06, with an rms error below 0.03. Most of
the large histogram-only misses are flagged by a warning (the fit then
sits on one of its search limits, and moving the limit moves g2(0)),
but not all of them are, so pass the singles rates whenever you have
them.

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
true g2(0): a cautious number. Example 11 gives the probability for
g2(0) itself.

### 3. A model-based interval, and why the singles rate helps

```python
import warnings
import numpy as np
from sparq import EmitterSite, HBTConfig, expected_histogram, profile_likelihood_ci

site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                        rho=0.95, blinking=False), n_emitters=1)
cfg = HBTConfig()
hist = np.random.default_rng(0).poisson(
    expected_histogram(site, 60.0, cfg)).astype(float)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)   # see the text below
    free = profile_likelihood_ci(hist, 60.0, 150e3, cfg)
    pinned = profile_likelihood_ci(hist, 60.0, 150e3, cfg, c0_prior=(1.0, 0.01))
    wider = profile_likelihood_ci(hist, 60.0, 150e3, cfg, c0_prior=(1.0, 0.01),
                                  t2_bounds=(50.0, 8000.0))
print(f"true g2(0): {site.g2_0:.4f}")
print(f"flat level free:        [{free['lo']:.3f}, {free['hi']:.3f}]")
print(f"flat level pinned (1%): [{pinned['lo']:.3f}, {pinned['hi']:.3f}]")
print(f"on a search limit: free {free['at_bound']}, pinned {pinned['at_bound']}")
print(f"pinned, tau2 up to 8000 ns: [{wider['lo']:.3f}, {wider['hi']:.3f}]")
```

```
true g2(0): 0.0975
flat level free:        [0.064, 0.295]
flat level pinned (1%): [0.078, 0.124]
on a search limit: free [], pinned ['tau2_high']
pinned, tau2 up to 8000 ns: [0.077, 0.123]
```

`profile_likelihood_ci` gives a 95 % interval for g2(0) itself, using
the full emitter model and the exact Poisson likelihood. With the
flat level left free, the interval is wide for the reason given under
example 1. Pass the singles rate as `r_hat` (here 150e3 counts per
second, both detectors) and its relative uncertainty as
`c0_prior=(1.0, sd)`, and the interval narrows. With the flat level
pinned, the best fit puts the shelving time at the upper end of its
search range (800 ns), and the function warns about it (silenced here).
Searching up to 8000 ns moves the interval by 0.001, so here the limit
does not matter; when the interval does move, report the wider one.
Version 0.12.0 corrects this function: up to 0.11.0 it printed
[0.067, 0.182] and [0.084, 0.126] for these data (see
[Corrections](#corrections-in-earlier-versions)).

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
typical-data time: 1.42 s
P[g2 < 0.5] at that time: 0.9500
window g2 the data settle on: 0.087
refused: the site's window-averaged g2 is 0.739 >= the threshold 0.5: the window ratio does not change with acquisition time, so no run length certifies this site below the threshold
```

The planner applies the Bayesian verdict of example 2 to the
*average* histogram the site would give, and searches for the
shortest time at which it reaches the confidence. Real runs scatter
around that average: only 59 % of runs of that length certify this
site (example 12 shows how to plan for a chosen share of runs). The
refusal is exact, not a timeout: both the zero-delay counts and the
far counts grow in proportion to the measuring time, so their ratio
never changes, and a site whose ratio is not below the threshold can
never be certified.

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
these error rates hold only approximately; the tests measure them on
simulated runs. Example 13 shows a test whose error bounds need no
emitter model at all.

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
solution and is refused. That formula is exact for unrelated
(Poisson) clicks; for antibunched or bunched light the loss is
slightly different (`deadtime_throughput`, example 16). Example 20
shows how to measure rho with an interval and carry it into the
corrected value.

### 7. From raw time tags to g2

```python
import numpy as np
from sparq import (EmitterSite, HBTConfig, simulate_photon_stream, g2_measured,
                   correlate, correlate_start_stop, normalize_g2)

site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                        rho=0.95, blinking=False), n_emitters=1)
cfg = HBTConfig()
rng = np.random.default_rng(2)
t_a, t_b = simulate_photon_stream(site, 5.0, rng)   # 5 s of photon tags (ns)

hist = correlate(t_a, t_b, cfg)                     # every pair counts
out = normalize_g2(hist, len(t_a), len(t_b), 5.0, cfg)
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
374955 + 373926 tags, 28.0 accidental coincidences per bin
g2 at zero delay: 0.04 +- 0.04
g2 far out (mean of 30 edge bins): 1.235
model g2 at the same delays: 1.183
pairs: all-pairs 3276, start-stop 3268
```

`normalize_g2` divides the coincidences by the level two unrelated
detectors would give, `N1 N2 w / T` per bin (N1, N2 the click counts,
w the bin width, T the measuring time), and gives Poisson one-sigma
error bars. The zero-delay bin expects only about 3 counts here, so it
scatters a lot from run to run. The far bins sit above 1 because of
this emitter's shelving shoulder, as the model line shows; they are
not an error. `correlate` counts every pair of clicks (as a software
correlator does). `correlate_start_stop` copies an older kind of
timing card that records only the first stop in each start's window;
at high rates it loses pairs, so fit the `correlate` histogram. Tag
files can be read with `load_timetags_csv` (two columns: channel,
time in ns) or, for PicoQuant hardware, `load_ptu_timetags`
(example 15).

### 8. Heralded pair sources: the low-efficiency formulas

```python
from sparq import heralded_g2_limit, car_for_purity

for car in (10.0, 100.0):
    print(f"CAR {car:5.0f}: g2_h(0) {heralded_g2_limit(car):.4f} "
          f"(laser-pumped), {heralded_g2_limit(car, 'thermal'):.4f} (thermal)")
print(f"CAR for g2_h(0) = 0.01: {car_for_purity(0.01):.1f} (laser-pumped), "
      f"{car_for_purity(0.01, 'thermal'):.1f} (thermal)")
```

```
CAR    10: g2_h(0) 0.1900 (laser-pumped), 0.3800 (thermal)
CAR   100: g2_h(0) 0.0199 (laser-pumped), 0.0398 (thermal)
CAR for g2_h(0) = 0.01: 199.5 (laser-pumped), 399.5 (thermal)
```

When every detector sees only a small share of the photons, a pair
source whose pair number follows K independent thermal modes gives

    CAR = 1 + 1/K + 1/mu,      g2_h(0) = (1 + 1/K)(2 CAR - 1)/CAR^2,

with CAR the raw ratio C/A of coincidences to accidentals and mu the
mean number of pairs per window. "Laser-pumped" is the many-mode
(Poissonian) case K -> infinity, `(2 CAR - 1)/CAR^2` (H. Wang et al.,
arXiv:2404.03236); "thermal" is K = 1, `(4 CAR - 2)/CAR^2`, possible
only for CAR >= 2. `car_for_purity` inverts them, and `modes=K` covers
the cases in between. These are not lower limits: with an efficient
herald detector the true g2_h(0) is lower, and dark counts can make it
higher (example 14 computes it for your detectors).

### 9. When the answer depends on where the search stops

```python
import warnings
import numpy as np
from sparq import EmitterSite, HBTConfig, expected_histogram, analyze_histogram

# A fast emitter (quantum-dot-like, illustrative): 0.15 ns antibunching
# time, 5 ns shelving time -- far below the default search ranges.
site = EmitterSite(dict(tau1=0.15, tau2=5.0, a=0.4, rho=0.97,
                        rate_kcps=400.0, blinking=False), n_emitters=1)
cfg = HBTConfig(tau_max=10.0125, n_bins=801, sigma_irf=0.005)  # 25 ps bins
counts = np.random.default_rng(0).poisson(expected_histogram(site, 300.0, cfg))

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    res = analyze_histogram(cfg.bin_centers, counts, T_s=300.0, cfg=cfg,
                            n_bootstrap=0)
print(f"default ranges: g2(0) = {res['g2_0']:.3f}, on a limit: {res['at_bound']}")
print("warning issued:", len(caught) == 1)

res = analyze_histogram(cfg.bin_centers, counts, T_s=300.0, cfg=cfg,
                        n_bootstrap=0, t1_bounds="auto", t2_bounds="auto")
print(f"'auto' ranges:  g2(0) = {res['g2_0']:.3f} (true {site.g2_0:.4f}), "
      f"tau1 {res['fit']['tau1']:.3f} ns within {res['t1_bounds']}")
```

```
default ranges: g2(0) = 0.562, on a limit: ['tau1_low', 'tau2_low', 'a_high']
warning issued: True
'auto' ranges:  g2(0) = 0.045 (true 0.0591), tau1 0.142 ns within (0.03, 80.0)
```

The fit searches `tau1` and `tau2` within ranges (by default 0.3 to
80 ns and 50 to 800 ns, NV-centre scales). This emitter is far
faster, so the default fit ends on the edges of those ranges, reports
which ones (`at_bound`), refits with them ten times wider, and, because
that changes g2(0), warns. With `"auto"` the ranges are widened until
the fit no longer sits on an edge (at most eight tenfold steps, within
1e-3 ns to 10 times the window for `tau1` and 100 times for `tau2`).
Your own ranges, set from what you know about the emitter, work too.

### 10. A dip of another shape

```python
import numpy as np
from sparq import HBTConfig, g2_model, fit_model

# A resonantly driven emitter (illustrative): Rabi frequency 3/ns,
# decay rate 1.4/ns, dephasing 0.1/ns. Its dip rings instead of
# rising smoothly, which the three-level formula cannot describe.
cfg = HBTConfig(tau_max=10.05, n_bins=201, sigma_irf=0.05)   # 0.1 ns bins
truth = dict(omega=3.0, gamma=1.4, gamma_deph=0.1, delta=0.0)
nodes, weights = cfg.bin_nodes()                  # bin-averaged model
T, rate = 120.0, 2e5                              # s, counts/s (both detectors)
flat = (rate / 2) ** 2 * cfg.bin_width * 1e-9 * T
mean = flat * (g2_model(nodes, "coherent", truth, cfg.sigma_irf, rho2=0.9)
               @ weights)                         # true g2(0) = 1 - 0.9
counts = np.random.default_rng(0).poisson(mean)

for model, fixed in (("three_level", None), ("coherent", dict(delta=0.0))):
    fit = fit_model(counts, T, rate, cfg, model, fixed=fixed,
                    c0_prior=(1.0, 0.01))
    print(f"{model:12s} g2(0) = {fit['g2_0']:.3f}, p-value {fit['p_value']:.1e}")
```

```
three_level  g2(0) = 0.000, p-value 8.3e-11
coherent     g2(0) = 0.110, p-value 4.0e-01
```

`fit_model` fits a dip model by the exact Poisson likelihood and
reports a goodness-of-fit **p-value**: the chance that data from the
fitted model would fit at least this badly. Here the three-level
formula is rejected (p about 1e-10) and reads g2(0) as 0, while the
model of a coherently driven emitter fits and recovers g2(0) = 0.1
within its error. The models are the three-level formula, a
`"multi_exponential"` form with several shoulders (several dark
states), and `"coherent"`, a two-level emitter driven by a laser on or
near resonance, computed from its exact master equation. `g2_model`
evaluates any of them, with the jitter blur and background. The
p-value assumes Poisson counts: fit your native bins, or bins merged
whole (`rebin_real(..., method="whole")`) at a whole multiple.

### 11. The probability for g2(0) itself

```python
import numpy as np
from sparq import EmitterSite, HBTConfig, expected_histogram, bayesian_g2_model

site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                        rho=0.95, blinking=False), n_emitters=1)
cfg = HBTConfig()
hist = np.random.default_rng(0).poisson(expected_histogram(site, 30.0, cfg))

post = bayesian_g2_model(hist, 30.0, 150e3, cfg, c0_prior=(1.0, 0.01), seed=0)
lo, hi = post.credible_interval(0.95)
print(f"true g2(0) {site.g2_0:.4f}; posterior median {post.median():.3f}, "
      f"95 % interval [{lo:.3f}, {hi:.3f}]")
print(f"P[g2(0) < 0.5] = {post.prob_below(0.5):.3f}; "
      f"R-hat {post.rhat:.3f}, effective samples {post.ess:.0f}")
```

```
true g2(0) 0.0975; posterior median 0.101, 95 % interval [0.069, 0.135]
P[g2(0) < 0.5] = 1.000; R-hat 1.017, effective samples 707
```

`bayesian_g2_model` gives the posterior of the model's g2(0) (without
detector jitter), where `bayesian_g2` (example 2) only bounds it from
above. It has no closed form, so it is sampled (an ensemble sampler);
R-hat near 1 and a few hundred effective samples or more mean the
sampling has settled. The priors are stated in `help(sparq.posterior)`:
g2(0) uniform on [0, 1], the lifetimes uniform in their logarithm
within the search ranges, the shoulder amplitude uniform up to 3, and
the flat-level scale from `c0_prior` (integrated out exactly). Without
`c0_prior` the interval is much wider, for the reason given under
example 1.

### 12. Plan for a chosen share of runs

```python
from sparq import (site_from_numbers, required_acquisition_time,
                   certification_probability, assured_acquisition_time)

site = site_from_numbers(tau1_ns=12.0, tau2_ns=200.0, a=0.3,
                         rate_kcps=120.0, rho=0.97)        # as in example 4
T_typ, _ = required_acquisition_time(site, confidence=0.95)
print(f"typical-data time {T_typ:.2f} s: a run that long certifies with "
      f"probability {certification_probability(site, T_typ):.2f}")
T, p, p_min = assured_acquisition_time(site, assurance=0.9, confidence=0.95)
print(f"time for 90 % of runs to certify: {T:.2f} s (probability {p:.3f}, "
      f"lowest after that {p_min:.3f})")
```

```
typical-data time 1.42 s: a run that long certifies with probability 0.59
time for 90 % of runs to certify: 3.16 s (probability 0.900, lowest after that 0.900)
```

`certification_probability` is the exact probability, over the
Poisson scatter of real runs, that a run of a given length certifies
the site (the verdict of example 2 reaching the confidence). It sums
over the two counts that verdict uses, with no simulation.
`assured_acquisition_time` finds the time after which that
probability stays at or above the assurance you ask for. The
probability rises with time in small steps (counts are whole numbers),
so it is scanned in 0.2 % steps to find where it stays up.

### 13. A sequential test that needs no emitter model

```python
import numpy as np
from sparq import EmitterSite, HBTConfig, expected_histogram, WindowSPRT

cfg = HBTConfig()
test = WindowSPRT(g_accept=0.25, g_reject=0.5, cfg=cfg, alpha=0.05, beta=0.05)
print("error bounds (wrong accept, wrong reject): "
      "%.4f, %.4f" % test.error_bounds)

site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                        rho=0.95, blinking=False), n_emitters=1)
rng = np.random.default_rng(1)
steps = 0
while test.decision == "continue":
    brightness = rng.uniform(0.5, 1.5)            # the rate may wander
    test.update(rng.poisson(brightness * expected_histogram(site, 0.05, cfg)))
    steps += 1
print(f"decision: {test.decision} after {steps} increments "
      f"({test.k0:.0f} central and {test.kr:.0f} reference counts)")
```

```
error bounds (wrong accept, wrong reject): 0.0526, 0.0526
decision: accept after 61 increments (3 central and 853 reference counts)
```

`WindowSPRT` uses only the counts in the zero-delay window and in the
far reference window. Given how many counts fell in the two windows
together, the share in the zero-delay window depends only on the
window-averaged g2, not on the count rate or on anything about the
emitter. So its error bounds hold for any rate (even one that changes
between increments, as here) and for every g2 in each range: at most
0.0526 for wrongly accepting any site with window g2 of 0.5 or more,
and the same for wrongly rejecting any site at 0.25 or less (Wald's
inequalities, alpha/(1 - beta) and beta/(1 - alpha)). Between 0.25
and 0.5 either answer can come out. It tests the window-averaged g2 of
example 2, an upper bound on g2(0).

### 14. Heralded sources with real detectors

```python
import numpy as np
from sparq import heralded_source, heralded_from_car, heralded_g2_limit

# Illustrative laser-pumped source: 0.05 pairs per window, 5 % signal
# and 95 % idler (herald) detection efficiency, no dark counts.
src = heralded_source(mu=0.05, modes=np.inf, eta_s=0.05, eta_i=0.95)
print(f"CAR {src['car']:.2f}: exact g2_h(0) {src['g2_h']:.4f}, "
      f"low-efficiency formula {heralded_g2_limit(src['car']):.4f}")

# back from a measured CAR, with the measured efficiencies and dark counts
for s in heralded_from_car(20.0, np.inf, eta_s=0.05, eta_i=0.05,
                           dark_s=1e-4, dark_i=1e-4):
    print(f"CAR 20 with dark counts: mu {s['mu']:.2e}, g2_h(0) "
          f"{s['g2_h']:.4f}, herald probability per window {s['p_herald']:.1e}")
```

```
CAR 20.53: exact g2_h(0) 0.0510, low-efficiency formula 0.0951
CAR 20 with dark counts: mu 8.24e-05, g2_h(0) 0.1782, herald probability per window 1.0e-04
CAR 20 with dark counts: mu 4.84e-02, g2_h(0) 0.0987, herald probability per window 2.5e-03
```

`heralded_source` computes the CAR and g2_h(0) exactly for
click/no-click detectors with given efficiencies and dark-count
probabilities per window. With an efficient herald detector a window
with two pairs heralds only once, so g2_h(0) is about half the
low-efficiency formula at the same CAR. `heralded_from_car` goes back
from a measured CAR. With dark counts the same CAR can come from two
pair rates (at very low rates the accidentals are mostly dark counts);
the measured herald probability per window says which one applies.

### 15. PicoQuant PTU files

```python
import os, tempfile
import numpy as np
from sparq import (EmitterSite, HBTConfig, simulate_photon_stream, correlate,
                   save_ptu_t2, read_ptu, load_ptu_timetags)

site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                        rho=0.95, blinking=False), n_emitters=1)
t_a, t_b = simulate_photon_stream(site, 1.0, np.random.default_rng(3))

# write the tags as a HydraHarp V2 T2 file (inputs 1 and 2), 1 ps ticks
ch = np.r_[np.ones(t_a.size, int), np.full(t_b.size, 2)]
t = np.r_[t_a, t_b]
order = np.argsort(t, kind="stable")
path = os.path.join(tempfile.mkdtemp(), "hbt.ptu")
save_ptu_t2(path, ch[order], t[order], 1e-12, "HydraHarp2T2")

info = read_ptu(path)
print(info["record_type"], info["mode"], f"{info['channel'].size} events,",
      f"resolution {info['global_resolution_s'] * 1e12:.0f} ps")
a, b = load_ptu_timetags(path, channel_a=1, channel_b=2)
same = np.array_equal(correlate(a, b, HBTConfig()), correlate(t_a, t_b, HBTConfig()))
print("histogram from the file equals the original:", same)
```

```
HydraHarp2T2 T2 150002 events, resolution 1 ps
histogram from the file equals the original: True
```

`read_ptu` reads PicoQuant's PTU time-tag files in all twelve record
formats of PicoQuant's file documentation (PicoHarp 300, HydraHarp,
TimeHarp 260, MultiHarp and PicoHarp 330, in T2 and T3 mode), and
`load_ptu_timetags` gives two sorted channels ready for `correlate`.
Channels are numbered as PicoQuant's own demo reader prints them: in
HydraHarp-type T2 files the sync input is 0 and input k is k + 1.
`save_ptu_t2` writes T2 files (used here and in the tests). No file
recorded on real hardware was available for testing; see
[How the results are checked](#how-the-results-are-checked).

### 16. Blinking and dead time in the fast simulator

```python
import numpy as np
from sparq import EmitterSite, HBTConfig, expected_histogram, site_g2

# Illustrative blinking emitter: on for 0.3 us, off for 0.15 us on average
blinky = EmitterSite(dict(tau1=8.0, tau2=100.0, a=1.5, rate_kcps=600.0,
                          rho=0.9, blinking=True, t_on_ms=3e-4,
                          t_off_ms=1.5e-4), n_emitters=1)
print(f"g2 at 0, 10, 60 ns: " + ", ".join(
    f"{g:.3f}" for g in site_g2(blinky, np.array([0.0, 10.0, 60.0]))))

steady = EmitterSite(dict(tau1=8.0, tau2=100.0, a=1.5, rate_kcps=600.0,
                          rho=0.9, blinking=False), n_emitters=1)
cfg = HBTConfig()
far = expected_histogram(steady, 10.0, cfg)[0]
far_dt = expected_histogram(steady, 10.0, cfg, dead_time_ns=45.0)[0]
print(f"45 ns dead time lowers the flat level by {100 * (1 - far_dt / far):.2f} %")
```

```
g2 at 0, 10, 60 ns: 0.265, 2.016, 1.971
45 ns dead time lowers the flat level by 4.23 %
```

A blinking emitter is switched on and off at random; the switching
adds bunching around the dip (g2 of 2.0 at 10 ns here) but cannot
fill the dip itself (0.265 at zero delay, the value this signal share
gives without blinking). The fast simulator now computes this exactly
for exponentially distributed on and off times, the same process the
photon-by-photon simulator draws. `dead_time_ns` lowers each
detector's counted rate, and so the flat level, as a detector that is
blind for 45 ns after each click would.

### 17. Pulsed laser: fit the whole comb

```python
import numpy as np
from sparq import HBTConfig, expected_hist_pulsed, analyze_pulsed

# Illustrative pulsed measurement: 80 MHz laser (12.5 ns period),
# 2 ns emitter lifetime, true g2(0) = 0.12, 10 s at 200 kcps,
# 0.25 ns bins over +-60.5 ns (a comb of ten peaks).
cfg = HBTConfig(tau_max=60.5, n_bins=484)
mean = expected_hist_pulsed(cfg, 10.0, 2e5, g2_0=0.12, tau_e=2.0,
                            sigma_pair=0.5, center_off=1.7)
counts = np.random.default_rng(0).poisson(mean)

old = analyze_pulsed(cfg.bin_centers, counts, method="area", n_bootstrap=0)
new = analyze_pulsed(cfg.bin_centers, counts, ci=0.95)
print(f"peak areas (0.10.0 default): g2(0) = {old['g2_0']:.3f}")
print(f"comb fit (new default):      g2(0) = {new['g2_0']:.3f}, 95 % interval "
      f"[{new['g2_0_low']:.3f}, {new['g2_0_high']:.3f}]")
print(f"fitted lifetime {new['fit']['tau_e']:.2f} ns, jitter "
      f"{new['fit']['sigma_pair']:.2f} ns, fit p-value {new['fit_p_value']:.2f}")
```

```
peak areas (0.10.0 default): g2(0) = 0.000
comb fit (new default):      g2(0) = 0.115, 95 % interval [0.091, 0.138]
fitted lifetime 1.97 ns, jitter 0.52 ns, fit p-value 0.34
```

The usual pulsed analysis adds up the counts around each peak and
subtracts a background level measured between the peaks. Here the
peaks are wide enough (2 ns lifetime, 12.5 ns apart) that their tails
fill the space between them, so that "background" is mostly peak tails,
and subtracting it removes the whole zero-delay peak: the answer 0 is
wrong. `analyze_pulsed` now fits the whole comb instead
(`fit_pulsed`): every peak has the same shape (a two-sided
exponential decay blurred by the detectors' jitter), the zero-delay
peak has g2(0) times the area of the others, and a flat background may
lie under all of them. The share of each peak that falls in each bin is
computed exactly, so overlapping peaks are handled, and the counts are
fitted by their Poisson likelihood. The interval is a profile
likelihood interval (as in example 3). `method="area"` still gives the
old peak-area answer. If the emitter blinks, the peaks next to zero
delay are higher than those far away; `envelope=True` fits that, and
then needs the detectors' count rates (`T_s`, `singles_cps`) to know
the height of the far peaks, unless the record reaches them.

### 18. Afterpulsing in the fast simulator

```python
from sparq import EmitterSite, HBTConfig, expected_histogram, deadtime_throughput

site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                        rho=0.95, blinking=False), n_emitters=1)
cfg = HBTConfig()
c = cfg.n_bins // 2                                # the zero-delay bin
for p in (0.0, 0.01, 0.05):                        # afterpulse probability
    hist = expected_histogram(site, 30.0, cfg, dead_time_ns=45.0,
                              afterpulse_p=p, afterpulse_tau_ns=80.0)
    # each detector counts its clicks plus its afterpulses
    counted = deadtime_throughput(site, 75e3, 45.0) * (1 + p)
    level = counted ** 2 * cfg.bin_width * 1e-9 * 30.0
    print(f"afterpulsing {p:.2f}: zero-delay bin / accidental level "
          f"{hist[c] / level:.4f}")
```

```
afterpulsing 0.00: zero-delay bin / accidental level 0.1321
afterpulsing 0.01: zero-delay bin / accidental level 0.1523
afterpulsing 0.05: zero-delay bin / accidental level 0.2275
```

An afterpulse repeats a detector's click some time later, and that
repeated click pairs with the other detector's clicks as if it were
new light. The result fills the dip, by about twice the afterpulse
probability (0.02 for 1 %), just as background light does, and adds a
shallow copy of the dip near plus and minus the dead time.
`expected_histogram` now includes it with the same detector model as
the photon-by-photon simulator. The zero-delay bin here reads 0.132
without afterpulsing, not the site's 0.0975, because the bin and the
jitter average over the walls of the dip.

### 19. Did the source change during the run?

```python
import numpy as np
from sparq import EmitterSite, HBTConfig, expected_histogram, check_stationarity

params = dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0, rho=0.95,
              blinking=False)                       # illustrative values
one = EmitterSite(dict(params), n_emitters=1)
two = EmitterSite(dict(params), n_emitters=2)       # same brightness, two emitters
cfg = HBTConfig()
t = np.full(8, 20.0)                                # eight snapshots of 20 s
rng = np.random.default_rng(4)
steady = rng.poisson(np.outer(t, expected_histogram(one, 1.0, cfg)))
r = check_stationarity(steady, t, cfg)
print(f"steady:  p-values rate {r['rate_p']:.2f}, window {r['window_p']:.2f}, "
      f"shape {r['shape_p']:.2f}; consistent: {r['consistent']}")

# the same site, but a second emitter takes part from the fifth snapshot on
late = rng.poisson(np.outer(t[4:], expected_histogram(two, 1.0, cfg)))
r = check_stationarity(np.vstack([steady[:4], late]), t, cfg)
print(f"changed: p-values rate {r['rate_p']:.2f}, window {r['window_p']:.1e}, "
      f"shape {r['shape_p']:.1e}; consistent: {r['consistent']}")
print("window g2 of each snapshot:", " ".join(f"{g:.2f}" for g in r["g2_window"]))
```

```
steady:  p-values rate 0.31, window 0.92, shape 0.25; consistent: True
changed: p-values rate 0.03, window 9.7e-31, shape 4.5e-82; consistent: False
window g2 of each snapshot: 0.08 0.09 0.11 0.12 0.57 0.58 0.55 0.46
```

Every analysis here assumes that the source stays the same while it is
measured. Bleaching, slow blinking, a drifting focus or a second
emitter that starts to take part break that, and the sum of such
snapshots describes no single state of the site. `check_stationarity`
tests a series of snapshots (each on the grid `cfg`, centered on the
dip) before you add them up. It makes three chi-square tests that need
no emitter model: are the total counts in proportion to the snapshot
times (**rate**), is the window g2 of example 2 the same in every
snapshot (**window**), and is the whole histogram the same up to a
scale (**shape**, with neighboring bins merged until every expected
count is at least 5)? `consistent` is False when any p-value falls
below 0.01/3, so a steady source is flagged in at most about 1 % of
runs. In the second series the window g2 jumps from about 0.1 to
about 0.55 after four snapshots, and both the window and the shape
tests see it. A test whose expected counts are too small for the
chi-square law is refused (merge snapshots or measure longer). Like
`bayesian_g2`, it needs Poisson counts: native bins, or bins merged
whole.

### 20. Background correction with a measured background

```python
from sparq import background_corrected_g2, signal_fraction_from_counts

# Illustrative counter readings (both detectors together): 10 s on the
# emitter, then 0.1 s on an empty spot beside it for the background.
rho = signal_fraction_from_counts(n_on=1_200_000, t_on=10.0,
                                  n_off=900, t_off=0.1, level=0.975)
print(f"signal fraction {rho['rho']:.4f}, 97.5 % interval "
      f"[{rho['lo']:.4f}, {rho['hi']:.4f}]")

# a measured g2(0) with its 97.5 % interval (illustrative)
g2, ci = 0.21, (0.18, 0.24)
exact = background_corrected_g2(g2, rho["rho"], ci=ci)
both = background_corrected_g2(g2, rho["rho"], ci=ci,
                               rho_ci=(rho["lo"], rho["hi"]))
print(f"corrected g2(0) {both['g2_corrected']:.3f}")
print(f"interval, rho taken as exact:   [{exact['ci'][0]:.3f}, {exact['ci'][1]:.3f}]")
print(f"interval, rho uncertainty too:  [{both['ci'][0]:.3f}, {both['ci'][1]:.3f}]")
```

```
signal fraction 0.9250, 97.5 % interval [0.9192, 0.9305]
corrected g2(0) 0.077
interval, rho taken as exact:   [0.042, 0.112]
interval, rho uncertainty too:  [0.029, 0.122]
```

The signal fraction of example 6 usually comes from two counter
readings: on the emitter (signal plus background) and on an empty spot
beside it (background only). `signal_fraction_from_counts` turns them
into rho with an exact interval: given the total of the two counts,
the background count is binomial, and the Clopper-Pearson interval
(Clopper and Pearson, Biometrika 26, 404 (1934)) for that binomial
gives an interval for rho that covers it at least as often as the
level says. `background_corrected_g2(..., rho_ci=...)` then carries
both uncertainties into the corrected value: it returns the smallest
and largest corrected g2 over every pair of values in the two
intervals. When each interval has level 97.5 %, the result covers the
true g2(0) in at least 95 % of runs (the chance that either interval
misses is at most 2.5 % + 2.5 %). Here the short background reading
widens the corrected interval from 0.070 to 0.093. This assumes that
the background beside the emitter equals the background on it, and
count rates low enough that dead time does not matter (or rates
corrected with `deadtime_corrected_rate` first).

### 21. Saturation curve

```python
import numpy as np
from sparq import fit_saturation, saturation_curve

# Illustrative saturation measurement: 0.5 s of counting at each power
# (the first with the laser blocked, for the dark counts)
power = np.array([0, 50, 100, 200, 300, 450, 600, 900, 1300, 2000.0])  # uW
truth = dict(R_inf=600e3, P_sat=350.0, bg_slope=60.0, dark=300.0)
counts = np.random.default_rng(5).poisson(saturation_curve(power, **truth) * 0.5)

fit = fit_saturation(power, counts, t_s=0.5)
print(f"R_inf {fit['R_inf'] / 1e3:.1f} kcps [{fit['R_inf_lo'] / 1e3:.1f}, "
      f"{fit['R_inf_hi'] / 1e3:.1f}], P_sat {fit['P_sat']:.0f} uW "
      f"[{fit['P_sat_lo']:.0f}, {fit['P_sat_hi']:.0f}] (95 %)")
print(f"background {fit['bg_slope']:.1f} cps/uW, dark {fit['dark']:.0f} cps, "
      f"p-value {fit['p_value']:.2f}")
print(f"signal fraction at 300 uW: {fit['rho'][4]:.3f}")
```

```
R_inf 605.6 kcps [597.0, 614.3], P_sat 355 uW [348, 362] (95 %)
background 58.2 cps/uW, dark 324 cps, p-value 0.48
signal fraction at 300 uW: 0.940
```

`fit_saturation` fits the counts at each excitation power with
`R_inf P / (P + P_sat) + b P + d`: the emitter's rate, which levels off
at `R_inf` and reaches half of it at `P_sat`, plus background light that
grows in proportion to the power (`b`) and the detectors' dark counts
(`d`). This is the curve that the package's three-level rate model gives
under continuous excitation when the pump rate is proportional to the
power: `saturation_from_rates` gives `R_inf` and the pump rate at
saturation from the model's rates, and the tests check the formula
against the model's steady state. The counts are fitted by their Poisson
likelihood, and the intervals are profile-likelihood intervals, as in
example 3. `rho` is the emitter's share of the light at each power (the
signal fraction of example 6) as the fit sees it. When all powers stay
far below `P_sat`, the curve has hardly bent, and a late, high
saturation cannot be told from a straight rise: the upper ends of the
intervals are then NaN, `saturated` is False and the function warns
(the tutorial below shows this). The p-value assumes Poisson counts. A
drifting focus or laser power makes the points scatter more, and a
curve of another shape does not fit; either way the p-value comes out
small.

### 22. An exact interval and test for the window g2

```python
import numpy as np
from sparq import (EmitterSite, HBTConfig, expected_histogram, bayesian_g2,
                   window_g2_ci, count_ratio_ci)

site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                        rho=0.95, blinking=False), n_emitters=2)  # a pair
cfg = HBTConfig()
hist = np.random.default_rng(2).poisson(expected_histogram(site, 30.0, cfg))
w = window_g2_ci(hist, cfg)                       # exact, frequentist
lo, hi = bayesian_g2(hist, cfg).credible_interval(0.95)
print(f"window g2 {w['g2']:.3f}: exact 95 % interval [{w['lo']:.3f}, "
      f"{w['hi']:.3f}], Bayesian [{lo:.3f}, {hi:.3f}]")
print(f"p-value of 'window g2 >= 0.5': {w['p_value']:.2f}")

# pulsed: counts in the zero-delay peak against the sum of 8 side peaks
r = count_ratio_ci(k0=37, kr=2410, n0=1, nr=8)
print(f"peak ratio {r['g2']:.4f}, exact 95 % interval [{r['lo']:.4f}, "
      f"{r['hi']:.4f}], p-value of '>= 0.5': {r['p_value']:.1e}")
```

```
window g2 0.601: exact 95 % interval [0.492, 0.726], Bayesian [0.495, 0.722]
p-value of 'window g2 >= 0.5': 0.97
peak ratio 0.1228, exact 95 % interval [0.0863, 0.1698], p-value of '>= 0.5': 1.8e-27
```

`window_g2_ci` gives the window g2 of example 2 (the zero-delay bin
against the far bins) with a confidence interval instead of a
posterior. Given how many counts fell in the two windows together, the
count in the zero-delay window is binomial, with a share set by the
window g2 alone, whatever the count rate (the conditional method of
Przyborowski and Wilenski, Biometrika 31, 313 (1940); example 13 uses
the same fact). The Clopper-Pearson interval of that binomial (Clopper
and Pearson, Biometrika 26, 404 (1934)) then covers the window g2 at
least as often as the level says, for any count rate. `p_value` is the
exact one-sided test of "window g2 >= 0.5": a small value says that the
site is below 0.5, and calling a site single when the p-value is at
most 0.05 does so for a site at 0.5 or above in at most 5 % of runs.
With many counts, as here, the exact interval and the Bayesian one of
example 2 nearly agree. `count_ratio_ci` does the same for any two
counts, for example the zero-delay peak of a pulsed histogram against
the sum of several side peaks; that ratio includes the background under
the peaks, which raises it.

### 23. Excited-state lifetime from a TCSPC decay

```python
import numpy as np
from sparq import fit_lifetime

# Illustrative TCSPC record: the delays (ns) after the laser pulse of
# 40 000 photons from an emitter with a 2.5 ns lifetime, 0.1 ns timing
# jitter and an 80 MHz laser (12.5 ns period), plus 1000 dark counts
rng = np.random.default_rng(23)
t = 1.0 + rng.exponential(2.5, 40_000) + rng.normal(0.0, 0.1, 40_000)
dtime = np.r_[t % 12.5, rng.uniform(0.0, 12.5, 1000)]
counts, edges = np.histogram(dtime, bins=500, range=(0.0, 12.5))
delay = 0.5 * (edges[1:] + edges[:-1])          # bin centers, ns

fit = fit_lifetime(delay, counts, t_rep=12.5)
print(f"lifetime {fit['tau'][0]:.3f} ns, 95 % interval "
      f"[{fit['tau_lo'][0]:.3f}, {fit['tau_hi'][0]:.3f}]")
print(f"jitter {fit['sigma_irf']:.3f} ns, decay starts at {fit['t0']:.3f} "
      f"ns, background {fit['background']:.0f} counts/ns, "
      f"p-value {fit['p_value']:.2f}")
```

```
lifetime 2.466 ns, 95 % interval [2.420, 2.513]
jitter 0.102 ns, decay starts at 1.002 ns, background 94 counts/ns, p-value 0.89
```

With a pulsed laser, a TCSPC card (time-correlated single-photon
counting) times each photon from the laser pulse before it; in T3 mode
`read_ptu` returns these delays as `dtime`, in units of the resolution
`resolution_s`. Their histogram is the emitter's decay. `fit_lifetime`
fits it with an exponential decay blurred by Gaussian timing jitter (the
instrument response), plus a flat background. With `t_rep` the decays of
earlier pulses that have not died out by the next pulse are added. Give
it for pulsed data: a share e^(-t_rep/tau) of each decay is left at the
next pulse (12 % for a 6 ns lifetime at 80 MHz), and when that share is
negligible the fits with and without it agree (to 1e-4 for a 0.8 ns
lifetime at 80 MHz in the tests); the third tutorial shows what goes
wrong without it. The fit maximizes the Poisson likelihood of the
counts, and the interval is a profile-likelihood interval.
`n_components=2` fits two lifetimes; `sigma_irf` and `t0` may be given
when they are known. A small p-value says that the model does not
describe the decay, for example a second lifetime or an instrument
response with a slower tail than a Gaussian. `expected_decay` gives the
model's mean counts.

## Tutorials

Three scripts in the [`examples`](examples) folder of the source
repository (and, from 0.14.0, of the source distribution) go from data
to a final result with uncertainties. Each one makes its own data with a
fixed random seed (illustrative numbers, not measurements) and uses only
the package. Run them from the repository folder, for example `python
examples/tutorial_saturation.py`. The output below is what they printed
with sparq-triage 0.14.0 (Python 3.13, NumPy 2.5.3, SciPy 1.18.1); each
script quotes it at the top, and a test runs the scripts and checks
their numbers.

**Saturation curve** (`examples/tutorial_saturation.py`, about 2 s).
Counts at 15 powers (one with the laser blocked, for the dark counts)
are fitted with `fit_saturation`. The script prints `R_inf` and `P_sat`
with their 95 % intervals, the background and dark rates, the emitter's
share of the light at two powers, and what happens when the same
emitter is measured only far below saturation.

```
R_inf = 849.1 kcps, 95 % interval [843.2, 855.1] (true 850)
P_sat = 419.0 uW, 95 % interval [415.3, 422.7] (true 420)
background 89.9 cps/uW, dark 233 cps; fit p-value 0.95
signal fraction at 400 uW: 0.920; at 2400 uW: 0.770
powers up to 12 uW only: saturated False, P_sat upper end nan, warning issued: True
```

**From time tags to a verdict** (`examples/tutorial_timetags.py`, about
15 s). Two seconds of clicks from the photon-by-photon simulator, with
detector jitter and dead time, are written to a PicoQuant PTU file and
read back. The coincidence histogram then gives the window g2 with its
exact interval and test (example 22), the Bayesian probability of
example 2, and a profile-likelihood interval for g2(0) itself with the
flat level pinned by the counted rates (example 3). The zero-delay bin
holds only 12 counts (against about 90 per far bin), so both estimates
scatter: in this run both come out below the true g2(0) of 0.1536,
which lies inside both intervals.

```
399568 + 398752 clicks in 2 s (399.2 kcps counted)
window g2 = 0.133, exact 95 % interval [0.069, 0.233]; p-value of g2 >= 0.5: 6.6e-09
Bayesian P[window g2 < 0.5] = 1.000000
model g2(0) = 0.113, 95 % interval [0.068, 0.162] (true 0.1536)
```

**Lifetime from a TCSPC decay** (`examples/tutorial_lifetime.py`, about
6 s of computing time). Photon delays after the pulses of an 80 MHz
laser are made photon by photon (an exponential decay, Gaussian timing
jitter, dark counts) and histogrammed in 25 ps bins, as a TCSPC card in
T3 mode would record them. `fit_lifetime` gives the lifetime with its 95
% interval, the width of the instrument response and a p-value. A 6 ns
lifetime leaves 12 % of each decay for the next period; fitted as if
every decay ended within its period, the lifetime comes out at 4.1 ns
and the p-value rejects the fit. An emitter with two decays (0.9 ns and
5 ns) is rejected by the one-lifetime fit, and the two-lifetime fit
returns both lifetimes with intervals and the share of the photons in
each. One run is shown: its first p-value, 0.04, is small by chance (in
40 runs of this kind, done once outside the test suite, 1 fell below
0.05).

```
lifetime 5.938 ns, 95 % interval [5.606, 6.225] (true 6); IRF FWHM 276 ps (true 283); p-value 0.04
without the laser period: lifetime 4.074 ns, p-value 2.8e-09
one lifetime: 1.159 ns, p-value 1.5e-49
two lifetimes: 0.924 ns, 95 % interval [0.887, 0.953] (true 0.9), share of photons 0.704
two lifetimes: 5.821 ns, 95 % interval [4.213, 6.728] (true 5.0), share of photons 0.296
two lifetimes: p-value 0.02
```

## What is in the package

Everything below except the last group is importable straight from
`sparq`. Each function's docstring (`help(sparq.bayesian_g2)`, for
example) gives its inputs, units and conventions.

**Analysis core: measured data** (NumPy and SciPy only)

- `analyze_histogram` -- g2(0) of a CW (continuous laser) histogram
  with a bootstrap interval, the fit's p-value and a check of the
  search limits (examples 1 and 9). `analyze_pulsed` -- the same for
  a pulsed-laser histogram, by a fit of the whole comb (default) or
  from peak areas (example 17). `fit_pulsed` -- that comb fit, with a
  profile-likelihood interval and a p-value; `peak_cdf` -- the share
  of one peak below a given delay.
- `fit_g2_histogram` -- the model fit that `analyze_histogram` runs
  once per copy; returns `(g2_0, ok)`, and with `return_details=True`
  the fitted parameters, which of them sit on a search limit, and the
  g2(0) of a refit with those limits ten times wider.
- `t1_bounds` / `t2_bounds` (arguments of `analyze_histogram`,
  `fit_g2_histogram`, `profile_likelihood_ci`) -- the ranges, in ns,
  searched for `tau1` and `tau2`, or `"auto"`.
- `profile_likelihood_ci` -- model-based interval for g2(0), with
  optional `c0_prior` (example 3).
- `bayesian_g2`, `G2Posterior` -- the exact posterior of the raw
  central-window g2 (example 2). `bayesian_g2_model`,
  `ModelPosterior` -- the sampled posterior of g2(0) itself
  (example 11).
- `fit_model`, `goodness_of_fit`, `g2_model`,
  `exp_conv_gauss_complex` -- other dip shapes and their p-values
  (example 10); `goodness_of_fit` gives a p-value by simulation.
- `SPRTCertifier` -- sequential test between two emitter models
  (example 5). `WindowSPRT` -- sequential test with no emitter model
  (example 13).
- `load_hbt_csv`, `save_hbt_csv` -- the `delay_ns,counts` file format.
  `rebin_real` -- re-bins a histogram onto the analysis grid, centered
  on the dip (`method="split"` shares each input bin between the grid
  bins it overlaps; `"whole"` is the 0.9.1 method).
  `robust_flat_rate` -- the singles rate implied by the median of the
  far bins. `locate_dip` -- where the dip is and whether it stands out
  clearly (`analyze_histogram` warns when it does not).
- `background_corrected_g2`, `signal_fraction`,
  `deadtime_corrected_rate` -- closed-form corrections (example 6).
  `signal_fraction_from_counts` gives rho from counter readings on
  and beside the emitter, with an exact interval, for
  `background_corrected_g2(rho_ci=...)` (example 20).
- `check_stationarity`: whether a series of snapshots comes from a
  source that did not change (example 19).
- `window_g2_ci`, `count_ratio_ci`: the window g2 (or any ratio of
  two counts) with an exact confidence interval and the exact test of
  "g2 >= 0.5" (example 22).
- `fit_saturation`, `saturation_curve`, `saturation_from_rates`: the
  saturation curve: count rate against excitation power, fitted with
  intervals for `R_inf` and `P_sat` (example 21 and the first tutorial).
- `fit_lifetime`, `expected_decay`: the excited-state lifetime (one or
  two) from a TCSPC decay histogram, with intervals, and the model's
  mean counts (example 23 and the third tutorial).
- `load_timetags_csv`, `save_timetags_csv`, `correlate`,
  `correlate_start_stop`, `normalize_g2` -- time tags to g2
  (example 7). `read_ptu`, `load_ptu_timetags`, `save_ptu_t2` --
  PicoQuant PTU files (example 15).
- `site_from_numbers`, `expected_posterior`,
  `required_acquisition_time`, `certification_probability`,
  `assured_acquisition_time` -- measurement planning (examples 4 and
  12).
- `heralded_g2_limit`, `car_for_purity`, `heralded_source`,
  `heralded_from_car` -- heralded sources (examples 8 and 14).

**Analysis core: models and simulators**

- `g2_three_level` -- the ideal model above. `g2_measured` -- the same
  with several emitters, background and instrument response.
  `g2_zero` -- g2(0) of a site with no instrument response.
  `site_g2` -- the g2 the fast simulator uses for a site (with
  blinking).
- `HBTConfig` -- the histogram grid (`tau_max`, `n_bins`, `sigma_irf`,
  `bin_average`; `bin_width`, `bin_centers`, `bin_nodes`).
- `EmitterSite` -- one site (`params` dict and `n_emitters`; `g2_0`,
  and `is_good` = g2(0) < 0.5, brighter than 60 kcps and not
  blinking).
- `Platform`, `PLATFORMS`, `register_platform`, `sample_site` --
  parameter ranges per emitter type. Four are built in: NV, hBN, GaN
  and SiV. Their ranges are described in the source as anchored to
  published photophysics, with the citations in the manuscript; the
  package itself does not list them. `register_platform` adds your own.
- `expected_histogram`, `sample_histogram`, `sample_event_stream` --
  the fast simulator: the mean histogram (with exact blinking and
  optional dead time and afterpulsing), Poisson draws from it, and the
  same split into
  time slices. `mean_detected_rate_cps`, `deadtime_throughput` -- the
  rates it uses.
- `simulate_photon_stream`, `DetectorImpairments` -- a slower
  photon-by-photon simulator with optional detector jitter, dead time
  and afterpulsing, and blinking.
- `liouvillian`, `steady_state`, `g2_exact`, `effective_params`,
  `rates_for_params`, `rates_from_site` -- an exact three-level
  rate-equation model of the emitter. `rates_for_params` gives rates
  whose g2 is exactly a given (`tau1`, `tau2`, `a`); the
  photon-by-photon simulator uses it.
- `expected_hist_pulsed`, `peak_shape`, `calibrate_comb`,
  `g2_peak_area` -- the pulsed-laser simulator, the location of the
  zero-delay peak, and the peak-area analysis of
  `analyze_pulsed(method="area")` (default repetition period 12.5 ns,
  i.e. 80 MHz).
- `__version__`.

**Machine-learning layer** (needs the `ml` extra; import the modules
explicitly)

- `sparq.estimators` -- a 1-D convolutional network (a standard
  pattern-recognizing neural network) on histograms (`HistCNN`,
  `TriageCNN`), a spiking neural network (whose model neurons pass
  on-off pulses, suited to data arriving over time) on time-sliced
  data (`SpikingG2Net`), their training loops and `evaluate`.
- `sparq.twin_torch` -- a differentiable version of the simulator, as
  a function of laser power and window width, and
  `fisher_info_g2zero`.
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
  negative, or either holds a NaN or infinity, or `T_s` is not a
  positive number (`analyze_histogram`, `analyze_pulsed`);
  `singles_cps` is not one or two positive rates;
- `fit_g2_histogram` and `profile_likelihood_ci` get a histogram whose
  length is not `cfg.n_bins`, counts that are negative or not finite,
  or a `T_s` or `r_hat` that is not a positive number;
- `fit_pulsed` gets delays that are not strictly increasing or
  decreasing, a record shorter than three laser periods, no counts,
  `T_s` without `singles_cps` or the reverse, or a non-positive
  `t_rep` or `sigma_pair`; `analyze_pulsed` an unknown `method`, or
  fit options with `method="area"`; `g2_peak_area` a period so short
  that no delay lies 4.5 ns from every peak (use the fit);
- `expected_histogram` gets `afterpulse_p` outside [0, 1] or a
  non-positive `afterpulse_tau_ns`; `expected_hist_pulsed` a lifetime
  `tau_e` or period `t_rep` that is not positive and finite, a negative
  `sigma_pair` or a non-finite `center_off` (from 0.13.0);
- the data's bins are wider than the analysis grid's bins, which would
  leave grid bins empty or invent resolution (`rebin_real`, and so
  `analyze_histogram`; pass a coarser `cfg`); the analysis window
  centered on the dip reaches beyond the recorded delays (the message
  gives the dip position and the largest `tau_max` that fits);
  `method="split"` gets delays that are neither increasing nor
  decreasing;
- a `t1_bounds` / `t2_bounds` range is not `(low, high)` with
  0 < low < high, or a string other than `"auto"`;
- a histogram has too few counts to profile, or `c0_prior` is not a
  pair of positive, finite numbers (`profile_likelihood_ci`, and the
  same checks in `fit_g2_histogram`, `fit_model` and
  `bayesian_g2_model`); `profile_likelihood_ci` also refuses a `level`
  outside (0, 1) and fewer than 3 grid points;
- `check_stationarity` gets fewer than two snapshots, snapshots not on
  the grid `cfg`, negative or non-finite counts, durations that are
  not positive, or so few counts that a test would have an expected
  count below `min_expected` (default 5);
- `bayesian_g2` gets a histogram of the wrong length, negative or
  non-finite counts, a grid with no bin centered on zero, an even or
  too-wide central window, an empty reference window, windows that
  overlap, or a prior with shape <= 0 or rate < 0; `G2Posterior.mean`
  when the mean does not exist (almost no far counts), and quantile or
  interval levels outside (0, 1);
- `fit_model` gets an unknown model, a fixed parameter the model
  does not have, or a `T_s` or `r_hat` that is not positive and
  finite; `bayesian_g2_model` a histogram of the wrong length,
  no counts, or `a_max` <= 0;
- `SPRTCertifier` gets error rates outside (0, 1), a hypothesis with
  zero expected counts in some bin, data on a different grid, negative
  or non-finite counts, or a time step that is not a positive number; `WindowSPRT` error rates outside (0, 1),
  `g_accept` not below `g_reject`, a grid with no bin centered on zero,
  windows that are empty or overlap, or negative or non-finite counts;
- `site_from_numbers` gets non-positive or non-finite times or rate,
  a negative `a`, `rho` outside (0, 1], or a non-integer or zero
  emitter count; `expected_posterior` and `certification_probability`
  a non-positive time;
- `required_acquisition_time` and `assured_acquisition_time` get a
  confidence or assurance outside (0, 1) or a bad `t_max_s`, a site
  whose window g2 is not below the threshold (no run can certify it;
  example 4), or a target not reached within `t_max_s` (the message
  gives the probability reached);
- `background_corrected_g2` gets `rho` outside (0, 1], a `g2_meas`
  or interval that is not finite, an interval with low > high, or a
  `rho_ci` that reaches 0 or does not contain `rho`;
  `signal_fraction` negative or non-finite rates or a zero total;
  `signal_fraction_from_counts` counts that are negative or not whole,
  times that are not positive, no counts on the emitter, or a
  background rate not below the rate on the emitter;
  `deadtime_corrected_rate` negative or non-finite inputs or a
  measured rate at or above `1/tau_d` (example 6);
  `expected_histogram` a negative dead time;
- `rates_for_params` gets `tau2` not above `tau1` (with `a` > 0), a
  negative `a`, or a pump fraction for which no rates exist (the
  message names the smallest that works); `simulate_photon_stream`
  gets a site for which no three-level rates exist (for example
  `tau2` not above `tau1` with `a` > 0);
- `heralded_g2_limit` gets a raw CAR below 1 + 1/K (1 for
  laser-pumped, 2 for thermal) or not finite, an unknown `statistics`
  or `car_definition`, or a non-positive `modes`; `car_for_purity` a
  target outside (0, 1] (laser-pumped) or (0, 1.5] (thermal);
  `heralded_source` efficiencies outside (0, 1], dark-count
  probabilities outside [0, 1), or a non-positive `mu`;
- a histogram file has the wrong header, a row without exactly two
  fields, fewer than 5 bins, non-finite values, negative counts or a
  delay axis that is not strictly increasing (`load_hbt_csv`);
  `save_hbt_csv` gets arrays of unequal length or fewer than 5 bins;
- a time-tag file is empty, does not have two columns, has non-finite
  entries or unknown channels (`load_timetags_csv`; use `channel_a`,
  `channel_b` to map your numbering); `save_timetags_csv` gets arrays
  of different shapes;
- a PTU file does not start with `PQTTTR`, has an unknown tag or
  record type, or ends before the number of records its header gives
  (`read_ptu`); a requested channel has no events
  (`load_ptu_timetags`, which names the channels that do);
  `save_ptu_t2` gets times that are negative or not increasing,
  channels out of range, or a T3 record type;
- time tags are not sorted in time: the stop tags `t_b` for
  `correlate`, both channels for `correlate_start_stop`;
- `normalize_g2` gets a histogram of the wrong length, negative or
  non-finite counts, click counts that are not positive, or a
  non-positive time; `correlate` and `correlate_start_stop` time tags
  that are not finite (from 0.13.0);
- `fit_saturation` gets powers or counts that are not finite, negative
  powers, counts that are negative or not whole, counting times that
  are not positive, arrays of different lengths, fewer distinct powers
  than the curve's parameters plus one (5 with the default background),
  no positive power, no counts, an unknown `background` or `interval`,
  or a `level` outside (0, 1); `saturation_from_rates` rates that are
  not positive (`k_es` may be 0);
- `fit_saturation` also refuses counts at zero power with
  `background="none"`, a curve that is 0 there (from 0.14.0);
- `fit_lifetime` gets delays that are not finite or not strictly
  increasing or decreasing, fewer than 5 bins (or too few for the
  parameters), counts that are negative, not finite or all zero, arrays
  of different lengths, a `t_rep` that is not a positive number, a
  negative or non-finite `sigma_irf`, a non-finite `t0`, an
  `n_components` other than 1 or 2, or a `level` outside (0, 1);
  `expected_decay` lifetimes that are not positive, negative areas or
  background, or a different number of areas and lifetimes;
- `window_g2_ci` gets what `bayesian_g2` refuses, counts that are not
  whole, or no counts in either window; `count_ratio_ci` counts that
  are negative or not whole, both counts zero, bin numbers that are not
  positive, a `level` outside (0, 1) or a threshold that is not
  positive;
- `register_platform` gets something that is not a `Platform`, a name
  already in use (unless `overwrite=True`), a range with low > high,
  non-positive bounds (negative ones for `a_rng`), `rho` outside (0, 1]
  or a blinking probability outside [0, 1];
- `load_fisequr` is not given an existing data directory.

## How the results are checked

237 automated tests run on every push to `main` and every pull
request. In CI they run on Python 3.10, 3.11, 3.12, 3.13 and 3.14 with
CPU PyTorch; once more on Python 3.14 without PyTorch, where the
machine-learning test files are skipped; and once more on Python 3.10
with the oldest versions the package allows (NumPy 1.24.0, SciPy
1.10.0, PyTorch 2.0.0). The PTU cross-checks run where the independent
readers `ptufile` and `phconvert` install, and are skipped elsewhere.
Most checks compare the package with something independent of it: a
closed-form result, a second calculation done another way, or the
known truth of a simulated data set. Tests that use random data use
fixed seeds. The main checks:

**Models and simulators**

- The ideal g2 is 0 at zero delay and 1 at very long delay, both to
  1e-12. `g2_zero` equals `g2_measured` at zero delay without jitter.
- The closed-form instrument-response blur matches a brute-force
  numerical convolution to 1e-4, and to 1e-6 for a lifetime of 0.05 ns
  under 0.5 ns of jitter (where 0.9.1 returned 0 or NaN); its extension
  to complex exponents matches the real formula to 1e-13 and a
  brute-force convolution to 1e-6.
- The exact three-level rate model gives the two-exponential formula,
  with the `tau1`, `tau2`, `a` computed from its rates, to 1e-10, on
  at least 25 of 30 random parameter sets (those whose eigenvalues are
  real). Its steady state leaves the rate equations unchanged to
  1e-12. `rates_for_params` round-trips through it to 1e-9 on 199 of
  200 random parameter sets (the other has no rates at that pump
  fraction and is refused).
- The two simulators against each other (chi-square tests, Poisson
  counts): the photon-by-photon histogram has the site's g2
  (p > 0.001; with the 0.9.1 rates p < 1e-6); with microsecond
  blinking the fast simulator's exact gate model passes and the 0.9.1
  flat-level model fails; with 45 ns dead time the detectors' counted
  rates match `deadtime_throughput` within 5 standard errors, while
  the Poisson-light formula r/(1 + r tau_d) is off by more, and the
  histogram passes with the dead time and fails without it.
- Bin averaging: the 8-point rule matches a 20 000-point average to
  1e-7. For a non-blinking site with bin averaging off, the fast
  simulator is the 0.9.1 formula bit for bit.
- The simulator's edge bin sits within 2 % of the accidental level and
  its central bin below 10 % of it; its Poisson draws have the right
  mean and variance; time slices add up to the same mean.
- Afterpulsing: the closed-form afterpulse integrals match numerical
  integration to 1e-12 (also where the shelving time equals the
  afterpulse time, a removable singularity); the site's g2 written as
  a sum of exponentials matches `site_g2` to 1e-12, with and without
  blinking. The jitter, left out of the afterpulse terms, changes them
  by less than 5e-4 of the flat level (times p) against a numerical
  integral of the blurred g2. With 20 % afterpulsing and 45 ns dead
  time the photon-by-photon histogram passes the chi-square test
  against the fast simulator (p > 1e-3) and fails it without the
  afterpulse terms (p < 1e-6), even with the flat level scaled to the
  counted rates; those rates match r (1 + p) within 5 standard errors.

**Analysis of histograms**

- Re-binning: a flat input of 0.25, 0.3, 0.5 or 0.7 ns bins gives a
  flat 1 ns histogram (within float32 rounding), while "whole" gives a
  ripple of more than 10 % where the widths do not divide; counts are
  kept; the split histogram is symmetric about the center. The dip
  center is found to 1e-3 ns on noise-free data and, over 15 noisy
  runs, with an rms error below 0.5 ns and below 0.6 times that of
  the 0.9.1 method.
- On noise-free bin-averaged data the fit recovers g2(0) to 1e-3; the
  0.9.1 center-sampled fit is off by more than 0.01.
- An analysis window that would reach beyond the recorded delays is
  refused (with the dip at +7 ns in a +-60.5 ns record the message
  names 53.5 ns, and with that window g2(0) comes out within 0.005).
  A dip narrower than a bin inside a 5 ns bunching peak is found to
  1e-3 ns (the 0.9.1 method put it tens of ns away); in records of
  +-500 ns (dip at +37 ns, with and without a shoulder) the dip is
  found within 1 ns in all 12 runs and is not reported as unclear,
  while a 0.5 ns dip on 1 ns bins in the same record is reported as
  unclear in every run (with a warning); a dip 1 ns beyond the range
  whose window fits is refused, not moved; a descending delay axis
  gives the same result as an ascending one.
- Over 15 simulated runs each of one and two emitters: with the
  singles rates every run is within 0.06 of the truth and the rms
  error is below 0.03, with no warning; without them the misses reach
  more than 0.1, and at least half of those are flagged. The fit
  p-value is not too small: over 20 runs (0.5 and 0.3 ns input bins)
  at most 15 % fall below 0.05 and the mean lies between 0.4 and
  0.75. (Over 300 runs, done once outside the test suite, 2 to 3 %
  fell below 0.05: the p-value errs on the large side.)
- `analyze_histogram` on one-emitter data with a 7 ns offset: dip
  found within 2 ns, g2(0) within 0.1, whole interval below 0.5; for a
  pair with the singles rates: within 0.05. The fit on a noise-free
  histogram lands within 0.05 of the true g2(0) for one and three
  emitters (this test needs PyTorch).
- A fast emitter (`tau1` = 0.15 ns, `tau2` = 5 ns): with the default
  ranges the fit is off by more than 0.2, sits on its limits and
  warns; with its own ranges or `"auto"` it is within 0.05, with no
  warning.
- Profile-likelihood intervals contain the truth where the instrument
  response matters, at 10 s and 60 s with `c0_prior` (the 60 s one
  less than half as wide), and without it (wider); they separate a
  single emitter from a pair. From 0.12.0, on the data of example 3,
  an independent Nelder-Mead profile (the model built from
  `g2_measured`) puts the deviance at both ends at the chi-square
  quantile 3.84 within 0.03 and its minimum within 2e-4 of `g2_hat`,
  with and without `c0_prior`; the 0.11.0 ends sat at 2.15 and 4.45
  (pinned) and at 2.63 and 1.16 (free).
- Pulsed data: `analyze_pulsed` within 0.05 of the true 0.12; the
  peak-area method within 0.03 on noise-free data with 1 ns lifetimes;
  the comb center within 2 bins (also, from 0.11.0, in a record of
  3.5 periods); the peak shape has unit area to 1e-3.
- Comb fit: `peak_cdf` matches a numerical integral of `peak_shape` to
  1e-7 and, without jitter, the two-sided exponential law to 1e-15;
  the exact bin shares match a midpoint sum over sixteen times finer
  bins of `expected_hist_pulsed` to 2e-5 of the largest bin (the
  midpoint rule's own error). On such noise-free data the fit returns
  g2(0) to 1e-4 at 80 MHz (1 and 2 ns lifetimes, with and without
  background) and at 160 MHz, where the peak-area method gives 0.104
  and 0 at 80 MHz and is refused at 160 MHz; with a blinking envelope
  it returns g2(0) to 1e-3, and its interval stays open above unless
  the singles rates are given. The interval's ends are confirmed by an
  independent Nelder-Mead minimization of the likelihood (deviance at
  the chi-square quantile within 0.03). The exact deviance moments
  behind the p-value match 200 000 simulated Poisson counts and direct
  sums to 1e-8 (their large-count expansion too, from 1000 to 1e5
  counts per bin), and a 65 536-bin histogram with peaks of 1e5 counts
  needs less than 100 MB for them. Done once
  outside the test suite (simulated runs at 80 MHz, 1 ns lifetime,
  g2(0) = 0.12): the 95 % intervals covered the truth in
  184 of 200 runs (0.92) with about 140 counts in the zero-delay peak,
  in 91 of 100 with a 2.5 ns lifetime and background, and in 94 of 100
  with about 7 counts in that peak; 6 % of 500 p-values fell below
  0.05.
- The pulsed simulator (from 0.13.0) equals a sum of `peak_shape` over
  161 peaks to 1e-12 for lifetimes of 1, 5 and 8 ns at 80 MHz, and the
  comb fit of its noise-free 8 ns histogram returns g2(0) = 0.12 to
  1e-4 (0.1031 with the 0.12.1 simulator, whose comb was cut short).

**Other dip shapes and the posterior of g2(0)**

- The three-level and multi-exponential models equal the package
  formula to 1e-12. The coherent model equals the closed form of
  Kimble, Dagenais and Mandel to 1e-12 without dephasing, and a direct
  integration of the optical Bloch equations to 1e-7 with dephasing
  and detuning.
- Fit p-values are uniform over 20 runs when the model is right
  (Kolmogorov-Smirnov p > 0.01, at most 20 % below 0.05). On a
  coherently driven emitter the three-level fit has p < 1e-3 and the
  coherent fit p > 0.01, recovering g2(0) within 0.05; the simulated
  p-value of `goodness_of_fit` agrees.
- The ensemble sampler reproduces the closed-form beta-prime posterior
  of `bayesian_g2` (quantiles within 0.02). `bayesian_g2_model`'s 90 %
  intervals contain the truth in at least 11 of 15 runs; its median
  and 95 % interval agree with the profile likelihood within 0.03 and
  0.04; without `c0_prior` the interval is at least 1.5 times wider.

**Bayesian verdict and planning**

- The hand-built posterior of `bayesian_g2` matches SciPy's
  independent beta-prime distribution (to 1e-12, quantiles 1e-9),
  integrates to 1, agrees with direct integration of the Poisson
  likelihood and with random Gamma-ratio draws, and its 95 % intervals
  cover the truth in 90 % to 100 % of 200 simulated histograms.
- The planner equals `bayesian_g2` on `expected_histogram`; the
  planned time reaches the confidence and 0.8 of it does not; at 4
  times the planned time more than 90 % of 200 simulated runs certify.
- `certification_probability` agrees with 2000 simulated runs within
  4 standard errors at three times; the assured time reaches its
  assurance, stays above it on 41 later times, and is missed just
  before; 2000 simulated runs at that time certify at the assured
  rate.

**Sequential tests**

- `SPRTCertifier`: the evidence does not depend on how the data are
  split; over 80 + 80 simulated runs the observed error rates are at
  most 0.125; mean decision times lie between Wald's prediction and
  twice it.
- `WindowSPRT`: over 500 simulated runs at each of g = 0.5 and 0.7 (and
  0.25 and 0.1), with a steady rate and with a rate that changes
  between increments, the wrong decisions stay within Wald's bounds
  (plus three standard errors), and every run decides.

**Stationarity of a series of snapshots**

- `check_stationarity`: its three statistics equal those of SciPy's
  `chisquare` and `chi2_contingency` (also after bins are merged,
  where every expected count is at least 5); over 200 simulated steady
  series of six snapshots its p-values are uniform (Kolmogorov-Smirnov
  p > 0.01, at most 9 % below 0.05) and at least 194 are called
  consistent; a second emitter from the fourth snapshot on gives a
  window p-value below 1e-6, and a 20 % dimmer source a rate p-value
  below 1e-6 while the window and shape tests stay above 1e-3.

**Corrections, heralded sources, time tags, files**

- The background correction inverts the simulator's own background
  model to 1e-12; the dead-time correction round-trips to 1e-12 and
  its forward formula matches the photon-by-photon detector chain on
  Poisson light within 2 %.
- `signal_fraction_from_counts`: its ends equal Clopper-Pearson limits
  solved independently from binomial tail sums (to 1e-9), and over
  2000 simulated pairs of counter readings (two settings) its 90 %
  intervals cover the true rho at least 90 % of the time (within three
  standard errors). With `rho_ci`, the corrected interval equals the
  smallest and largest value on a 201 x 201 grid over the two
  intervals (to 1e-12), for intervals below, across and above 1; and
  a 97.5 % g2 interval with a 97.5 % rho interval covered the true
  g2(0) in at least 95 % of 2000 simulated runs.
- Heralded formulas: both satisfy their defining quadratics; the
  smallest possible CAR gives exactly 1 (laser-pumped, CAR 1) and 1.5
  (thermal, CAR 2); at large CAR they approach 2/CAR and 4/CAR; the
  inversions round-trip; the net-CAR form equals the 0.9.1 thermal
  formula to 1e-15. The exact detector model reproduces both
  low-efficiency relations to 1e-5 for 1, 3 and infinitely many modes;
  a photon-by-photon Monte Carlo of four sources (with high
  efficiencies and dark counts) agrees with it within 4 standard
  errors; it shows g2_h(0) below 0.6 times the formula with an
  efficient herald and above 1.1 times with dark counts;
  `heralded_from_car` recovers mu to 1e-8 and finds both solutions
  when dark counts make two; a perfect signal efficiency gives the
  same result as one just below it.
- PTU files: every record type (six T2, six T3; with overflow records,
  including T2 records that carry 128 or more overflows, and marker
  records) is decoded to the written channels, times and delays
  exactly. The same files decoded by `phconvert` give identical times,
  channels and delays, and by `ptufile` identical times and delays;
  for a T2 record that carries 154 overflows, `ptufile` added 26
  (PicoQuant's demo reader and `phconvert` add all 154), so its T2
  check uses data without such long gaps. Simulated tags through a PTU file give the
  same histogram within 2 counts per bin. From 0.14.0 `save_ptu_t2`
  writes its records without a Python loop; for every T2 record type,
  with sync events, gaps of several wrap periods and a gap of more than
  twice what one overflow record can carry, the records equal those of
  the 0.13.0 loop exactly (they are whole numbers) and the files are the
  same byte for byte.
- `correlate` equals a brute-force count of every pair; start-stop
  never exceeds it; from 0.13.0 `correlate`, `correlate_start_stop` and
  the detectors' dead-time pass equal the 0.12.1 loops count for count
  (on simulated tags, dense random tags with ties and delays on bin
  edges, and with the pairs split into several blocks); independent click streams normalize to g2 = 1;
  histogram and time-tag files round-trip exactly and malformed ones
  are refused.

**Saturation curve and the exact window interval** (new in 0.13.0)

- The saturation law equals the emission rate of the three-level rate
  model, computed from the steady state of its rate matrix (an
  eigenvector, `sparq.exact`), to 1e-12 for 30 random rate sets and
  pump rates from 1e-3 to 100 times the radiative rate.
- `fit_saturation`: from nearly noise-free counts (1e4 s per power) it
  returns `R_inf` and `P_sat` to 1e-6. On a simulated curve an
  independent Nelder-Mead minimization of the Poisson likelihood finds
  no better fit, and at both ends of both 95 % intervals the deviance,
  computed independently, is the chi-square quantile 3.84 within 0.01.
  Of 40 simulated curves of a dim emitter (160 to 4600 counts per
  power), at least 34 of the 95 % intervals for `P_sat` hold the truth.
  Powers far below saturation give open intervals and a warning, and a
  curve of another shape, R (1 - e^(-P/P0)), gets a p-value below 1e-6.
  Done once outside the test suite, over 200 simulated curves each of
  a bright emitter (2e4 to 6e5 counts per power) and of the dim one, the
  95 % intervals held the true `P_sat` in 190 and 188 runs and the true
  `R_inf` in 190 and 191; 2.5 % and 1 % of the p-values fell below 0.05
  (the p-value errs on the large side).
- `count_ratio_ci`: its ends equal Clopper-Pearson limits solved
  independently from binomial tail sums (to 1e-9), and its p-value the
  binomial tail sum (to 1e-12). Over 2000 simulated runs whose count
  rate changes from run to run, the 95 % intervals hold the true window
  g2 in at least 95 % of runs (less three standard errors), and the
  test at 0.05 calls a site with window g2 of 0.5 single in at most 5 %
  (plus three standard errors) and one at 0.1 in more than half. On a
  300 s histogram `window_g2_ci` and the credible interval of
  `bayesian_g2` agree within 3 %.

**Lifetime fit** (new in 0.14.0)

- The bin integrals of `expected_decay` equal differences of SciPy's
  `exponnorm` distribution function (an independent implementation of
  the exponential blurred by a Gaussian) to 1e-12 of the decay's
  photons, for jitter from 1/2000 to 50 times the lifetime; the
  distribution function equals the convolution integral, computed by
  quadrature, to 1e-9. With a laser period, the closed geometric sum
  over earlier pulses equals the direct sum over the pulses to 1e-12 of
  the largest bin (lifetimes from a twentieth of the period to more
  than twice it), and the bins of one period hold exactly one pulse's
  photons (to 1e-12).
- `fit_lifetime`: on noise-free histograms built from `exponnorm` it
  returns the lifetime and the areas to 1e-6 (one lifetime, a lifetime
  whose decay reaches far into the next periods, two lifetimes, and a
  single decay without a laser period, also with the jitter and t0
  given). On a noisy decay an independent Nelder-Mead minimization of
  the Poisson likelihood finds no better fit, and at both ends of the
  95 % interval its profile deviance is the chi-square quantile 3.84
  within 0.01. Of 20 simulated dim decays (about 3100 photons, the
  jitter given), at least 16 of the 95 % intervals hold the true
  lifetime. Two lifetimes
  fitted with one, and an 8 ns lifetime at 80 MHz fitted without the
  laser period (it comes out below 6 ns), get p-values below 1e-6; a
  lifetime 100 times longer than the record leaves the upper end of
  its interval open, and one far shorter than the jitter the lower end,
  each with a warning. Done once outside the test suite, over 200
  simulated decays each of a dim emitter (2 ns lifetime, about 3100
  photons in 125 bins) and of a bright one (6 ns, about 1e5 photons in
  500 bins, at 80 MHz), the 95 % intervals held the true lifetime in 191
  and 190 runs; 6.5 % and 3 % of the p-values fell below 0.05.

**Machine learning**

- An untrained convolutional or spiking network has a balanced
  accuracy below 0.7 (0.5 is chance); after
  150 training steps on simulated histograms its balanced accuracy on
  400 held-out sites is above 0.9, its training loss has halved, and
  its g2(0) error is below 0.7 times the untrained one.
- The soft actor-critic update learns a two-state, two-action problem
  with a known answer: Q-values within 0.1 of the rewards and the
  paying action chosen in each state.
- The graph encoder gives the same output (to 1e-6) for every
  numbering of the levels, passes finite gradients, and tells two
  platforms apart.
- `fisher_info_g2zero`: the scatter of the maximum-likelihood estimate
  over 300 simulated histograms is 1/I within 20 % (Cramer-Rao), and
  profiling out the other parameters gives less information.
- Shapes and interfaces of the networks, environment, baselines and
  replay buffer are checked as before; `load_fisequr` merges the parts
  of a series in synthetic files of its expected layout.
- The version matches the installed metadata and CITATION.cff; the
  core modules import with PyTorch blocked.
- The three tutorials run and print the numbers they quote (to 1 %, so
  that the check also holds with other NumPy and SciPy versions).

Not covered by tests: `TriageEnv` with a trained estimator and the
figures of the manuscript, and `load_fisequr` on the real data set.

## Corrections in earlier versions

**0.14.0 (this release) corrects this.**

- **A saturation fit that could not fit.** `fit_saturation` with
  `background="none"` and counts at zero power (the curve without dark
  counts is 0 there) returned, without intervals, `R_inf` = 429 and
  `P_sat` = 0.012 for counts whose fit with dark counts
  (`background="constant"`) gives 1974 and 190, with an infinite
  deviance and no explanation; with intervals
  it stopped with a NaN error from inside the interval search. It now
  refuses such data and names the cause.

**0.13.0 corrected these.**

- **The pulsed simulator cut its comb short.** `expected_hist_pulsed`
  added up the peaks only to one laser period beyond the recorded
  delays, so the tails of farther peaks were missing near the ends of
  the record. That is negligible for short lifetimes (below 4e-6 of the
  edge bins for 2 ns at 80 MHz) but not for long ones: the edge bins
  were 0.6 % low for 5 ns and 3.2 % low for 8 ns, and the comb fit of
  the noise-free 8 ns histogram gave g2(0) = 0.1031 (lifetime 7.65 ns)
  instead of 0.12. The comb now includes every peak whose tail reaches
  the record, and the fit returns 0.1200. The fit itself
  (`fit_pulsed`) was right; the simulated data were not.
- **Docstrings.** `fit_g2_histogram` and the module documentation
  called its fit Levenberg-Marquardt; with search limits, SciPy's
  `curve_fit` uses its trust-region reflective method. The docstring of
  `sparq.physics` said that the fast simulator models neither blinking
  nor dead time nor afterpulsing; it has modeled them since 0.10.0 and
  0.11.0.
- **Time tags that are not numbers.** `correlate` and
  `correlate_start_stop` left a NaN time tag out without a word; they
  now refuse tags that are not finite.

**0.12.0 corrected these.**

- **Profile-likelihood intervals were wrong.** `profile_likelihood_ci`
  minimized over the shelving time `tau2` in nanoseconds, a scale on
  which the optimizer hardly moved it from its start value (250 ns).
  It then measured the deviance from the smallest value on its grid
  of g2(0) (steps of 0.025) and drew the ends by straight lines
  between grid points. For the data of example 3 it gave
  [0.084, 0.126] with the flat level pinned and [0.067, 0.182] with it
  free; the profile computed independently gives [0.078, 0.124] and
  [0.064, 0.295]. The free interval was far too narrow. It now
  searches the lifetimes and the flat level on a log scale, writes the
  objective as half the Poisson deviance, and refines the best value
  and both ends between grid points. It takes two to three times as
  long as before (about 10 s for example 3).
- **Inputs that ran on.** `SPRTCertifier.update` accepted a NaN time
  step (the evidence became NaN and the test never decided) and
  negative counts; `deadtime_corrected_rate`, `signal_fraction` and
  `background_corrected_g2` returned NaN for NaN inputs; `fit_model`
  accepted an infinite `r_hat`. These are now refused.

**0.11.0 corrected these.**

- **Pulsed g2(0) too low when the peaks overlap.** The peak-area
  analysis, the default of `analyze_pulsed` up to 0.10.0, subtracts a
  background level taken more than 4.5 ns from every peak. With
  lifetimes that are not short compared with the laser period, the
  peaks' tails are still there, and the result is too low: on
  noise-free 80 MHz data with true g2(0) = 0.12 it gave 0.104 for a
  1 ns lifetime and 0.000 for 2 ns. The default is now a fit of the
  whole comb, which gives 0.120 for both (example 17). At periods
  below 9 ns the peak-area method returned NaN; it is now refused.
  With at most four peaks in the record, the zero-delay peak was
  always taken to be the first one; this is fixed.
- **Inputs accepted without a clear error.** `analyze_histogram`
  returned g2(0) = 1.0 (with `ok` False) for a negative acquisition
  time, and failed with unrelated errors for a zero time or NaN
  counts; `fit_g2_histogram` returned (1.0, False) for a histogram on
  another grid. These are now refused with a message that says why.

**0.10.0 corrected these.**

- **The two simulators disagreed.** The photon-by-photon simulator
  turned (`tau1`, `tau2`, `a`) into rates with an approximate formula,
  so its g2 was not the site's: for `tau1` = 8 ns, `tau2` = 100 ns,
  `a` = 1.5 the rates gave a shelving time of 43 ns instead of 100 ns.
  It now uses exact rates (`rates_for_params`), and a chi-square test
  against the fast simulator passes where the old rates fail it
  (p < 1e-6).
- **Blinking filled the dip in the fast simulator.** It added a flat
  level everywhere, including at zero delay, and switched the
  background off together with the emitter, unlike the
  photon-by-photon simulator. For the example of the tests the
  zero-delay value was 0.654 of the uncorrelated level instead of
  0.345. It is now the exact on/off-gate result, which matches the
  photon-by-photon simulator.
- **Model values at bin centers.** A correlator counts pairs anywhere
  in a bin, but the simulator and the fits used g2 at the bin center.
  On noise-free 1 ns bins of the README's site the fit then gave
  g2(0) = 0.086 instead of 0.0975. Both now average over the bin
  (`HBTConfig(bin_average=False)` restores the old behaviour).
- **Re-binning.** Putting each input bin whole into one grid bin gave a
  ripple (for 0.3 ns input bins, grid bins of 3 and 4 input bins) and,
  even for 0.5 ns bins, a zero-delay bin centered 0.25 ns off the dip.
  On noise-free 0.3 ns data (true g2(0) 0.0975) the fit gave 0.180
  with "whole" and 0.087 with "split" without the singles rates, and
  0.095 and 0.098 with them. The
  default is now proportional sharing with the dip center found to a
  fraction of a bin, and the fits use the shared bins' true variance.
- **Heralded thermal formula.** The thermal branch returned
  `(4 CAR + 2)/(CAR + 1)^2`, the single-mode result written for the
  net ratio (C - A)/A, while the laser-pumped branch used the raw
  ratio C/A that the documentation described. Both now use C/A
  (`car_definition="net"` gives the net form). The formulas were also
  called lower limits; the exact detector model shows they are not
  (example 14).
- **The jitter blur failed for short lifetimes.** For a lifetime below
  about a seventh of the pair jitter (0.07 ns at the default 0.35 ns
  per detector) the blurred exponential lost all precision (0 instead
  of 0.080 at zero delay for 0.05 ns) and turned to NaN far from zero
  delay, so the simulator returned NaN bins. It is now computed with
  the scaled error function and matches a brute-force convolution.
- **Analysis windows beyond the data.** When the recorded delays did
  not reach the analysis window's edges around the dip (a +-60 ns
  record with a cable delay, analyzed with the default +-60.5 ns
  window), the outer grid bins were left empty or partly empty and the
  fit was wrong without a warning. This is now refused, with the
  largest window that fits. The dip is also found differently: as the
  most significant local minimum (below both sides) over a range of
  widths, then refined by the symmetry of g2. The 0.9.1 smoothed
  minimum missed a dip narrower than a bin inside a bunching peak by
  tens of ns. The bootstrap now also locates the dip anew in each copy,
  so its interval includes the uncertainty of the dip position.
- **README claims based on one lucky run.** 0.9.1 said the analysis
  was within 0.1 (one emitter) and 0.12 (two) of the truth, from a
  single seeded run each; over 40 runs the histogram-only fit missed
  by more than 0.12 in about a fifth of them. The claims now come from
  many runs, with and without the singles rates (example 1).

**0.9.1 fixed three silent failures.** `correlate` accepted unsorted
stop tags and lost most pairs; `analyze_histogram` accepted data
binned more coarsely than its grid and returned a wrong g2(0); and
`load_timetags_csv` read a one-column file with two rows as one tag.
All three are now refused.

**0.6.0 changed the fit model.** Before 0.6.0 the fit and the profile
likelihood used the form `1 - d e1 + a e2` with `d` capped at 1. That
biased g2(0) upward for any emitter with a strong shoulder. Both now
use the physical form (the dip depth multiplies both exponentials),
and both include the instrument response, so they estimate the
jitter-free g2(0).

The full history is in [CHANGELOG.md](CHANGELOG.md).

## Limits

- Fits describe the three-level dip, several shoulders, or a
  coherently driven two-level emitter. Other shapes are not modelled;
  the fit p-value shows when a model does not describe the data, but
  cannot say which model would.
- A histogram alone pins g2(0) only loosely when the shoulder is slow
  compared with the window; give the detectors' singles rates. The
  warning about search limits catches most, not all, of the fits that
  go wrong this way.
- `"auto"` widens the lifetime ranges within fixed hard limits
  (1e-3 ns to 10 or 100 times the window).
- The dip is found automatically only when it stands out from the
  noise: in a check at about 7 counts per 0.5 ns bin, 2 of 20 runs of
  a 15 ns dip were placed more than 3 ns off (7 of 20 at 2 counts per
  bin). A dip about one bin wide can be lost in a long record even at
  high counts (a 0.5 ns dip on 1 ns bins in a +-500 ns record: 5 of 40
  runs at 170 counts per bin). A misplaced dip can make a pair or
  three emitters look like a single emitter. In every such run of
  these checks `locate_dip` reported the dip as unclear and
  `analyze_histogram` warned; the bootstrap, which locates the dip
  anew in each copy, then gives a wide interval. Heed the warning and
  pass `center` from a longer run or a calibration. Its refinement assumes a symmetric g2, which
  detectors with asymmetric timing jitter would break. The data must
  cover the analysis window around the dip.
- The fit p-value of `analyze_histogram` is approximate and errs on the
  large side; `fit_model` on native bins gives a calibrated one. Split re-binning makes
  neighbouring grid bins share input bins, so the Poisson-count tools
  (`bayesian_g2`, `fit_model`, `bayesian_g2_model`, `WindowSPRT`)
  should get native bins or bins merged whole.
- `bayesian_g2` and `WindowSPRT` work with the window-averaged g2, an
  upper bound on g2(0). The g2(0) posterior of `bayesian_g2_model` is
  sampled, and its width depends on the stated priors when the flat
  level is free.
- The planners use the Bayesian verdict of `bayesian_g2`; the assured
  time holds for the Poisson scatter of runs of a site whose
  parameters are as given.
- `SPRTCertifier`'s error rates are exact only if its two hypotheses
  are known exactly; `WindowSPRT` has guaranteed bounds but tests the
  window g2.
- The heralded-source model assumes pair numbers from independent
  thermal modes, perfectly correlated signal and idler, independent
  losses and click/no-click detectors, and no detector dead time.
- The fast simulator's dead time changes the rates only (not the shape
  of g2); the photon-by-photon simulator changes both. Its afterpulsing
  follows the photon-by-photon detector chain (afterpulses are not
  themselves blocked by dead time and give no further afterpulses) and
  leaves the jitter out of the afterpulse terms. The analysis tools
  do not model afterpulsing: like background light, it raises the
  measured g2(0).
- `profile_likelihood_ci` takes about 10 s for a 121-bin histogram.
  Its interval is that of the three-level model within the search
  ranges; when a nuisance parameter ends on a range limit (a warning
  says so), check the interval with a wider range, as in example 3.
- `fit_model` is a local search. When a model fits badly it can stop
  short of its best fit, by an amount that depends on the SciPy
  version (example 10: deviance 350.16 with SciPy 1.18, 345.75 with
  SciPy 1.10). The p-value is then too small, which does not change a
  clear rejection.
- `check_stationarity` uses the chi-square law of its statistics, which
  needs at least about 5 expected counts per cell; it refuses smaller
  counts instead of guessing. It tells that the source changed, not
  how, and a change slower than the whole run, or faster than one
  snapshot, is not seen.
- The rho interval of `signal_fraction_from_counts` assumes that the
  background beside the emitter equals the background on it.
- `fit_saturation` assumes the curve `R_inf P/(P + P_sat) + b P + d`
  (continuous excitation, a pump rate proportional to the power, and
  background light proportional to the power) and Poisson counts. Its
  intervals are too narrow when the points scatter more than that, for
  example from a drifting focus; a small p-value warns of it. `P_sat`
  and `R_inf` are bounded only when the powers reach well into
  saturation.
- `window_g2_ci` and `count_ratio_ci` give exact but conservative
  intervals (they cover at least as often as the level says), and like
  `bayesian_g2` they are for the window g2, an upper bound on g2(0)
  when the dip is wider than the window.
- `fit_lifetime` assumes a Gaussian instrument response, decays that
  are exponential (one or two lifetimes) and a flat background, and
  Poisson counts. An instrument response with a slower tail than a
  Gaussian is not described (measure the response on the laser light
  to see its shape); with enough counts the p-value shows the misfit.
  High count rates (more than one photon per laser
  period now and then, detector dead time) distort a TCSPC histogram;
  this is not modeled. A decay much longer than the period looks flat
  and is taken for background, and one much shorter than the jitter
  cannot be told apart from it: the interval is then open on that side
  and a warning is issued. The decay's rise must be in the record. The
  search starts from lifetimes between the bin width and the record
  length (or the period) and is local from there; with two lifetimes
  the likelihood can have more than one minimum, and a component far
  outside that range is not looked for.
- The pulsed comb fit assumes that every peak, the zero-delay one
  included, has the same shape (a two-sided exponential blurred by
  Gaussian jitter) and that the background under the comb is flat;
  its p-value shows when the peaks do not fit that shape. Its 95 %
  intervals covered the truth in 91 % to 94 % of simulated runs, a
  little below 95 %. With `envelope=True` and no singles rates the
  interval often stays open above. The fit takes about a second, and
  the interval a few seconds more.
- PTU reading was checked against two independent readers on files
  written from PicoQuant's format description, not on files recorded
  by hardware. Other vendors' binary formats are not read; export
  them to the two-column CSV format.
- The machine-learning tests check that the networks learn on the
  simulator and that the agent's update solves a small problem with a
  known answer; they do not check the performance figures of the
  manuscript.

## Where it comes from

The package is the installable part of the paper *"Closed-Loop,
Event-Driven Machine Learning for Autonomous Triage of Single-Photon
Emitters"* by T. M. Mahim, M. N. Islam, M. M. Rahman and A. S. M.
Mohsin, *ACS Photonics* (2026), published online on 15 October 2026,
[doi:10.1021/acsphotonics.6c01984](https://doi.org/10.1021/acsphotonics.6c01984).
Volume and pages are not yet assigned. The
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
