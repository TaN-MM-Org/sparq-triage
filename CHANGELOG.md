# Changelog

## 0.14.0 (2026-10-18)

Fits the excited-state lifetime of an emitter from a TCSPC decay
histogram with intervals, refuses a saturation fit that could not fit,
and writes PTU files and evaluates the planning functions faster with
the same results. A third worked tutorial.

### Added

- `fit_lifetime` and `expected_decay` (new module `sparq.lifetime`,
  exported from `sparq`): the decay of a TCSPC histogram (the photons'
  delays after the laser pulse, for example the T3 `dtime` of
  `read_ptu`), modeled as one or two exponential decays blurred by
  Gaussian timing jitter (the instrument response) plus a flat
  background. Each bin's share of a decay is a difference of the
  closed-form distribution function at its edges (no midpoint rule),
  evaluated with the scaled complementary error function. With the
  laser period `t_rep` the decays of earlier pulses that have not died
  out are added: the terms in which the jitter matters one by one, the
  rest as a geometric series in closed form. The counts are fitted by
  their Poisson likelihood (the areas and the background exactly, by
  an active-set Newton method, for each shape; t0, the jitter and the
  lifetimes by a bounded quasi-Newton search), with profile-likelihood
  intervals for each lifetime, the share of the photons in each
  component, and a goodness-of-fit p-value (the exact deviance moments
  of `fit_pulsed`). `sigma_irf` and `t0` may be given. A lifetime the
  data do not bound (much longer than the record or much shorter than
  the jitter) gets an open interval end (NaN above, 0 below),
  `resolved` False and a RuntimeWarning. No literature value is used;
  the model is checked against SciPy's independent `exponnorm` (1e-12),
  the convolution integral by quadrature (1e-9) and a direct sum over
  the pulses back to where less than 1e-18 of a decay is left (1e-12). The fit recovers the parameters of noise-free
  histograms to 1e-6; an independent Nelder-Mead minimization finds no
  better fit, and the profile deviance at both ends of the interval is
  3.84 within 0.01; at least 16 of 20 simulated 95 % intervals hold the
  truth (done once outside the suite: 191 and 190 of 200, for a dim
  and a bright emitter); two lifetimes fitted with one, and a long
  lifetime fitted without the laser period, get p < 1e-6. README
  example 23.
- `examples/tutorial_lifetime.py` (see Tutorials).

### Fixed

- `fit_saturation` with `background="none"` and counts at zero power,
  where that curve is 0: without intervals it returned R_inf = 429 and
  P_sat = 0.012 with an infinite deviance (the fit with dark counts,
  `background="constant"`, gives 1974 and 190), and with intervals it
  stopped with "The function value at x=4.292044655985884e-07 is NaN;
  solver cannot continue." from inside brentq. Such data are now refused with a
  ValueError that names the cause (test
  `test_saturation_without_background_refuses_counts_at_zero_power`).

### Performance

Measured on this machine (2 CPUs shared with other jobs), median of
seven interleaved runs of the old and the new code:

- `save_ptu_t2`: the records are formed without a Python loop over the
  events: 1119 ms -> 222 ms (5.0 times) for the 798 320 clicks of the
  time-tag tutorial. The records are whole numbers and equal those of
  the 0.13.0 loop exactly; the test keeps that loop and compares the
  records and the whole file byte for byte (every T2 record type, sync
  events, gaps of many wrap periods).
- `HBTConfig.bin_nodes`: the 4-point Gauss-Legendre nodes are computed
  once instead of on every call (the same NumPy call, so the same
  numbers; tested exactly): 200 calls of `expected_histogram`
  500 ms -> 309 ms (1.6 times); `certification_probability`, which
  calls it once per evaluation, 220 ms -> 178 ms for 20 calls
  (1.2 times; the probabilities are the same).

### Changed

- `fit_saturation` refuses counts at zero power with
  `background="none"` (see Fixed).
- Tutorials: the outputs quoted in `tutorial_saturation.py` and
  `tutorial_timetags.py` are what 0.14.0 prints (unchanged from
  0.13.0); the docstrings now name 0.14.0.
- Packaging: a new `MANIFEST.in` puts `examples/` into the source
  distribution, so that `tests/test_tutorials.py` runs the tutorials
  from an unpacked sdist instead of skipping them. The wheel is
  unchanged; building needs nothing new.
- Type hints on the new public functions.

### Tutorials

- `examples/tutorial_lifetime.py` (about 6 s of computing time): photon
  delays after the pulses of an 80 MHz laser, made photon by photon,
  histogrammed in 25 ps bins and fitted: a 6 ns lifetime with its 95 %
  interval and the width of the instrument response; the same data
  fitted without the laser period (the lifetime comes out at 4.1 ns,
  p = 2.8e-9); and an emitter with two lifetimes, rejected by the
  one-lifetime fit and resolved by the two-lifetime fit. Its printed
  output is quoted at the top of the script and in the README;
  `tests/test_tutorials.py` runs it and checks the numbers (to 1 %).

### Tests

- 237 tests (225 before): test_release_014 (11) and test_tutorials (one
  more). test_metadata also imports `sparq.lifetime` with PyTorch
  blocked.

## 0.13.0 (2026-10-17)

Fits the saturation curve of an emitter with intervals, adds an exact
frequentist interval and test for the window g2, corrects the pulsed
simulator for long lifetimes, and makes the correlators and the
detector dead time faster with the same results. Two worked tutorials.

### Added

- `fit_saturation`, `saturation_curve` and `saturation_from_rates`
  (new module `sparq.saturation`, exported from `sparq`): the count
  rate against excitation power, R_inf P/(P + P_sat) + b P + d, fitted
  to the counts at each power by their Poisson likelihood (for a given
  P_sat the rest is a convex problem, solved exactly; P_sat is searched
  on a log grid and refined), with profile-likelihood intervals for
  R_inf and P_sat, the fitted background slope b and dark rate d, the
  signal fraction at each power and a goodness-of-fit p-value (the
  exact deviance moments of `fit_pulsed`). `background` may be
  "linear" (default), "constant" or "none"; `interval` "both", "P_sat"
  or None. When the powers stay far below saturation the upper ends
  are NaN, `saturated` is False and a RuntimeWarning is issued. The
  curve is the steady-state emission rate of the package's three-level
  rate model with a pump rate proportional to the power;
  `saturation_from_rates` gives R_inf and the pump rate at saturation
  from the rates. No literature value is used: the law is derived from
  the rate equations and checked against the steady state of
  `sparq.exact.liouvillian` to 1e-12 (30 random rate sets). The fit
  recovers R_inf and P_sat to 1e-6 from nearly noise-free counts; an
  independent Nelder-Mead minimization finds no better fit, and the
  independently computed deviance at both ends of both intervals is
  3.84 within 0.01; at least 34 of 40 simulated 95 % P_sat intervals
  hold the truth (done once outside the suite: 190 and 188 of 200 for
  P_sat, 190 and 191 for R_inf, for a bright and a dim emitter); open
  intervals are flagged for unsaturated data, and a curve of another
  shape gets p < 1e-6. README example 21.
- `window_g2_ci` and `count_ratio_ci` (`sparq.bayes`, exported from
  `sparq`): the window g2 of `bayesian_g2`, or any ratio of two
  Poisson counts per bin, with the exact conditional (binomial)
  confidence interval (Przyborowski and Wilenski, Biometrika 31, 313
  (1940); Clopper and Pearson, Biometrika 26, 404 (1934)) and the exact
  one-sided p-value of "g >= threshold". Checked against
  Clopper-Pearson limits solved from binomial tail sums (1e-9), the
  binomial tail sum (1e-12), coverage and test size on 2000 simulated
  runs with a changing count rate, and the Bayesian interval at many
  counts (3 %). README example 22.
- `examples/tutorial_saturation.py` and `examples/tutorial_timetags.py`
  (see Tutorials).

### Fixed

- `expected_hist_pulsed` added the peaks only up to one period beyond
  the record, so the tails of farther peaks were missing near its ends:
  the edge bins were low by 3.7e-6 (2 ns lifetime at 80 MHz), 2.3e-4
  (3 ns), 0.57 % (5 ns) and 3.2 % (8 ns). The exact comb fit of the
  noise-free simulated histogram gave g2(0) = 0.1189 (5 ns) and 0.1031
  (8 ns, fitted lifetime 7.65 ns) instead of 0.12; it now gives 0.1200
  (test `test_pulsed_simulator_keeps_far_peak_tails`). The comb now
  holds every peak whose tail reaches the record (30 lifetimes plus 8
  jitter widths). README example 17 prints the same as before.
- Docstrings: `fit_g2_histogram` and the `sparq.analysis` and
  `sparq.estimators` module docstrings called the fit
  Levenberg-Marquardt; `curve_fit` with search limits uses SciPy's
  trust-region reflective method. The `sparq.physics` module docstring
  said that the histogram twin models neither blinking, dead time nor
  afterpulsing; it has since 0.10.0 and 0.11.0.

### Changed

- `correlate` and `correlate_start_stop` refuse time tags that are not
  finite (a NaN tag was silently left out) and arrays that are not
  1-D; both take the tags as float64. `expected_hist_pulsed` refuses a
  `tau_e` or `t_rep` that is not positive and finite, a negative or
  non-finite `sigma_pair` and a non-finite `center_off` (test
  `test_pulsed_simulator_refuses_bad_shapes`). The photon-by-photon
  simulator refuses a NaN dead time (`DetectorImpairments`), with which
  0.12.1 dropped every click.
- Type hints on the public functions touched.

### Performance

All with the same results; the tests keep the 0.12.1 loops and compare
count for count. Measured on README example 7's tags (375 000 per
channel), median of three interleaved series of seven runs each, on
the same machine:

- `correlate`: pairs formed without a Python loop over the start tags,
  in blocks of about a million pairs: 107.5 ms -> 47.2 ms (2.3 times).
- `correlate_start_stop`: 103.0 ms -> 45.8 ms (2.2 times).
- The detectors' dead-time pass in the photon-by-photon simulator
  (`DetectorImpairments`): only clicks closer than the dead time to
  the previous click are visited: 85.2 ms -> 1.1 ms (80 times) at
  45 ns. The simulator as a whole is still dominated by drawing the
  emission times.
- Tests: the brute-force convolutions in
  `test_complex_blur_matches_real_formula_and_numerics` are computed by
  FFT (the same numbers to 1e-15 at the compared points): 350 s ->
  0.15 s on the loaded test machine.

### Tutorials

- `examples/tutorial_saturation.py` (about 2 s): counts at 15 powers to
  R_inf and P_sat with 95 % intervals, the background and dark rates,
  the signal fraction, and the open intervals of data far below
  saturation.
- `examples/tutorial_timetags.py` (about 15 s): simulated clicks with
  detector jitter and dead time, through a PicoQuant PTU file, to the
  histogram, the exact window-g2 interval and test, the Bayesian
  probability and a profile-likelihood interval for g2(0).
- Their printed output is quoted at the top of each script and in the
  README ("Tutorials"); `tests/test_tutorials.py` runs both and checks
  the numbers (to 1 %).

### Tests

- 225 tests (208 before): test_release_013 (15) and test_tutorials (2).
  test_metadata also imports `sparq.saturation` with PyTorch blocked.

## 0.12.1 (2026-10-15)

The SPARQ paper is published, and the citation now points to it.

### Changed

- Citation: the paper "Closed-Loop, Event-Driven Machine Learning for
  Autonomous Triage of Single-Photon Emitters" (T. M. Mahim, M. N.
  Islam, M. M. Rahman and A. S. M. Mohsin) was published online in
  *ACS Photonics* on 15 October 2026,
  doi:10.1021/acsphotonics.6c01984. `CITATION.cff` and the README now
  give the DOI and the journal's title capitalization instead of
  "accepted, in production, no DOI yet". Volume and pages are not yet
  assigned. No code changed.
- Packaging: the license is now given as the SPDX expression
  `license = "Apache-2.0"` with `license-files = ["LICENSE"]`, and the
  old license table and license classifier are gone. Building no longer
  prints setuptools' deprecation warnings (their support ends on
  2027-02-18). Building from source now needs setuptools 77 or newer;
  installing the package is unchanged.

## 0.12.0 (2026-10-10)

Corrects the profile-likelihood interval, adds a stationarity check
for a series of snapshots and a measured signal fraction with an
interval that the background correction can carry, and refuses more
inputs that gave NaN.

### Added

- `check_stationarity` (`sparq.sequential`, exported from `sparq`):
  three Pearson chi-square tests on a series of histograms of one
  site, conditional on totals so that no emitter model is needed:
  totals in proportion to the snapshot times (rate, dof J - 1), the
  same window g2 in every snapshot (the windows of `bayesian_g2`, a
  2 x J homogeneity test), and the same histogram shape up to a scale
  (J x K homogeneity test with neighboring bins merged until every
  expected count is at least `min_expected`, default 5). Tests with
  smaller expected counts are refused. `consistent` is False when any
  p-value is below alpha/3 (Bonferroni, alpha = 0.01 by default).
  Checked against SciPy's `chisquare` and `chi2_contingency`, for
  uniform p-values on 200 simulated steady series, and for detection
  of a second emitter (window p < 1e-6) and of a 20 % dimmer source
  (rate p < 1e-6, window and shape p > 1e-3). README example 19.
- `signal_fraction_from_counts` (`sparq.analysis`, exported from
  `sparq`): rho = 1 - (n_off/t_off)/(n_on/t_on) from counter readings
  on and beside the emitter, with the Clopper-Pearson interval
  (Clopper and Pearson, Biometrika 26, 404 (1934)) of the binomial
  share of background counts mapped to rho. Checked against
  Clopper-Pearson limits solved from binomial tail sums (1e-9) and
  for coverage at or above the level on 2000 simulated readings.
- `background_corrected_g2(..., rho_ci=(lo, hi))`: the corrected
  interval over both the g2 interval and a rho interval (the box
  corners, since the correction is monotone in each), with the
  Bonferroni statement that two 97.5 % intervals give at least 95 %.
  Checked against a 201 x 201 grid of the box and by a coverage
  simulation. Without `rho_ci` the result is unchanged. README
  example 20.

### Fixed

- `profile_likelihood_ci`: the nuisance parameters were minimized in
  natural units, where L-BFGS-B left tau2 at about its 250 ns start
  value; the deviance was measured from the smallest grid value
  (grid step 0.025) and the ends were interpolated linearly; g2_hat
  was a grid point. On the data of README example 3 (seed 0, 60 s):
  with `c0_prior=(1.0, 0.01)` [0.0839, 0.1261] -> [0.0784, 0.1243],
  g2_hat 0.1000 -> 0.1008; without it [0.0670, 0.1815] ->
  [0.0639, 0.2948], g2_hat 0.0750 -> 0.0819. An independent
  Nelder-Mead profile put the deviance at the old ends at 2.15 / 4.45
  and 2.63 / 1.16 instead of 3.84; at the new ends it is 3.84 within
  0.03 (test `test_profile_interval_ends_sit_at_the_chi_square_quantile`).
  The profile now runs over log tau1, log tau2, a and log c0 with the
  objective written as half the Poisson deviance, and g2_hat and the
  ends are refined between grid points (to about 1e-4).
- `SPRTCertifier.update` accepted a NaN time step (the evidence became
  NaN and the test never decided) and negative or non-finite counts;
  `deadtime_corrected_rate`, `signal_fraction` and
  `background_corrected_g2` returned NaN for NaN inputs; `fit_model`
  accepted an infinite `r_hat` and a `c0_prior` with an infinite sd
  (the prior was then silently ignored). All are now refused with
  ValueError (test `test_other_new_input_checks`).

### Changed

- `profile_likelihood_ci` refuses a `level` outside (0, 1) and fewer
  than 3 grid points, and reports a malformed `c0_prior` as ValueError
  (it was a TypeError for a single number). It takes two to three
  times as long as before (about 10 s for README example 3), and its
  RuntimeWarning about search limits now appears where the profile
  really ends on a limit (for README example 3 with the flat level
  pinned: tau2 at 800 ns; with 8000 ns the interval moves by 0.001).
- Docs: README example 3 shows the corrected intervals and the search
  limit check; example 11 the output with NumPy 2.5.3 and SciPy 1.18.1
  (the sampler starts from a SciPy fit, and its output differs in the
  last digit between versions; the 0.11.0 code gives the same numbers
  here); the README and the docstring of `fit_model` say that its
  local search can stop short of the best fit of a badly fitting
  model by a SciPy-dependent amount (example 10: deviance 350.16 with
  SciPy 1.18, 345.75 with SciPy 1.10); the paper is described as
  accepted in ACS Photonics, in production, with no DOI yet, and
  CITATION.cff lists it as the preferred citation (status in-press).
  The docstring of `sparq.twin_torch` gave a(s) = a_1 s where the code
  uses 2 a_1 s/(1 + s); the docstring now matches the code. The
  docstring of `sparq.timetags` no longer says that no PTU reader
  exists.
- Type hints on the public functions touched.

### Tests

- 208 tests (195 before), the new ones in test_release_012.

## 0.11.0 (2026-09-30)

Pulsed histograms are now analyzed by an exact fit of the whole comb of
peaks, the fast simulator models detector afterpulsing, and several
inputs that gave wrong or cryptic results are refused.

### Added

- `fit_pulsed` (`sparq.pulsed`): Poisson maximum-likelihood fit of a
  pulsed histogram with the model `expected_hist_pulsed` simulates
  (center peak g2_0 A, side peaks A, optional blinking envelope, flat
  floor), with each peak's share of each bin computed exactly
  (`peak_cdf`, the distribution function of the jitter-blurred
  two-sided exponential). Overlapping peaks are no problem. Returns
  g2(0) with a profile-likelihood interval, the fitted peak shape,
  and a goodness-of-fit p-value. Between the peaks the bins hold few
  counts, where the usual chi-square law of the deviance does not
  hold, and how far it is off depends on the data: for 100 simulated
  histograms (80 MHz, 484 bins of 0.25 ns, 10 s at 200 kcps, true
  g2(0) 0.12, 1 ns lifetime, 0.5 ns jitter, up to 109 counts per bin
  and 90 bins with fewer than 1, NumPy seed 123) the chi-square law
  put 21 of 100 p-values below 0.05. The p-value therefore uses the
  exact mean and variance of each bin's Poisson deviance: 6 of the
  same 100, and 30 of 500 runs of this kind (seeds 7 and 99), fell
  below 0.05. The moments are summed exactly for bins with a mean of
  up to 1000 counts, in blocks of bounded size, and above that taken
  from the expansions 1 + 1/(6m) + 1/(6m^2) and 2 + 2/(3m) + 4/(3m^2),
  whose coefficients were read off the exact sums and which agree
  with them to 1e-8 from m = 1000 to 1e5 (tested). Optional:
  `sigma_pair` and `center` when known, `envelope=True` for a
  blinking envelope, and `T_s` with `singles_cps` to fix the
  side-peak area from the detectors' count rates. In simulated runs
  (done once, outside the test suite) the 95 % intervals covered the
  truth in 91 % to 94 % of runs.
- `analyze_pulsed(method=...)`: "fit" (new default, see Behaviour
  changes) or "area" (the 0.10.0 method); `envelope`, `T_s`,
  `singles_cps`, `sigma_pair` are passed to the fit.
- `expected_histogram(afterpulse_p=, afterpulse_tau_ns=)`:
  afterpulsing in the fast simulator, as in the photon-by-photon
  detector chain (each counted click is followed with probability p
  by one click dead_time + Exp(afterpulse_tau) later). The
  afterpulse terms are closed forms (the site's g2 written as a sum of
  exponentials, also for blinking sites). They fill the dip by about
  2p and add shallow copies of it near +-dead time. The detectors'
  jitter is kept in the direct term and left out of the afterpulse
  terms (a difference below 5e-4 of the flat level, times p, in the
  test).

### Fixed

- `analyze_pulsed` and `g2_peak_area` (the peak-area method)
  subtract a "floor" measured more than 4.5 ns from every peak. When
  the peaks' tails reach there, that floor holds peak tails and g2(0)
  comes out too low: noise-free data at 80 MHz with true g2(0) = 0.12
  gave 0.1035 for a 1 ns lifetime and 0.0000 for a 2 ns lifetime (a
  "perfect" emitter). The new default fit gives 0.1200 for both.
  `g2_peak_area` itself is unchanged except as below.
- `g2_peak_area` at repetition periods below 9 ns (no delay 4.5 ns
  from every peak) returned NaN with NumPy "Mean of empty slice"
  warnings; it now raises ValueError that points to the fit.
- `calibrate_comb` on records with at most four peaks excluded every
  peak and returned the first one as the zero-delay peak; it now
  leaves out only peaks cut by the record's ends.
- `analyze_histogram`: NaN counts gave a TypeError from deep inside the
  fit, `T_s = 0` a ZeroDivisionError, and a negative `T_s` returned
  g2_0 = 1.0 with ok = False and no error; non-finite delays or
  counts and a non-positive or non-finite `T_s` now raise ValueError.
  `analyze_pulsed` returned g2_0 = NaN for NaN counts; it now raises.
- `fit_g2_histogram` returned (1.0, False) and `profile_likelihood_ci`
  raised a broadcasting error for a histogram whose length is not
  `cfg.n_bins`; both now raise ValueError naming the expected length,
  and refuse negative or non-finite counts and a non-positive or
  non-finite `T_s` or `r_hat`. Valid inputs are passed on unchanged
  (the histogram keeps its data type), so their fits do not change.

### Behaviour changes

- `analyze_pulsed` now fits the comb by default. Before/after on
  noise-free 80 MHz data with true g2(0) = 0.12: 0.1035 -> 0.1200
  (1 ns lifetime), 0.0000 -> 0.1200 (2 ns). On the seeded noisy data of
  `test_pulsed_analysis_recovers_truth` (1 ns, 20 s): 0.0936 (68 %
  bootstrap interval 0.0877 to 0.1037) -> 0.1076 (68 % profile
  interval 0.1007 to 0.1147). With a blinking envelope (a = 0.5,
  tau2 = 50 ns, noise-free) the area method gave 0.0776, the default
  fit (side peaks taken as equal) 0.0943 and the fit with
  `envelope=True` 0.1200. The fit takes about a second, and the
  interval a few seconds more; `method="area"` gives the 0.10.0
  numbers. For the fit, `g2_0_std` is NaN (a profile interval has no
  standard deviation), `n_bootstrap` only switches the interval on
  (> 0) or off, and `seed` is not used.
- The refusals listed under Fixed replace the earlier outputs quoted
  there.

### Tests

- 195 tests (172 before). New files: test_pulsed_fit (peak_cdf
  against a numerical integral of `peak_shape` to 1e-7 and the
  Laplace law to 1e-15; the exact bin integrals against a
  sixteen-fold finer midpoint sum of `expected_hist_pulsed` to 2e-5 of
  the largest bin; noise-free recovery of g2(0) to 1e-4 at 80 and
  160 MHz, with background, and with the envelope; the profile
  interval's ends checked by a separate Nelder-Mead minimization, to
  0.03 in deviance; the deviance moments against 200 000 simulated
  counts and against direct sums to 1e-8, and a 65 536-bin histogram
  with peaks of 1e5 counts within 100 MB), test_afterpulsing (closed forms against numerical
  integrals to 1e-12; the photon-by-photon simulator with 20 %
  afterpulsing passes the chi-square test with the new terms, p >
  1e-3, and fails without them, p < 1e-6, even with the level scaled
  to the counted rates) and test_validation_011.

## 0.10.0 (2026-09-24)

Works through the limits listed in 0.9.1, and corrects several results
that were wrong without an error.

### Added

- Other dip shapes: `sparq.models` with `fit_model` (exact Poisson
  likelihood, goodness-of-fit p-value, AIC), `goodness_of_fit`
  (p-value by simulation), `g2_model` and `exp_conv_gauss_complex`.
  Models: "three_level", "multi_exponential" (several shoulders) and
  "coherent" (a two-level emitter driven by a laser, from its exact
  Lindblad master equation).
- `analyze_histogram`: `singles_cps` (the detectors' count rates fix
  the flat level; with them, 15 simulated runs per case were all
  within 0.06 of the truth), `fit_p_value`, `fit`, `at_bound`,
  `g2_0_widened`, `bound_matters` (with a RuntimeWarning), and
  `rebin`. `fit_g2_histogram`: `return_details`, `c0_prior`,
  `check_bounds`, `var`. Search ranges may be "auto" in
  `analyze_histogram`, `fit_g2_histogram` and `profile_likelihood_ci`;
  `profile_likelihood_ci` reports `at_bound`.
- `locate_dip`: the dip position with its significance and whether
  another place is about as significant; `analyze_histogram` returns
  `dip_ambiguous` and warns when the dip does not stand out.
- `rebin_real(method="split")`, the new default: proportional sharing
  of input bins, the dip center found to a fraction of a bin by
  symmetry, and `return_variance`.
- `HBTConfig.bin_average` (default True) and `HBTConfig.bin_nodes`:
  the simulator and the fits average g2 over each bin.
- `bayesian_g2_model` / `ModelPosterior` (`sparq.posterior`): the
  posterior of g2(0) itself under the three-level model, with the
  flat-level scale integrated out exactly, sampled with an
  affine-invariant ensemble sampler (`stretch_sampler`).
- `certification_probability` and `assured_acquisition_time`: exact
  planning over the Poisson scatter of runs.
- `WindowSPRT`: a sequential test on the window counts, conditional on
  their total, whose error bounds hold for any count rate and over
  whole ranges of g2.
- `heralded_source` and `heralded_from_car`: exact CAR and heralded
  g2(0) for click/no-click detectors with efficiencies, dark counts
  and any number of thermal modes; `modes=` and `car_definition=` in
  `heralded_g2_limit` and `car_for_purity`.
- `sparq.ptu`: `read_ptu` (all twelve PicoQuant T2/T3 record types),
  `load_ptu_timetags` and `save_ptu_t2`.
- Fast simulator: exact blinking (`site_g2`), `dead_time_ns` in
  `expected_histogram` with `deadtime_throughput` (dead-time losses
  for correlated light), `mean_detected_rate_cps`.
- `rates_for_params`: the exact rates for a given (tau1, tau2, a).

### Fixed

- The photon-by-photon simulator's g2 was not the site's (approximate
  rate mapping; for tau1 = 8, tau2 = 100, a = 1.5 the shelving time
  came out 43 ns). It now uses `rates_for_params`. It also thins the
  photons block by block (it used to keep every emitted photon in
  memory first) and
  draws blinking periods in blocks (microsecond blinking used to take
  one Python step per period). Its random stream therefore differs
  from 0.9.1.
- The fast simulator's blinking added a flat level at every delay,
  filling the dip, and switched the background off with the emitter.
  Now exact, and consistent with the photon-by-photon simulator.
- The simulator and the fits used g2 at bin centers; real bins hold
  the average (noise-free fit on 1 ns bins: 0.086 instead of 0.0975).
- Re-binning with whole input bins gave a ripple and an off-center
  zero-delay bin (still available as `method="whole"`).
- `heralded_g2_limit(car, "thermal")` used the net CAR while the
  Poissonian branch used the raw one; both now use the raw CAR C/A.
  Values for "thermal" change (at CAR 10: 0.3800, was 0.3471);
  `car_definition="net"` gives the 0.9.1 values. The documentation
  no longer calls the formulas lower limits.
- `_exp_conv_gauss` (the jitter blur used by the simulator and the
  fits) lost all precision for lifetimes below about a seventh of the
  pair jitter and returned NaN far from zero delay; it is now computed
  with `scipy.special.erfcx`.
- `rebin_real` left grid bins empty or partly empty, silently, when
  the analysis window around the dip reached beyond the recorded
  delays; this is now refused with the largest `tau_max` that fits.
  The dip is located as the most significant local minimum over a
  range of widths and refined by symmetry (the 0.9.1 smoothed minimum,
  still used by `method="whole"`, missed narrow dips inside bunching
  peaks), and the bootstrap of `analyze_histogram` locates it anew in
  every copy.
- The README's accuracy claims for `analyze_histogram` came from one
  seeded run each; they now come from many runs.
- `train_model` and `DiscreteSAC.update` no longer warn when turning
  losses into numbers.

### Changed behaviour to note

- `heralded_g2_limit(car, "thermal")` refuses a raw CAR below 2 (a
  single-mode source cannot give one); 0.9.1 accepted any CAR >= 1.
- `simulate_photon_stream` refuses sites for which no three-level rates
  exist (for example `tau2` <= `tau1` with `a` > 0), which 0.9.1
  simulated with its approximate rates.
- `analyze_histogram` and `rebin_real` refuse analysis windows that
  reach beyond the data (see Fixed).
- `analyze_histogram`'s default re-binning, the bin averaging and the
  per-copy dip location change its results slightly for the same data;
  `rebin="whole"` and `HBTConfig(bin_average=False)` give the 0.9.1
  re-binning and model values (but also refuse windows beyond the
  data).

### Tests

- 172 tests (88 before). New files: test_simulators, test_analysis_010,
  test_models, test_posterior, test_assurance, test_window_sprt,
  test_heralded_exact, test_ptu, test_ml_behaviour.
- test_cw_analysis_flags_a_pair now passes the singles rates (the
  histogram-only fit it used passed for its seed only); the heralded
  tests use the raw-CAR thermal formula.
- CI installs the independent PTU readers `ptufile` and `phconvert`
  where they install; the PTU cross-checks skip without them.

## 0.9.1 (2026-09-22)

Bug fixes for three inputs that gave wrong results without an error,
tests that run on the oldest allowed NumPy, and a rewritten README.

### Fixed

- `correlate` searches the stop tags by bisection, which needs them
  sorted, but did not check. An unsorted `t_b` gave a histogram with
  most pairs missing (9 pairs instead of 376 on the new test's data).
  It is now refused with ValueError. The order of `t_a` does not matter and is
  still accepted.
- `rebin_real`, and so `analyze_histogram`, accepted data binned more
  coarsely than the analysis grid. Some grid bins stayed empty and the
  fit returned a wrong g2(0) with `ok=True` (about 0.30 for a
  noise-free histogram with 2 ns bins, whose true g2(0) is 0.0975).
  Such data are now refused with ValueError; a coarser `cfg` accepts
  them.
- `load_timetags_csv` read a one-column file with exactly two rows as
  one (channel, time) tag. It is now refused like other files without
  two columns.
- Docstrings: `correlate_start_stop` keeps delays in
  [-tau_max, +tau_max), not (-tau_max, +tau_max]; `make_batch`
  returns 5 auxiliary values per acquisition, not 3.

### Tests

- New: `test_correlate_refuses_unsorted_stop_channel`,
  `test_timetag_file_with_one_column_is_refused`,
  `test_input_grid_coarser_than_analysis_grid_is_refused`
  (88 tests in total, up from 85).
- `test_peak_shape_has_unit_area` used `np.trapezoid`, which exists
  only from NumPy 2.0, although the package allows NumPy 1.24; it now
  falls back to `np.trapz`. The package code itself did not use it.
- The start-stop test now also asserts that the start-stop histogram
  is not above the all-pairs histogram in any bin. The 0.7.0 notes,
  the test module docstring and the `correlate_start_stop` docstring
  already said "elementwise", but the test only compared totals.
- CI: Python 3.14 added to the main matrix (with CPU PyTorch), and a
  new `oldest-dependencies` job runs the suite on Python 3.10 with
  NumPy 1.24.0, SciPy 1.10.0 and PyTorch 2.0.0 (from PyPI), which pass.

### Changed

- README rewritten for readers outside the field: a guide to the
  terms, units, eight examples with their printed output, the full
  list of refusals, and each test-backed claim with the tolerance the
  test uses. Claims the tests do not support were removed (for
  example, that uncorrelated light normalizes to "exactly" 1: the test
  checks agreement within the error bars).
- Classifiers now list Python 3.13 and 3.14, which CI tests.
- CONTRIBUTING: the suite takes minutes, not seconds.

### Notes on earlier entries

- 0.8.0 says the planner is "bitwise consistent" with `bayesian_g2`;
  the test compares within 1e-15 (probability) and 1e-12 (interval),
  not bit for bit.
- 0.9.0 lists "monotonicity" among the anchors; only the
  Poissonian (laser-pumped) formula is tested for it. The heralded
  test module docstring said CAR = 1 gives 1 "in both conventions";
  the test (correctly) asserts 1 and 3/2, and the docstring is fixed.

## 0.9.0 (2026-09-18)

Heralded sources, and a future-proofing pass.

- `heralded.heralded_g2_limit` / `car_for_purity`: the exact purity
  limit a measured coincidence-to-accidental ratio permits --
  g2_h(0) = (2 CAR - 1)/CAR^2 for Poissonian (laser-pumped) pair
  statistics and (4 CAR + 2)/(CAR + 1)^2 for thermal -- and the
  exact closed-form inversions (Wang et al., arXiv:2404.03236).
  Stated as floors set by the pair statistics alone; real sources
  sit at or above them.
- CI gains a torch-free Python 3.14 job, so the dependency-light
  core stays ahead of upstream ML wheels.
- Anchors: both closed forms verified against their defining
  quadratics as an independent algebraic path; inversions exact to
  machine precision; the CAR = 1 boundaries exactly 1 and 3/2; the
  2/CAR and 4/CAR falloffs; monotonicity; refusals for
  uncertifiable inputs.

## 0.8.0 (2026-09-17)

Lab adaptability: the acquisition planned before it is run.

- `lab.site_from_numbers`: an `EmitterSite` from plainly named lab
  numbers (lifetimes in ns, count rate in kcps, signal fraction).
- `lab.expected_posterior`: the exact closed-form Bayesian verdict
  (`bayesian_g2`) evaluated on the site's average histogram
  (`expected_histogram`) -- typical-data planning, stated as such.
- `lab.required_acquisition_time`: the shortest run whose typical
  data certify the site at a chosen confidence, by bisection; a site
  whose window-averaged g2 is not below the threshold is refused by
  the exact time-independence of the window ratio, with the number
  named, and an unreachable confidence within t_max is refused with
  the probability actually attained there.
- Anchors: the window ratio identical at 1 s and 1000 s to 1e-12; the
  planner bitwise consistent with the two public code paths it
  composes; the returned time bracketed on both sides of the
  confidence; the four-emitter site refused; 200 seeded Monte-Carlo
  runs at a 4x margin certifying > 90%.

## 0.7.0 (2026-09-13)

From-the-hardware release: raw photon time tags in, normalized g2
with error bars out, both correlator conventions labs actually use.

### Added

- `load_timetags_csv` / `save_timetags_csv`: exact two-column
  (channel, time_ns) tag-list contract with a bit-for-bit round trip,
  channel remapping for your numbering, and refusals for non-finite
  entries, unknown channels and empty files. No proprietary binary
  formats are parsed on purpose: a PTU reader written without the
  vendor spec would be a guess.
- `normalize_g2`: the standard accidental normalization
  g2(tau) = C(tau) T / (N1 N2 w) with per-bin Poisson one-sigma
  error bars and the accidental level reported explicitly (Brouri
  et al., Opt. Lett. 25, 1294 (2000); Fox, Quantum Optics (2006),
  ch. 6). Anchors: two independent Poisson streams read g2 = 1
  within the reported errors with reduced chi-square near 1; a
  simulated single emitter's tags normalize to a clear antibunching
  dip with wings at 1.
- `correlate_start_stop`: the classic single-stop TAC/TCSPC
  convention (first stop per start), for emulating hardware
  histograms. Anchors: exact equality with the all-pairs `correlate`
  on streams too sparse for two stops per window; elementwise
  undercount (pile-up) on dense streams, never inventing pairs;
  at most one count per start.
- The existing all-pairs `correlate` is now pinned against an
  independent brute-force O(N^2) double loop, exactly.

## 0.6.0 (2026-09-12)

Adaptability release: the analysis pipeline no longer assumes an
NV-scale emitter or a negligible instrument response.

### Added

- `load_hbt_csv` / `save_hbt_csv`: documented `delay_ns,counts` file
  contract for measured histograms -- exact round trip, refusals for
  wrong headers, field counts, negative counts and non-increasing
  delay axes.
- `t1_bounds` / `t2_bounds` on `analyze_histogram`,
  `fit_g2_histogram` and `profile_likelihood_ci`: the lifetime
  windows the fits search, defaulting to the historical NV-scale
  window, validated on entry. A fast-emitter test (tau1 = 0.15 ns,
  tau2 = 5 ns) demonstrates the default-window failure and the
  custom-window recovery.
- `c0_prior` on `profile_likelihood_ci`: an optional Gaussian
  constraint on the flat-level normalization, representing the
  independently measured singles rates. Without it the interval is
  honestly wide -- a slow bunching shoulder trades against the
  normalization, a real near-degeneracy of the histogram alone,
  probed explicitly during development and now documented.

### Changed

- The LM fit and the profile likelihood now use the physical
  parameterization (dip depth rho^2 multiplying both exponentials,
  g2(0) = 1 - rho^2). The previous 1 - d e1 + a e2 form capped d at
  1, which biased g2(0) upward for any emitter with a strong
  bunching shoulder, and hid a (d, a) ridge behind the cap.
- Both fit models now include the instrument response (closed-form
  Gaussian-convolved exponentials at `cfg.sigma_irf`), so the
  estimand is the IRF-free g2(0); a test separates the two where the
  IRF matters.
- `load_fisequr` requires the dataset directory explicitly (the
  previous default was one machine's path) and raises with an
  explanation when it is missing.

## 0.5.0 (2026-09-10)

### Added

- `bayesian_g2` / `G2Posterior`: exact-Poisson Bayesian posterior of
  the raw central-window g2 ratio. Conjugate Gamma updates on the
  central and reference per-bin rates (Jeffreys prior by default)
  give a closed-form scaled beta-prime posterior for the ratio:
  exact density, CDF, quantiles, equal-tailed credible intervals,
  posterior mean/median/mode, and the triage verdict probability
  P[g2 < 1/2 | data]. Acquisition time and detection rates cancel in
  the ratio, so none are needed. The estimand (window-averaged raw
  g2, an upper bound on g2(0)) is stated plainly; dip-shape-aware
  inference remains `profile_likelihood_ci`.
- Anchors: hand-built law against `scipy.stats.betaprime` to 1e-12;
  unit mass and closed-form moments by quadrature; agreement with a
  direct numerical marginalization of the exact Poisson likelihood
  over the nuisance rate; seeded Monte-Carlo Gamma-ratio CDF
  agreement; nominal credible-interval coverage on twin-generated
  histograms; verdict ordering of single- versus two-emitter sites.

## 0.4.0 (2026-09-05)

### Added

- `background_corrected_g2`: exact inversion of the Poissonian-
  background map g2_meas = 1 + rho^2 (g2_true - 1) (Brouri et al.,
  Opt. Lett. 25, 1294 (2000)) -- the same forward model as the
  package's `g2_zero(..., rho)`, so the round trip is machine-exact
  and asserted in the tests. Maps confidence-interval endpoints
  through the same affine transform, truncates at the physical floor
  while returning the untruncated value, refuses rho outside (0, 1].
- `signal_fraction`: rho = S/(S+B) with input validation.
- `deadtime_corrected_rate`: exact inversion of the non-paralyzable
  dead-time throughput r_meas = r/(1 + r tau_d), validated against
  the Monte-Carlo detector chain statistically; refuses rates at or
  beyond saturation instead of extrapolating.

### Changed

- CI matrix: Python 3.10, 3.11, 3.12, 3.13.

Earlier versions: see the release notes on
https://github.com/TaN-MM-Org/sparq-triage/releases
