# Changelog

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
