# Changelog

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
