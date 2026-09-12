# SPARQ

[![Tests](https://github.com/TaN-MM-Org/sparq-triage/actions/workflows/tests.yml/badge.svg)](https://github.com/TaN-MM-Org/sparq-triage/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/sparq-triage?label=PyPI&color=blue)](https://pypi.org/project/sparq-triage/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22278040-blue)](https://doi.org/10.5281/zenodo.22278040)

Is this light source emitting one photon at a time? That question --
the standard test for single-photon emitters, measured as a dip in
the coincidence histogram of a two-detector (HBT) experiment --
usually costs a long dwell at every candidate site. `sparq` answers
it faster and more rigorously: exact statistical analysis of measured
histograms with honest uncertainties, sequential tests that stop the
moment the data suffice, and a machine-learning layer that learns to
triage many sites efficiently. The installable package behind the
manuscript *"Closed-loop, event-driven machine learning for
autonomous triage of single-photon emitters"*; the
[companion repository](https://github.com/Tanvir-Mahmud-Mahim/a-spiking-RL-triage-of-solid-state-single-photon-emitters)
reproduces the paper itself.

## Install

```bash
pip install sparq-triage        # physics core (numpy, scipy)
pip install sparq-triage[ml]    # adds PyTorch for the estimators, twin and RL
```

The core -- analytic correlation functions, the exact-statistics
histogram twin, master-equation reference, pulsed analysis, and every
analysis tool below -- imports without PyTorch.

## Analyze your own measurement

`analyze_histogram` runs the full conventional pipeline on any
measured CW HBT histogram (dip centering, re-binning, normalization,
multi-start fit) and reports g2(0) with a bootstrap confidence
interval that propagates shot noise through every step;
`analyze_pulsed` does the same for pulsed data via peak areas.

The pipeline adapts to *your* emitter and instrument (v0.6) instead
of assuming the defaults it was born with:

- `load_hbt_csv` / `save_hbt_csv`: a documented two-column file
  contract (`delay_ns,counts`) with an exact round trip and refusals
  for malformed files.
- Configurable lifetime windows (`t1_bounds` / `t2_bounds`): the
  defaults suit NV-center-scale emitters; a much faster quantum dot
  or a slow bunching shoulder needs its own window, and a window that
  excludes the truth silently rails a fit -- the tests demonstrate
  both the failure and the recovery.
- The fit model is the physical parameterization (dip depth
  multiplying both exponentials) and includes the instrument response
  (closed-form Gaussian-convolved exponentials at `cfg.sigma_irf`),
  so the estimate targets the true g2(0), not the softened dip the
  raw histogram shows.
- `register_platform` makes your emitter a first-class citizen
  everywhere a platform name is accepted; the built-in priors (NV,
  hBN, GaN, SiV) are literature-anchored defaults, not a limit.

## Four routes to an uncertainty, each honest about what it assumes

- **Bootstrap** (`analyze_histogram`): shot noise propagated through
  the full pipeline.
- **Profile likelihood** (`profile_likelihood_ci`): a
  Wilks confidence interval from the exact Poisson likelihood, honest
  at low counts where linearized errors are not. With the
  normalization free, a slow bunching shoulder trades against the
  flat level -- a real near-degeneracy of the histogram alone, and
  the interval says so by being wide; pass the independently measured
  singles rates as `c0_prior` and it tightens to what the data
  genuinely support.
- **Exact Bayesian posterior** (`bayesian_g2`): the closed-form
  finite-count posterior of the raw central-window g2 -- density,
  quantiles, credible intervals and the verdict probability
  P[g2 < 1/2 | data] as exact expressions in four counts, no
  sampling, no asymptotics. The estimand (a window average, an upper
  bound on g2(0)) is stated plainly.
- **Sequential certification** (`SPRTCertifier`): Wald's sequential
  test on accumulating histograms -- acquisition stops the moment the
  evidence crosses the chosen error rates, certifying bright sites in
  a fraction of the fixed dwell.

Exact closed-form corrections complete the toolbox:
`background_corrected_g2` inverts the Poissonian-background map (the
same forward model the package's own twin applies, so the round trip
is machine-exact), `signal_fraction` builds its input from measured
rates, and `deadtime_corrected_rate` inverts detector dead-time,
refusing rates at or beyond saturation instead of extrapolating.

## The machine-learning layer (optional)

With the `[ml]` extra: neural estimators (CNN and spiking network)
trained physics-in-the-loop, a differentiable twin of the whole
measurement protocol, a closed-loop triage environment with a soft
actor-critic agent and prioritized replay, and a graph encoder for
cross-platform transfer. All of it consumes the same exact-statistics
histogram twin the physics core provides.

## How it is checked

67 tests (Python 3.9-3.13, ML tests skip without torch, run in CI on
every push), each pinned to an exact reference: the two-exponential
correlation law against the master-equation eigendecomposition; the
closed-form instrument-response convolution against brute-force
quadrature; Poisson statistics of the histogram twin; the Bayesian
posterior against SciPy's independent implementation and a direct
numerical marginalization; exact background and dead-time correction
round trips; the fast-emitter window failure and recovery; profile
intervals covering the truth with and without the singles-rate prior;
exact file-contract round trips; and the shape and gradient contracts
of the ML components.

## Real data

The experimental quantum-dot HBT measurements used by
`sparq.datasets.load_fisequr` are from the openly licensed
[sps-quality](https://github.com/UTS-CASLab/sps-quality) repository
(Kedziora et al., *Mach. Learn.: Sci. Technol.* **4**, 045042
(2023)); they are not redistributed here, and the loader takes the
dataset directory explicitly.

## Contributing and support

Bug reports, questions and pull requests are welcome through
[GitHub issues](https://github.com/TaN-MM-Org/sparq-triage/issues);
see [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and
the design rules. Tagged releases are published to PyPI by CI.

## License and citation

Apache-2.0 (see LICENSE). Please cite the associated paper if you use
this code; citation metadata is in [CITATION.cff](CITATION.cff).
Every release is archived on Zenodo under the concept DOI
[10.5281/zenodo.22278040](https://doi.org/10.5281/zenodo.22278040),
which always resolves to the latest version.
