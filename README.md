# SPARQ

[![Tests](https://github.com/TaN-MM-Org/sparq-triage/actions/workflows/tests.yml/badge.svg)](https://github.com/TaN-MM-Org/sparq-triage/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/sparq-triage?label=PyPI&color=blue)](https://pypi.org/project/sparq-triage/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22278041-blue)](https://doi.org/10.5281/zenodo.22278041)

**S**piking **P**hysics-in-the-loop **A**utonomous **R**einforcement triage
of **Q**uantum emitters: the installable `sparq` package behind the manuscript
*"Closed-loop, event-driven machine learning for autonomous triage of
single-photon emitters."* The manuscript's companion repository,
[a-spiking-RL-triage-of-solid-state-single-photon-emitters](https://github.com/Tanvir-Mahmud-Mahim/a-spiking-RL-triage-of-solid-state-single-photon-emitters),
holds the experiment scripts, figure scripts and results that reproduce the
paper; this repository is the software's home for development, releases and
support.

## Installation

```bash
pip install sparq-triage        # physics core (numpy, scipy)
pip install sparq-triage[ml]    # adds PyTorch for the estimators, twin and RL
```

The core (`sparq.physics`, `sparq.exact`, `sparq.pulsed`) imports without
PyTorch: the analytic HBT correlation functions, the exact-statistics
histogram twin, the master-equation reference and the pulsed comb analysis.

```python
import numpy as np
from sparq import HBTConfig, sample_site, expected_histogram

rng = np.random.default_rng(0)
site = sample_site(rng, platform="NV")     # literature-anchored priors
mu = expected_histogram(site, T_s=5.0, cfg=HBTConfig())
print(site.g2_0, site.is_good, mu.shape)
```

## Analyzing your own data

`analyze_histogram` runs the full conventional pipeline on any measured CW
HBT histogram (dip centering, re-binning, flat-level normalization,
multi-start fit) and reports g2(0) with a parametric-bootstrap confidence
interval that propagates shot noise through every analysis step;
`analyze_pulsed` does the same for pulsed combs via the peak-area method.
Neither needs PyTorch.

```python
from sparq import analyze_histogram
res = analyze_histogram(delay_ns, counts, T_s=30.0, n_bootstrap=200)
print(res["g2_0"], (res["g2_0_low"], res["g2_0_high"]),
      res["single_emitter_confident"])
```

## Sequential certification and rigorous intervals

`SPRTCertifier` implements Wald's sequential probability ratio test on
accumulating HBT histograms with exact Poisson log-likelihoods: acquisition
stops the moment the evidence crosses the error-rate thresholds, which on
twin benchmarks certifies bright sites in a fraction of a second instead of
a fixed 30 s dwell, at the nominal error rates. `profile_likelihood_ci`
gives a Wilks profile-likelihood confidence interval for g2(0) from the
exact Poisson likelihood, honest at low counts where linearized fit errors
are not. Both are torch-free; the plug-in-hypothesis caveat and the
empirical validation are documented in the module.

```python
from sparq import SPRTCertifier, profile_likelihood_ci
cert = SPRTCertifier(site_single, site_pair, alpha=0.05, beta=0.05)
while cert.update(new_counts, dt) == "continue":
    ...                                   # keep acquiring
print(cert.decision, cert.T_total, cert.expected_times())
```

## Exact closed-form corrections (new in v0.4)

`background_corrected_g2` inverts the Poissonian-background map
g2_meas = 1 + rho^2 (g2_true - 1) (Brouri et al., Opt. Lett. 25, 1294
(2000)) -- exactly the forward model the package's own `g2_zero(...,
rho)` applies, so the round trip is machine-exact and asserted in the
tests; the correction maps confidence-interval endpoints through the
same affine transform, truncates at the physical floor g2 = 0 without
hiding the untruncated value, and refuses rho outside (0, 1].
`signal_fraction` builds rho from measured signal and background rates.
`deadtime_corrected_rate` inverts the non-paralyzable dead-time
throughput r_meas = r/(1 + r tau_d) -- the exact renewal-theory rate of
the greedy dead-time pass in the Monte-Carlo detector chain, validated
against it statistically -- and refuses measured rates at or beyond the
saturation rate instead of extrapolating. All three are torch-free.

```python
from sparq import background_corrected_g2, signal_fraction

rho = signal_fraction(signal_rate, background_rate)
res = background_corrected_g2(g2_measured, rho, ci=(lo, hi))
print(res["g2_corrected"], res["ci"])
```

## Registering your own platform

The built-in priors (NV, hBN, GaN, SiV) are literature-anchored defaults,
not a limit: `register_platform` adds any emitter with your own
photophysical ranges, after which it works everywhere a platform name is
accepted (site sampling, the dataset generators, the triage environment,
the graph encoder's template).

```python
from sparq import Platform, register_platform, sample_site
register_platform(Platform("MyQD", (0.5, 2.0), (20, 400), (0.0, 0.5),
                           (50, 500), (0.7, 0.99), 0.05, (5, 100), (0.5, 10)))
site = sample_site(rng, platform="MyQD")
```

## What is in the package

```
sparq/
  physics.py            emitter photophysics, platform priors, HBT twin
                        (exact Poisson histogram statistics) and the full
                        Monte-Carlo photon-stream simulator w/ detector
                        impairments
  exact.py              numerically exact master-equation g2(tau)
  pulsed.py             pulsed-excitation twin + comb calibration +
                        conventional peak-area analysis
  analysis.py           g2 analysis of measured data: bootstrap and
                        profile-likelihood uncertainties, exact
                        background and dead-time corrections (torch-free)
  sequential.py         Wald SPRT certifier on exact Poisson likelihoods
                        (torch-free)
  datasets.py           synthetic acquisition generators + loader for the
                        real sps-quality quantum-dot HBT data
  estimators.py         LM-fit baseline, CNN, surrogate-gradient spiking
                        network, physics-in-the-loop training
  twin_torch.py         differentiable twin (adjoint/pathwise gradients
                        through the measurement protocol) + profile
                        Fisher information
  sac_per.py            discrete-action Soft Actor-Critic + prioritized
                        experience replay (sum-tree)
  rl_env.py             closed-loop emitter-triage environment + baselines
  gnn.py                level-structure template graphs + message-passing
                        encoder for cross-platform transfer
```

## Tests

```bash
pip install -e .[test]
pytest tests -q     # a few seconds; ML tests skip when torch is absent
```

The suite pins the physics to exact references: the two-exponential g2 law
against the master-equation eigen-decomposition, the closed-form IRF
convolution against brute-force quadrature, Poisson statistics of the
histogram twin, comb calibration and peak-area recovery, sum-tree replay
proportionality, the exact background/dead-time correction round
trips, and the shape/gradient contracts of the estimators, the
differentiable protocol twin and the triage environment. It runs in CI on
every push and pull request.

## Real data

The experimental quantum-dot HBT measurements used by
`sparq.datasets.load_fisequr` are from the openly licensed
[sps-quality](https://github.com/UTS-CASLab/sps-quality) repository
(Kedziora et al., *Mach. Learn.: Sci. Technol.* **4**, 045042 (2023));
they are not redistributed here.

## Contributing and support

Bug reports, questions and pull requests are welcome through
[GitHub issues](https://github.com/TaN-MM-Org/sparq-triage/issues); see
[CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and the design
rules. Tagged releases are published to PyPI by CI.

## License and citation

Apache-2.0 (see LICENSE). Please cite the associated paper if you use this
code; citation metadata is in [CITATION.cff](CITATION.cff).
