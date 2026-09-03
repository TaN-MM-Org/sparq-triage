# SPARQ

[![Tests](https://github.com/TaN-MM-Org/sparq-triage/actions/workflows/tests.yml/badge.svg)](https://github.com/TaN-MM-Org/sparq-triage/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/sparq-triage)](https://pypi.org/project/sparq-triage/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

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
proportionality, and the shape/gradient contracts of the estimators, the
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
