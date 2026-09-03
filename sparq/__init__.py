"""SPARQ: spiking physics-in-the-loop autonomous reinforcement triage of
quantum emitters.

Core, dependency-light API (NumPy/SciPy only) re-exported here:

* analytic HBT correlation functions and the exact-statistics histogram
  twin (:mod:`sparq.physics`),
* the numerically exact three-level master-equation reference
  (:mod:`sparq.exact`),
* the pulsed-excitation twin and conventional peak-area analysis
  (:mod:`sparq.pulsed`).

The machine-learning components require PyTorch (install the ``ml``
extra: ``pip install sparq-triage[ml]``) and are imported explicitly:

* estimators (curve fit, CNN, spiking network): :mod:`sparq.estimators`
* differentiable protocol twin: :mod:`sparq.twin_torch`
* soft actor-critic with prioritized replay: :mod:`sparq.sac_per`
* closed-loop triage environment: :mod:`sparq.rl_env`
* level-structure graph encoder: :mod:`sparq.gnn`
* synthetic batch generators and the real-data loader: :mod:`sparq.datasets`
  (NumPy only, but its batches feed the torch estimators)
"""

from .physics import (
    DetectorImpairments,
    EmitterSite,
    HBTConfig,
    PLATFORMS,
    Platform,
    correlate,
    expected_histogram,
    g2_measured,
    g2_three_level,
    g2_zero,
    sample_event_stream,
    sample_histogram,
    sample_site,
    simulate_photon_stream,
)
from .exact import effective_params, g2_exact, liouvillian, rates_from_site, steady_state
from .pulsed import calibrate_comb, expected_hist_pulsed, g2_peak_area, peak_shape

__version__ = "0.1.0"

__all__ = [
    "DetectorImpairments", "EmitterSite", "HBTConfig", "PLATFORMS",
    "Platform", "correlate", "expected_histogram", "g2_measured",
    "g2_three_level", "g2_zero", "sample_event_stream", "sample_histogram",
    "sample_site", "simulate_photon_stream",
    "effective_params", "g2_exact", "liouvillian", "rates_from_site",
    "steady_state",
    "calibrate_comb", "expected_hist_pulsed", "g2_peak_area", "peak_shape",
    "__version__",
]
