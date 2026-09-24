"""SPARQ: spiking physics-in-the-loop autonomous reinforcement triage of
quantum emitters.

Core, dependency-light API (NumPy/SciPy only) re-exported here:

* analytic HBT correlation functions and the exact-statistics histogram
  twin (:mod:`sparq.physics`),
* the numerically exact three-level master-equation reference
  (:mod:`sparq.exact`),
* the pulsed-excitation twin and conventional peak-area analysis
  (:mod:`sparq.pulsed`),
* measured-data analysis, exact posteriors, planning, sequential tests
  and heralded sources (:mod:`sparq.analysis`, :mod:`sparq.bayes`,
  :mod:`sparq.lab`, :mod:`sparq.sequential`, :mod:`sparq.heralded`),
* more dip models and goodness of fit (:mod:`sparq.models`), the
  model-based posterior of g2(0) (:mod:`sparq.posterior`) and PicoQuant
  PTU files (:mod:`sparq.ptu`), new in 0.10.0.

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
    register_platform,
    sample_site,
    simulate_photon_stream,
    mean_detected_rate_cps,
    site_g2,
    deadtime_throughput,
)
from .analysis import (analyze_histogram, analyze_pulsed,
                       background_corrected_g2, deadtime_corrected_rate,
                       fit_g2_histogram, profile_likelihood_ci,
                       signal_fraction)
from .sequential import SPRTCertifier
from .datasets import (load_hbt_csv, locate_dip, rebin_real, robust_flat_rate,
                       save_hbt_csv)
from .bayes import G2Posterior, bayesian_g2
from .exact import (effective_params, g2_exact, liouvillian, rates_for_params,
                    rates_from_site, steady_state)
from .pulsed import calibrate_comb, expected_hist_pulsed, g2_peak_area, peak_shape
from .heralded import (car_for_purity, heralded_from_car, heralded_g2_limit,
                       heralded_source)
from .lab import (assured_acquisition_time, certification_probability,
                  expected_posterior, required_acquisition_time,
                  site_from_numbers)
from .models import (exp_conv_gauss_complex, fit_model, g2_model,
                     goodness_of_fit)
from .posterior import ModelPosterior, bayesian_g2_model
from .ptu import load_ptu_timetags, read_ptu, save_ptu_t2
from .sequential import WindowSPRT
from .timetags import (correlate_start_stop, load_timetags_csv,
                       normalize_g2, save_timetags_csv)

__version__ = "0.10.0"

__all__ = [
    "DetectorImpairments", "EmitterSite", "HBTConfig", "PLATFORMS",
    "Platform", "correlate", "expected_histogram", "g2_measured",
    "g2_three_level", "g2_zero", "sample_event_stream", "sample_histogram",
    "sample_site", "simulate_photon_stream",
    "register_platform",
    "analyze_histogram", "analyze_pulsed", "fit_g2_histogram",
    "profile_likelihood_ci", "SPRTCertifier",
    "G2Posterior", "bayesian_g2",
    "load_hbt_csv", "save_hbt_csv", "rebin_real", "robust_flat_rate",
    "locate_dip",
    "background_corrected_g2", "deadtime_corrected_rate", "signal_fraction",
    "effective_params", "g2_exact", "liouvillian", "rates_from_site",
    "steady_state",
    "calibrate_comb", "expected_hist_pulsed", "g2_peak_area", "peak_shape",
    "correlate_start_stop", "load_timetags_csv", "normalize_g2",
    "save_timetags_csv",
    "site_from_numbers", "expected_posterior",
    "required_acquisition_time",
    "heralded_g2_limit", "car_for_purity",
    "heralded_source", "heralded_from_car",
    "mean_detected_rate_cps", "site_g2", "deadtime_throughput",
    "rates_for_params",
    "fit_model", "g2_model", "goodness_of_fit", "exp_conv_gauss_complex",
    "bayesian_g2_model", "ModelPosterior",
    "certification_probability", "assured_acquisition_time",
    "WindowSPRT",
    "read_ptu", "load_ptu_timetags", "save_ptu_t2",
    "__version__",
]
