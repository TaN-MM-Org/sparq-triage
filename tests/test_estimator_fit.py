"""The conventional curve-fit baseline on noiseless synthetic histograms."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")  # sparq.estimators imports torch

from sparq.estimators import balanced_accuracy, fit_g2_histogram
from sparq.physics import EmitterSite, HBTConfig, expected_histogram


def _site(n):
    params = dict(platform="NV", tau1=15.0, tau2=250.0, a=0.3,
                  rate_kcps=150.0, rho=0.95, blinking=False,
                  t_on_ms=10.0, t_off_ms=1.0)
    return EmitterSite(params, n)


def test_fit_recovers_g2_zero_on_clean_data():
    cfg = HBTConfig()
    for n in (1, 3):
        site = _site(n)
        hist = expected_histogram(site, 30.0, cfg)
        est, ok = fit_g2_histogram(hist, 30.0,
                                   site.params["rate_kcps"] * 1e3, cfg)
        assert ok
        assert abs(est - site.g2_0) < 0.05
        assert (est < 0.5) == (site.g2_0 < 0.5)


def test_fit_flags_empty_histograms():
    cfg = HBTConfig()
    est, ok = fit_g2_histogram(np.zeros(cfg.n_bins), 1.0, 1e5, cfg)
    assert not ok and est == 1.0


def test_balanced_accuracy():
    y = np.array([0, 0, 1, 1])
    assert balanced_accuracy(y, np.array([0, 0, 1, 1])) == 1.0
    assert balanced_accuracy(y, np.array([1, 1, 0, 0])) == 0.0
    assert balanced_accuracy(y, np.array([0, 1, 1, 0])) == 0.5
