"""Exact planning over the scatter of real runs (new in 0.10.0): the
certification probability against simulated runs, and the assured
time."""
import numpy as np
import pytest

from sparq import (HBTConfig, bayesian_g2, required_acquisition_time,
                   site_from_numbers)
from sparq.lab import assured_acquisition_time, certification_probability
from sparq.physics import sample_histogram

GOOD = site_from_numbers(tau1_ns=12.0, tau2_ns=200.0, a=0.3,
                         rate_kcps=120.0, rho=0.97)


def _simulated(site, t, n, seed, conf=0.95):
    cfg = HBTConfig()
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(n):
        h = np.asarray(sample_histogram(site, t, cfg, rng), float)
        ok += bayesian_g2(h, cfg).prob_below(0.5) >= conf
    return ok / n


def test_certification_probability_matches_simulated_runs():
    t_typ, _ = required_acquisition_time(GOOD, confidence=0.95)
    for k, t in enumerate((0.5 * t_typ, t_typ, 2.0 * t_typ)):
        p = certification_probability(GOOD, t)
        sim = _simulated(GOOD, t, 2000, 10 + k)
        assert abs(sim - p) < 4 * np.sqrt(p * (1 - p) / 2000) + 1e-3
    # at the typical-data time a sizeable share of runs fall short
    assert certification_probability(GOOD, t_typ) < 0.8


def test_assured_time_reaches_its_assurance():
    t, p, p_min = assured_acquisition_time(GOOD, assurance=0.9)
    assert p >= 0.9 and p_min >= 0.9
    # it stays up after t, on a grid independent of the search's
    for f in np.linspace(1.0, 3.0, 41):
        assert certification_probability(GOOD, f * t) >= 0.9
    # and just before t it was below
    assert certification_probability(GOOD, t / 1.002) < 0.9
    t_typ, _ = required_acquisition_time(GOOD, confidence=0.95)
    assert t > t_typ
    sim = _simulated(GOOD, t, 2000, 99)
    assert sim > 0.9 - 4 * np.sqrt(0.09 / 2000)


def test_assured_time_refusals():
    bad = site_from_numbers(12.0, 200.0, 0.3, 120.0, rho=0.97,
                            n_emitters=4)
    with pytest.raises(ValueError, match="window-averaged g2"):
        assured_acquisition_time(bad)
    dim = site_from_numbers(12.0, 200.0, 0.3, rate_kcps=0.5, rho=0.97)
    with pytest.raises(ValueError, match="t_max_s"):
        assured_acquisition_time(dim, assurance=0.99, t_max_s=2.0)
    with pytest.raises(ValueError):
        certification_probability(GOOD, -1.0)
    with pytest.raises(ValueError):
        assured_acquisition_time(GOOD, assurance=1.0)
