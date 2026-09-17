"""Acquisition-planning anchors: the window-averaged g2 ratio is
exactly independent of acquisition time (both window means scale
linearly with it); the planner's posterior is exactly
`bayesian_g2` on `expected_histogram` (two public code paths); the
required time brackets the confidence on both sides; a site whose
window g2 is not below threshold is refused by that exact argument;
and seeded Monte-Carlo runs at a 4x margin certify > 90% of the
time."""
import numpy as np
import pytest

from sparq import (HBTConfig, bayesian_g2, expected_histogram,
                   expected_posterior, required_acquisition_time,
                   site_from_numbers)
from sparq.physics import sample_histogram

CFG = HBTConfig()
GOOD = site_from_numbers(tau1_ns=12.0, tau2_ns=200.0, a=0.3,
                         rate_kcps=120.0, rho=0.97, n_emitters=1)
BAD = site_from_numbers(tau1_ns=12.0, tau2_ns=200.0, a=0.3,
                        rate_kcps=120.0, rho=0.97, n_emitters=4)


def test_window_ratio_time_independent():
    """g2_window at 1 s equals g2_window at 1000 s exactly: the ratio
    of two quantities both linear in T."""
    a = expected_posterior(GOOD, 1.0)["g2_window"]
    b = expected_posterior(GOOD, 1000.0)["g2_window"]
    assert abs(a - b) < 1e-12 * abs(a)


def test_planner_is_exactly_the_public_posterior():
    """Same average histogram, same posterior: the planner must equal
    `bayesian_g2(expected_histogram(...))` called directly."""
    t = 3.0
    mu = expected_histogram(GOOD, t, CFG)
    post = bayesian_g2(mu, CFG)
    rep = expected_posterior(GOOD, t, CFG)
    assert abs(rep["prob_below"] - post.prob_below(0.5)) < 1e-15
    lo, hi = post.credible_interval(0.95)
    assert abs(rep["interval"][0] - lo) < 1e-12
    assert abs(rep["interval"][1] - hi) < 1e-12


def test_required_time_brackets_confidence():
    t, rep = required_acquisition_time(GOOD, confidence=0.99)
    assert rep["prob_below"] >= 0.99
    shorter = expected_posterior(GOOD, 0.8 * t)["prob_below"]
    assert shorter < 0.99
    # more time can only help at this operating point
    longer = expected_posterior(GOOD, 2.0 * t)["prob_below"]
    assert longer >= rep["prob_below"] - 1e-12


def test_uncertifiable_site_refused_with_exact_reason():
    """Four emitters push the window g2 above 1/2; since the window
    ratio never changes with time, no run length certifies the site --
    refused, naming the number, instead of searching forever."""
    g_w = expected_posterior(BAD, 1.0)["g2_window"]
    assert g_w >= 0.5
    with pytest.raises(ValueError, match="window-averaged g2"):
        required_acquisition_time(BAD)


def test_unreached_confidence_within_t_max_refused():
    dim = site_from_numbers(12.0, 200.0, 0.3, rate_kcps=0.5,
                            rho=0.97, n_emitters=1)
    with pytest.raises(ValueError, match="t_max_s"):
        required_acquisition_time(dim, confidence=0.999, t_max_s=2.0)


def test_monte_carlo_certification_at_margin():
    """At 4x the planned time, seeded Poisson-sampled runs must
    certify (posterior verdict >= 0.95) more than 90% of the time --
    the stated practical margin."""
    t, _ = required_acquisition_time(GOOD, confidence=0.95)
    rng = np.random.default_rng(23)
    ok = 0
    n = 200
    for _ in range(n):
        hist = sample_histogram(GOOD, 4.0 * t, CFG, rng)
        post = bayesian_g2(np.asarray(hist, dtype=float), CFG)
        ok += post.prob_below(0.5) >= 0.95
    assert ok / n > 0.9


def test_site_from_numbers_matches_package_g2():
    """The convenience constructor must reproduce the package's own
    g2(0) bookkeeping."""
    from sparq import g2_zero
    want = g2_zero(12.0, 200.0, 0.3, 1, 0.97)
    assert abs(GOOD.g2_0 - want) < 1e-15
    with pytest.raises(ValueError, match="rho"):
        site_from_numbers(12.0, 200.0, 0.3, 100.0, rho=1.5)
    with pytest.raises(ValueError, match="n_emitters"):
        site_from_numbers(12.0, 200.0, 0.3, 100.0, n_emitters=0)


def test_input_refusals():
    with pytest.raises(ValueError, match="confidence"):
        required_acquisition_time(GOOD, confidence=1.0)
    with pytest.raises(ValueError, match="T_s"):
        expected_posterior(GOOD, -1.0)
