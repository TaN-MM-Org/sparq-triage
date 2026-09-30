"""Inputs refused from 0.11.0 that earlier gave a cryptic error or a
result without an error."""
import numpy as np
import pytest

from sparq import (EmitterSite, HBTConfig, analyze_histogram, analyze_pulsed,
                   expected_histogram, fit_g2_histogram, profile_likelihood_ci)

SITE = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                        rho=0.95, blinking=False), 1)
GRID = HBTConfig(tau_max=90.25, n_bins=361)
COUNTS = np.random.default_rng(0).poisson(
    expected_histogram(SITE, 30.0, GRID)).astype(float)


def test_analyze_histogram_refuses_bad_inputs():
    bad = COUNTS.copy()
    bad[100] = np.nan                      # 0.10.0: TypeError deep inside
    with pytest.raises(ValueError, match="finite"):
        analyze_histogram(GRID.bin_centers, bad, 30.0, n_bootstrap=0)
    d = GRID.bin_centers.copy()
    d[-1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        analyze_histogram(d, COUNTS, 30.0, n_bootstrap=0)
    # 0.10.0: ZeroDivisionError for 0, and g2_0 = 1.0 with ok=False
    # (no error) for a negative time
    for T in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="T_s"):
            analyze_histogram(GRID.bin_centers, COUNTS, T, n_bootstrap=0)


def test_fits_refuse_a_histogram_on_another_grid():
    cfg = HBTConfig()
    # 0.10.0: (1.0, False) from fit_g2_histogram and a broadcasting
    # error from profile_likelihood_ci
    with pytest.raises(ValueError, match="121 bins"):
        fit_g2_histogram(COUNTS, 30.0, 150e3, cfg)
    with pytest.raises(ValueError, match="121 bins"):
        profile_likelihood_ci(COUNTS, 30.0, 150e3, cfg)
    h = expected_histogram(SITE, 30.0, cfg)
    for T, r in ((0.0, 150e3), (30.0, -1.0), (np.nan, 150e3)):
        with pytest.raises(ValueError, match="positive"):
            fit_g2_histogram(h, T, r, cfg)
    neg = h.copy()
    neg[3] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        profile_likelihood_ci(neg, 30.0, 150e3, cfg)


def test_valid_fits_are_unchanged_by_the_checks():
    """The checks pass the histogram on with its data type, so a float32
    histogram (as rebin_real returns) is fitted exactly as before."""
    from sparq.analysis import _check_hist_inputs
    cfg = HBTConfig()
    h = np.random.default_rng(1).poisson(
        expected_histogram(SITE, 30.0, cfg)).astype(np.float32)
    out = _check_hist_inputs(h, 30.0, 150e3, cfg)
    assert out.dtype == np.float32 and np.array_equal(out, h)


def test_analyze_pulsed_refuses_non_finite_counts():
    d = np.arange(484) * 0.25 - 60.375
    c = np.ones(484)
    c[7] = np.nan                           # 0.10.0: g2_0 = nan, no error
    for method in ("fit", "area"):
        with pytest.raises(ValueError, match="finite"):
            analyze_pulsed(d, c, method=method, n_bootstrap=0)
