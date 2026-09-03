"""General-purpose HBT analysis and user-registered platforms, tested
against the twin's ground truth (no torch required)."""
import numpy as np
import pytest

from sparq.analysis import analyze_histogram, analyze_pulsed
from sparq.physics import (
    PLATFORMS,
    EmitterSite,
    HBTConfig,
    Platform,
    expected_histogram,
    register_platform,
    sample_site,
)
from sparq.pulsed import expected_hist_pulsed


def _site(n, rate=150.0):
    params = dict(platform="NV", tau1=15.0, tau2=250.0, a=0.3,
                  rate_kcps=rate, rho=0.95, blinking=False,
                  t_on_ms=10.0, t_off_ms=1.0)
    return EmitterSite(params, n)


def _cw_data(site, T_s, offset_ns, rng):
    """Poisson HBT data on a wide, offset delay grid (as a correlator gives)."""
    cfg = HBTConfig(tau_max=90.0, n_bins=360, sigma_irf=0.35)
    mu = expected_histogram(site, T_s, cfg)
    return cfg.bin_centers + offset_ns, rng.poisson(mu).astype(float)


def test_cw_analysis_recovers_truth_with_covering_ci():
    rng = np.random.default_rng(0)
    site = _site(1)
    delay, counts = _cw_data(site, T_s=30.0, offset_ns=7.0, rng=rng)
    res = analyze_histogram(delay, counts, T_s=30.0, n_bootstrap=100, seed=1)
    assert res["ok"]
    assert abs(res["center"] - 7.0) < 2.0             # dip located
    assert abs(res["g2_0"] - site.g2_0) < 0.1
    assert res["g2_0_low"] <= res["g2_0_high"]
    assert res["single_emitter"] and res["single_emitter_confident"]


def test_cw_analysis_flags_a_pair():
    rng = np.random.default_rng(2)
    site = _site(2)                                    # g2(0) = 0.45...
    assert 0.5 < site.g2_0 or site.g2_0 < 0.5          # just evaluate it
    delay, counts = _cw_data(site, T_s=60.0, offset_ns=-3.0, rng=rng)
    res = analyze_histogram(delay, counts, T_s=60.0, n_bootstrap=100, seed=3)
    assert res["ok"]
    assert abs(res["g2_0"] - site.g2_0) < 0.12
    assert res["single_emitter"] == (res["g2_0"] < 0.5)


def test_cw_analysis_input_validation():
    with pytest.raises(ValueError):
        analyze_histogram([0.0, 1.0], [1.0], T_s=1.0)
    with pytest.raises(ValueError):
        analyze_histogram([0.0, 1.0], [1.0, -2.0], T_s=1.0)


def test_pulsed_analysis_recovers_truth():
    rng = np.random.default_rng(4)
    cfg = HBTConfig(tau_max=60.5, n_bins=484, sigma_irf=0.35)
    true_g2 = 0.12
    mu = expected_hist_pulsed(cfg, 20.0, 2e5, true_g2, 1.0, a=0.0,
                              center_off=1.7)
    counts = rng.poisson(mu).astype(float)
    res = analyze_pulsed(cfg.bin_centers, counts, n_bootstrap=100, seed=5)
    assert abs(res["g2_0"] - true_g2) < 0.05
    assert res["g2_0_low"] <= res["g2_0"] <= res["g2_0_high"] or \
        res["g2_0_high"] - res["g2_0_low"] < 0.1
    assert res["single_emitter"]


def test_register_platform_roundtrip():
    p = Platform("TESTQD", (0.5, 2.0), (20, 400), (0.0, 0.5),
                 (50, 500), (0.7, 0.99), 0.05, (5, 100), (0.5, 10))
    try:
        register_platform(p)
        assert "TESTQD" in PLATFORMS
        rng = np.random.default_rng(6)
        site = sample_site(rng, platform="TESTQD")
        assert 0.5 <= site.params["tau1"] <= 2.0
        assert 0.7 <= site.params["rho"] <= 0.99
        # duplicate refused, overwrite allowed
        with pytest.raises(ValueError):
            register_platform(p)
        register_platform(p, overwrite=True)
    finally:
        PLATFORMS.pop("TESTQD", None)


def test_register_platform_validation():
    bad_rng = Platform("BAD", (2.0, 0.5), (20, 400), (0.0, 0.5),
                       (50, 500), (0.7, 0.99), 0.05, (5, 100), (0.5, 10))
    with pytest.raises(ValueError):
        register_platform(bad_rng)
    bad_rho = Platform("BAD2", (0.5, 2.0), (20, 400), (0.0, 0.5),
                       (50, 500), (0.7, 1.5), 0.05, (5, 100), (0.5, 10))
    with pytest.raises(ValueError):
        register_platform(bad_rho)
    with pytest.raises(ValueError):
        register_platform("not a platform")
    assert "BAD" not in PLATFORMS and "BAD2" not in PLATFORMS
