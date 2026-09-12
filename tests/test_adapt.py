"""v0.6 adaptability anchors: a user-registered platform runs through
the whole pipeline; the measured-histogram CSV contract round-trips
exactly and refuses malformed files; the lifetime windows are
configurable and validated, and an emitter far outside the NV-scale
defaults is recovered once its window is supplied (and demonstrably
NOT recovered under the default window); the IRF-aware fit targets
the IRF-free g2(0); and the dataset loader refuses a missing path
instead of assuming this machine's layout."""
import numpy as np
import pytest

import sparq
from sparq import (EmitterSite, HBTConfig, Platform, expected_histogram,
                   fit_g2_histogram, load_hbt_csv, profile_likelihood_ci,
                   register_platform, sample_site, save_hbt_csv)
from sparq.datasets import load_fisequr, robust_flat_rate


def test_user_registered_platform_runs_the_pipeline():
    plat = Platform("MyDot", (0.5, 2.0), (20.0, 120.0), (0.05, 0.5),
                    (100, 600), (0.9, 0.995), 0.0, (5, 200), (0.5, 40))
    register_platform(plat, overwrite=True)
    rng = np.random.default_rng(0)
    site = sample_site(rng, platform="MyDot", n_probs=(1.0, 0.0, 0.0, 0.0))
    assert 0.5 <= site.params["tau1"] <= 2.0
    cfg = HBTConfig(tau_max=30.25, n_bins=121, sigma_irf=0.1)
    h = rng.poisson(expected_histogram(site, 30.0, cfg)).astype(float)
    assert h.sum() > 0                      # the twin accepts the platform


def test_hbt_csv_round_trip_and_refusals(tmp_path):
    d = np.linspace(-60.5, 60.5, 122)[:-1] + 0.5
    c = np.random.default_rng(1).poisson(200.0, d.size).astype(float)
    path = tmp_path / "hbt.csv"
    save_hbt_csv(path, d, c)
    d2, c2 = load_hbt_csv(path)
    assert np.array_equal(d, d2) and np.array_equal(c, c2)   # exact
    bad = tmp_path / "bad.csv"
    bad.write_text("time,counts\n0.0,1.0\n")
    with pytest.raises(ValueError):
        load_hbt_csv(bad)                   # wrong header
    bad.write_text("delay_ns,counts\n0.0,1.0,2.0\n")
    with pytest.raises(ValueError):
        load_hbt_csv(bad)                   # wrong field count
    bad.write_text("delay_ns,counts\n0.0,1.0\n1.0,-2.0\n2.0,1\n3.0,1\n4.0,1\n")
    with pytest.raises(ValueError):
        load_hbt_csv(bad)                   # negative counts
    bad.write_text("delay_ns,counts\n0.0,1.0\n0.0,2.0\n2.0,1\n3.0,1\n4.0,1\n")
    with pytest.raises(ValueError):
        load_hbt_csv(bad)                   # non-increasing delay


def test_lifetime_windows_validated_and_needed():
    cfg = HBTConfig()
    h = np.full(cfg.n_bins, 100.0)
    for bad in ((0.0, 1.0), (2.0, 1.0), (1.0,), "x"):
        with pytest.raises(ValueError):
            fit_g2_histogram(h, 1.0, 1e5, cfg, t1_bounds=bad)
        with pytest.raises(ValueError):
            profile_likelihood_ci(h * 3, 1.0, 1e5, cfg, t2_bounds=bad)


def test_fast_emitter_recovered_only_with_its_window():
    """tau1 = 0.15 ns, tau2 = 5 ns: far outside the NV-scale default
    window on both axes.  Under the default window the fit rails and
    misses badly; with the emitter's own window it recovers the
    IRF-free g2(0)."""
    cfg = HBTConfig(tau_max=10.0125, n_bins=801, sigma_irf=0.005)
    site = EmitterSite(dict(tau1=0.15, tau2=5.0, a=0.4, rho=0.97,
                            rate_kcps=400.0, blinking=False), 1)
    rng = np.random.default_rng(0)
    T_s = 300.0
    h = rng.poisson(expected_histogram(site, T_s, cfg)).astype(float)
    r_hat = robust_flat_rate(h, cfg, T_s)
    g_def, _ = fit_g2_histogram(h, T_s, r_hat, cfg)
    g_win, _ = fit_g2_histogram(h, T_s, r_hat, cfg,
                                t1_bounds=(0.02, 2.0),
                                t2_bounds=(0.5, 50.0))
    assert abs(g_win - site.g2_0) < 0.05    # recovered
    assert abs(g_def - site.g2_0) > 0.2     # default window fails, visibly
    prof = profile_likelihood_ci(h, T_s, r_hat, cfg,
                                 t1_bounds=(0.02, 2.0),
                                 t2_bounds=(0.5, 50.0))
    assert prof["lo"] <= site.g2_0 + 0.05 and prof["g2_hat"] < 0.2


def test_irf_aware_fit_targets_irf_free_g2():
    """A slow IRF (0.35 ns) softens the measured dip well above the
    IRF-free g2(0); the IRF-aware model recovers the IRF-free value,
    which the raw center-bin ratio cannot."""
    cfg = HBTConfig()                        # sigma_irf = 0.35
    site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rho=0.95,
                            rate_kcps=150.0, blinking=False), 1)
    measured0 = float(sparq.g2_measured(np.array([0.0]), 15.0, 250.0,
                                        0.3, 1, 0.95, 0.35)[0])
    assert measured0 > site.g2_0 + 0.02      # the IRF really matters here
    rng = np.random.default_rng(3)
    h = rng.poisson(expected_histogram(site, 60.0, cfg)).astype(float)
    r = profile_likelihood_ci(h, 60.0, 150e3, cfg, c0_prior=(1.0, 0.01))
    assert r["lo"] <= site.g2_0 <= r["hi"]


def test_fisequr_loader_refuses_missing_path():
    with pytest.raises(ValueError):
        load_fisequr("/nonexistent/anywhere")
    with pytest.raises(TypeError):
        load_fisequr()                       # no default path anymore
