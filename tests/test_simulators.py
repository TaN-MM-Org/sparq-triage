"""The two simulators against each other (new in 0.10.0).

The photon-by-photon simulator draws every emission of a three-level
emitter, gates it with an on/off blinking process, adds background and
runs each detector's dead time; the fast simulator writes the mean
histogram down in closed form. They are independent code paths, so a
Poisson chi-square test between them checks both. Each check is also
run against the 0.9.1 behaviour to show that the test can tell the
difference.
"""
import numpy as np
import pytest
from scipy.stats import chi2

import sparq.physics as ph
from sparq import (DetectorImpairments, EmitterSite, HBTConfig, correlate,
                   g2_measured, simulate_photon_stream)
from sparq.exact import (effective_params, g2_exact, rates_for_params,
                         rates_from_site)
from sparq.physics import (deadtime_throughput, expected_histogram,
                           mean_detected_rate_cps, site_g2)


def _bin_average(cfg, f, n_sub=64):
    """Mean of f over each bin (a histogram counts the whole bin)."""
    sub = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    t = cfg.bin_centers[:, None] + sub[None, :] * cfg.bin_width
    return f(t).mean(axis=1)


def _chi2_p(h, mu):
    x = float(np.sum((h - mu) ** 2 / mu))
    return float(chi2.sf(x, h.size))


def test_rates_for_params_round_trip():
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(200):
        t1 = float(np.exp(rng.uniform(np.log(0.3), np.log(30.0))))
        t2 = t1 * float(np.exp(rng.uniform(np.log(3.0), np.log(300.0))))
        a = float(rng.uniform(0.0, 2.0))
        try:
            k = rates_for_params(t1, t2, a)
        except ValueError:
            continue                        # checked separately below
        assert min(k) >= 0.0
        e1, e2, ea = effective_params(*k) if a > 0 else (t1, t2, 0.0)
        assert abs(e1 - t1) < 1e-9 * t1
        assert abs(ea - a) < 1e-9 * max(a, 1.0)
        if a > 0:
            assert abs(e2 - t2) < 1e-9 * t2
        tau = np.linspace(0.0, 5 * t2, 300)
        want = 1.0 - (1.0 + a) * np.exp(-tau / t1) + a * np.exp(-tau / t2)
        assert np.abs(g2_exact(tau, *k) - want).max() < 1e-9
        checked += 1
    assert checked >= 190                   # 199 of the 200 draws
    # the approximate 0.9.1 mapping does not reproduce its inputs
    e1, e2, ea = effective_params(*rates_from_site(8.0, 100.0, 1.5))
    assert abs(e2 - 100.0) > 50.0


def test_rates_for_params_refusals():
    with pytest.raises(ValueError, match="tau2"):
        rates_for_params(10.0, 5.0, 0.3)
    with pytest.raises(ValueError):
        rates_for_params(10.0, 50.0, -0.1)
    with pytest.raises(ValueError, match="pump_fraction"):
        rates_for_params(1.5, 20.0, 3.0, pump_fraction=0.1)   # needs 0.194


def test_photon_simulator_has_the_sites_g2():
    """Chi-square of the simulated all-pairs histogram against the site's
    own g2 (bin-averaged, no jitter), normalized with the simulated
    accidentals. The 0.9.1 rate mapping fails the same test."""
    cfg = HBTConfig(sigma_irf=0.0)
    site = EmitterSite(dict(tau1=8.0, tau2=100.0, a=1.5, rate_kcps=600.0,
                            rho=0.9, blinking=False), 1)
    T = 4.0
    t_a, t_b = simulate_photon_stream(site, T, np.random.default_rng(1))
    h = correlate(t_a, t_b, cfg)
    acc = len(t_a) * len(t_b) * cfg.bin_width * 1e-9 / T
    mu = acc * _bin_average(cfg, lambda t: site_g2(site, t))
    assert _chi2_p(h, mu) > 1e-3
    # the detected rate is the site's rate
    r = (len(t_a) + len(t_b)) / T
    assert abs(r - 600e3) < 5 * np.sqrt(600e3 / T)
    # the same test with the 0.9.1 mapping has a vanishing p-value
    old = ph._site_rates
    try:
        ph._site_rates = rates_from_site
        t_a, t_b = simulate_photon_stream(site, T, np.random.default_rng(1))
    finally:
        ph._site_rates = old
    h = correlate(t_a, t_b, cfg)
    acc = len(t_a) * len(t_b) * cfg.bin_width * 1e-9 / T
    mu = acc * _bin_average(cfg, lambda t: site_g2(site, t))
    assert _chi2_p(h, mu) < 1e-6


def test_fast_simulator_blinking_matches_photon_simulator():
    """Microsecond blinking, so the gate's own bunching is visible within
    the window. The exact gate model passes; the 0.9.1 flat-level model,
    which filled the dip, fails."""
    cfg = HBTConfig(sigma_irf=0.0)
    p = dict(tau1=8.0, tau2=100.0, a=1.5, rate_kcps=600.0, rho=0.9,
             blinking=True, t_on_ms=3e-4, t_off_ms=1.5e-4)
    site = EmitterSite(p, 1)
    T = 8.0
    t_a, t_b = simulate_photon_stream(site, T, np.random.default_rng(3))
    r = (len(t_a) + len(t_b)) / T
    assert abs(r - mean_detected_rate_cps(site)) < 5 * np.sqrt(r / T)
    h = correlate(t_a, t_b, cfg)
    acc = len(t_a) * len(t_b) * cfg.bin_width * 1e-9 / T
    mu = acc * _bin_average(cfg, lambda t: site_g2(site, t))
    assert _chi2_p(h, mu) > 1e-3
    # blinking cannot fill the dip: at zero delay the same value as
    # without blinking at the time-averaged signal share
    d = p["t_on_ms"] / (p["t_on_ms"] + p["t_off_ms"])
    rho_b = p["rho"] * d / (p["rho"] * d + 1 - p["rho"])
    assert abs(site_g2(site, 0.0) - (1 - rho_b ** 2)) < 1e-12
    # the 0.9.1 model
    duty = d

    def old(t):
        return (1.0 + (g2_measured(t, 8.0, 100.0, 1.5, 1, p["rho"], 0.0)
                       - 1.0) + (1.0 / duty - 1.0) * p["rho"] ** 2)
    assert _chi2_p(h, acc * _bin_average(cfg, old)) < 1e-6
    # the fast simulator's histogram is this g2, bin-averaged, on the
    # flat level
    e = expected_histogram(site, T, cfg)
    flat = (0.5 * mean_detected_rate_cps(site)) ** 2 * 1e-9 * T
    assert np.allclose(e / flat, _bin_average(cfg, lambda t: site_g2(
        site, t), n_sub=4000), rtol=1e-7, atol=0)


def test_fast_simulator_dead_time_matches_photon_simulator():
    cfg = HBTConfig()
    site = EmitterSite(dict(tau1=8.0, tau2=100.0, a=1.5, rate_kcps=600.0,
                            rho=0.9, blinking=False), 1)
    T = 6.0
    imp = DetectorImpairments(dead_time_ns=45.0, afterpulse_p=0.0,
                              sigma_irf_ns=0.35)
    t_a, t_b = simulate_photon_stream(site, T, np.random.default_rng(5),
                                      imp=imp)
    want = deadtime_throughput(site, 300e3, 45.0)
    for t in (t_a, t_b):
        assert abs(len(t) / T - want) < 5 * np.sqrt(want / T)
    # the Poisson-light formula r/(1 + r tau_d) is off for this bunched
    # light by more than the noise
    poisson = 300e3 / (1 + 300e3 * 45e-9)
    assert abs((len(t_a) + len(t_b)) / (2 * T) - poisson) \
        > 5 * np.sqrt(poisson / (2 * T))
    h = correlate(t_a, t_b, cfg)
    assert _chi2_p(h, expected_histogram(site, T, cfg,
                                         dead_time_ns=45.0)) > 1e-3
    assert _chi2_p(h, expected_histogram(site, T, cfg)) < 1e-6
    # for Poisson light the throughput is the exact renewal formula
    flat = EmitterSite(dict(tau1=8.0, tau2=100.0, a=0.0, rate_kcps=600.0,
                            rho=1e-9, blinking=False), 1)
    assert abs(deadtime_throughput(flat, 3e5, 45.0) - poisson) < 1e-6 * poisson
    with pytest.raises(ValueError):
        expected_histogram(site, T, cfg, dead_time_ns=-1.0)


def test_non_blinking_fast_simulator_unchanged():
    """For a non-blinking site without dead time, with bin averaging off,
    the fast simulator is the 0.9.1 formula, bit for bit."""
    cfg = HBTConfig(bin_average=False)
    site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                            rho=0.95, blinking=False), 2)
    p = site.params
    r = 0.5 * p["rate_kcps"] * 1e3 * 1.0
    g2 = g2_measured(cfg.bin_centers, p["tau1"], p["tau2"], p["a"], 2,
                     p["rho"], cfg.sigma_irf)
    old = r * r * (cfg.bin_width * 1e-9) * 3.0 * g2
    assert np.array_equal(expected_histogram(site, 3.0, cfg), old)
