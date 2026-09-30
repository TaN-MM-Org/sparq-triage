"""Afterpulsing in the fast simulator (new in 0.11.0).

References: numerical integrals (scipy.integrate.quad) of the delay
densities for the closed forms, the site's own `site_g2` for the
exponential form, and the photon-by-photon detector chain, an
independent code path, for the whole histogram.
"""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import chi2

from sparq import (DetectorImpairments, EmitterSite, HBTConfig, correlate,
                   expected_histogram, simulate_photon_stream)
from sparq.physics import (_afterpulse_terms, _exp_after_delay, _exp_laplace,
                           _g2_exponentials, deadtime_throughput, site_g2)


def _quad_after(T, a, z):
    f = lambda y: np.exp(-y / a) / a * np.exp(-abs(z + y) / T)  # noqa: E731
    if z >= 0:
        return quad(f, 0, np.inf, limit=200, epsabs=1e-14)[0]
    return (quad(f, 0, -z, limit=200, epsabs=1e-14)[0]
            + quad(f, -z, np.inf, limit=200, epsabs=1e-14)[0])


def _quad_laplace(T, a, x):
    f = lambda v: np.exp(-abs(v) / a) / (2 * a) * np.exp(-abs(x + v) / T)  # noqa: E731,E501
    lo, hi = min(0.0, -x), max(0.0, -x)
    return (quad(f, -np.inf, lo, limit=200, epsabs=1e-14)[0]
            + quad(f, lo, hi, limit=200, epsabs=1e-14)[0]
            + quad(f, hi, np.inf, limit=200, epsabs=1e-14)[0])


def test_closed_forms_match_quadrature():
    # includes T = a (a removable singularity), T just off a, T << a
    for T, a in ((8.0, 80.0), (80.0, 80.0), (80.0, 80.0 * (1 + 1e-9)),
                 (0.05, 80.0), (300.0, 20.0)):
        for z in (-200.0, -60.0, -5.0, -0.1, 0.0, 3.0, 50.0):
            assert abs(_exp_after_delay(T, a, z) - _quad_after(T, a, z)) \
                < 1e-12
            assert abs(_exp_laplace(T, a, z) - _quad_laplace(T, a, z)) \
                < 1e-12


def test_exponential_form_of_the_site_g2():
    for blinking in (False, True):
        site = EmitterSite(dict(tau1=8.0, tau2=100.0, a=1.5, rate_kcps=600.0,
                                rho=0.9, blinking=blinking, t_on_ms=3e-4,
                                t_off_ms=1.5e-4), 2)
        x = np.linspace(-3000.0, 3000.0, 2001)
        g = 1 + sum(c * np.exp(-np.abs(x) / T)
                    for c, T in _g2_exponentials(site))
        assert np.max(np.abs(g - site_g2(site, x, 0.0))) < 1e-12


def test_left_out_jitter_is_small_in_the_afterpulse_terms():
    """[G*f] from the jitter-free closed form against a numerical
    integral of the jitter-blurred g2 (0.35 ns per detector)."""
    site = EmitterSite(dict(tau1=8.0, tau2=100.0, a=1.5, rate_kcps=600.0,
                            rho=0.9, blinking=False), 1)
    tau = np.array([-60.0, -46.0, -45.0, -44.0, 0.0, 30.0])
    plus, _, _ = _afterpulse_terms(site, tau, 45.0, 80.0)
    for t, v in zip(tau, plus):
        g = lambda y: np.exp(-y / 80) / 80 * site_g2(  # noqa: E731
            site, np.array([t + 45.0 + y]), 0.35)[0]
        z = -(t + 45.0)
        q = quad(g, 0, np.inf, limit=500, epsabs=1e-13)[0] if z <= 0 else \
            quad(g, 0, z, limit=500, epsabs=1e-13)[0] \
            + quad(g, z, np.inf, limit=500, epsabs=1e-13)[0]
        assert abs(q - v) < 5e-4


def test_afterpulsing_matches_the_photon_simulator():
    """Strong afterpulsing (p = 0.2, 80 ns) with 45 ns dead time: the
    photon-by-photon histogram passes a chi-square test against the fast
    simulator with afterpulsing, and fails it without, even with the
    flat level scaled to the measured rates."""
    cfg = HBTConfig()
    site = EmitterSite(dict(tau1=8.0, tau2=100.0, a=1.5, rate_kcps=600.0,
                            rho=0.9, blinking=False), 1)
    T, p = 6.0, 0.2
    imp = DetectorImpairments(dead_time_ns=45.0, afterpulse_p=p,
                              afterpulse_tau_ns=80.0, sigma_irf_ns=0.35)
    t_a, t_b = simulate_photon_stream(site, T, np.random.default_rng(5),
                                      imp=imp)
    want = deadtime_throughput(site, 300e3, 45.0) * (1 + p)
    for t in (t_a, t_b):
        assert abs(len(t) / T - want) < 5 * np.sqrt(want / T)
    h = correlate(t_a, t_b, cfg)

    def p_value(mu):
        return float(chi2.sf(np.sum((h - mu) ** 2 / mu), h.size))
    with_ap = expected_histogram(site, T, cfg, dead_time_ns=45.0,
                                 afterpulse_p=p, afterpulse_tau_ns=80.0)
    without = expected_histogram(site, T, cfg, dead_time_ns=45.0)
    assert p_value(with_ap) > 1e-3
    assert p_value(without) < 1e-6
    assert p_value(without * (1 + p) ** 2) < 1e-6
    # far from zero delay the level is the measured rates' (1 + p)^2
    far = expected_histogram(site, T, HBTConfig(tau_max=5000.5, n_bins=10001),
                             dead_time_ns=45.0, afterpulse_p=p)[0]
    flat = (want / (1 + p)) ** 2 * 1e-9 * T
    assert abs(far / flat - (1 + p) ** 2) < 1e-6


def test_afterpulsing_defaults_and_refusals():
    cfg = HBTConfig()
    site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                            rho=0.95, blinking=False), 1)
    assert np.array_equal(expected_histogram(site, 3.0, cfg),
                          expected_histogram(site, 3.0, cfg, afterpulse_p=0.0))
    for bad in (-0.1, 1.5, np.nan):
        with pytest.raises(ValueError):
            expected_histogram(site, 3.0, cfg, afterpulse_p=bad)
    with pytest.raises(ValueError):
        expected_histogram(site, 3.0, cfg, afterpulse_p=0.1,
                           afterpulse_tau_ns=0.0)
