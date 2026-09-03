"""Tests of the analytic correlation functions and the histogram twin."""
import numpy as np

from sparq.physics import (
    EmitterSite,
    HBTConfig,
    _exp_conv_gauss,
    expected_histogram,
    g2_measured,
    g2_three_level,
    g2_zero,
    sample_event_stream,
    sample_histogram,
)


def _site(tau1=10.0, tau2=200.0, a=0.0, rate=100.0, rho=1.0, n=1):
    params = dict(platform="NV", tau1=tau1, tau2=tau2, a=a, rate_kcps=rate,
                  rho=rho, blinking=False, t_on_ms=10.0, t_off_ms=1.0)
    return EmitterSite(params, n)


def test_ideal_g2_vanishes_at_zero_delay():
    for a in (0.0, 0.3, 1.5):
        assert abs(g2_three_level(0.0, 12.0, 300.0, a)) < 1e-12


def test_ideal_g2_approaches_one_at_large_delay():
    assert abs(g2_three_level(1e6, 12.0, 300.0, 0.7) - 1.0) < 1e-12


def test_g2_zero_matches_measured_limit():
    """g2_zero must equal g2_measured at tau = 0 without IRF smearing."""
    for n, rho in [(1, 1.0), (2, 0.9), (4, 0.7)]:
        got = g2_measured(0.0, 12.0, 300.0, 0.4, n_emitters=n, rho=rho,
                          sigma_irf=0.0)
        # at tau = 0 the a-terms cancel: dip = (1+a) - a = 1
        assert np.isclose(got, g2_zero(12.0, 300.0, 0.4, n, rho))


def test_exp_conv_gauss_matches_numerical_convolution():
    """The closed-form Gaussian convolution against brute-force quadrature."""
    T, s = 12.0, 1.2
    t = np.linspace(-80, 80, 4001)
    dt = t[1] - t[0]
    gauss = np.exp(-t ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi))
    num = np.convolve(np.exp(-np.abs(t) / T), gauss, mode="same") * dt
    tau = np.linspace(-40, 40, 161)
    ana = _exp_conv_gauss(tau, T, s)
    assert np.abs(ana - np.interp(tau, t, num)).max() < 1e-4


def test_exp_conv_gauss_zero_width_is_plain_exponential():
    tau = np.linspace(-30, 30, 101)
    assert np.allclose(_exp_conv_gauss(tau, 8.0, 0.0),
                       np.exp(-np.abs(tau) / 8.0))


def test_expected_histogram_flat_level_and_antibunching_dip():
    """Window edges sit at the accidental-coincidence flat level; the
    center bin is suppressed for a pure single emitter."""
    site = _site()
    cfg = HBTConfig()
    T_s = 5.0
    mu = expected_histogram(site, T_s, cfg)
    flat = (0.5 * site.params["rate_kcps"] * 1e3) ** 2 \
        * (cfg.bin_width * 1e-9) * T_s
    assert abs(mu[0] / flat - 1.0) < 0.02          # tau = -60 ns >> tau1
    assert mu[cfg.n_bins // 2] / flat < 0.1        # IRF-limited dip


def test_sample_histogram_is_poisson_with_the_expected_mean():
    site = _site(rate=200.0)
    cfg = HBTConfig()
    rng = np.random.default_rng(7)
    mu = expected_histogram(site, 2.0, cfg)
    draws = np.stack([sample_histogram(site, 2.0, cfg, rng)
                      for _ in range(400)])
    z = (draws.mean(0) - mu) / np.sqrt(mu / 400)
    assert np.abs(z).max() < 5.0                    # 5-sigma per bin
    # Poisson: variance equals the mean
    ratio = draws.var(0) / mu
    assert abs(np.median(ratio) - 1.0) < 0.15


def test_event_stream_slices_sum_to_the_histogram_mean():
    site = _site(rate=200.0)
    cfg = HBTConfig()
    rng = np.random.default_rng(11)
    mu = expected_histogram(site, 2.0, cfg)
    s = np.stack([sample_event_stream(site, 2.0, cfg, rng, 32).sum(0)
                  for _ in range(400)])
    z = (s.mean(0) - mu) / np.sqrt(mu / 400)
    assert np.abs(z).max() < 5.0


def test_two_emitters_halve_the_dip():
    one, two = _site(n=1), _site(n=2)
    assert one.g2_0 == 0.0
    assert np.isclose(two.g2_0, 0.5)
    assert one.is_good and not two.is_good
