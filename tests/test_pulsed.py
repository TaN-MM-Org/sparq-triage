"""Pulsed-excitation twin, comb calibration, and peak-area analysis."""
import numpy as np

from sparq.physics import HBTConfig
from sparq.pulsed import (
    T_REP_NS,
    calibrate_comb,
    expected_hist_pulsed,
    g2_peak_area,
    peak_shape,
)


def test_peak_shape_has_unit_area():
    t = np.linspace(-60, 60, 12001)
    area = np.trapezoid(peak_shape(t, 1.2, 0.5), t)
    assert abs(area - 1.0) < 1e-3


def test_comb_calibration_recovers_the_center():
    cfg = HBTConfig(tau_max=60.5, n_bins=484, sigma_irf=0.35)
    true_center = 1.7
    mu = expected_hist_pulsed(cfg, 10.0, 2e5, 0.12, 1.0, a=0.0,
                              center_off=true_center)
    center, phase, period = calibrate_comb(cfg.bin_centers, mu)
    assert period == T_REP_NS
    assert abs(center - true_center) <= 2 * cfg.bin_width


def test_peak_area_recovers_g2_zero():
    cfg = HBTConfig(tau_max=60.5, n_bins=484, sigma_irf=0.35)
    for g2_0 in (0.05, 0.12, 0.45):
        mu = expected_hist_pulsed(cfg, 10.0, 2e5, g2_0, 1.0, a=0.0,
                                  center_off=1.7)
        center, _, _ = calibrate_comb(cfg.bin_centers, mu)
        got = g2_peak_area(cfg.bin_centers, mu, center)
        assert abs(got - g2_0) < 0.03


def test_side_peaks_are_uniform_without_memory():
    """With a = 0 and g2_0 = 1 every peak has the same height."""
    cfg = HBTConfig(tau_max=60.5, n_bins=484, sigma_irf=0.35)
    mu = expected_hist_pulsed(cfg, 10.0, 2e5, 1.0, 1.0, a=0.0)
    c = cfg.bin_centers
    peaks = [mu[np.abs(c - k * T_REP_NS) < 1.0].max() for k in (-2, -1, 0, 1, 2)]
    assert np.ptp(peaks) / np.mean(peaks) < 0.01
