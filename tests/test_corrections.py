"""v0.4 exact-correction anchors: the background correction as the
algebraic inverse of the package's own forward model, endpoint-mapped
confidence intervals, and the non-paralyzable dead-time correction
against both its exact round trip and the Monte-Carlo detector chain."""
import numpy as np
import pytest

from sparq import (background_corrected_g2, deadtime_corrected_rate,
                   signal_fraction)
from sparq.physics import (DetectorImpairments, _detector_chain, g2_zero)


def test_background_correction_inverts_the_forward_model_exactly():
    """g2_zero applies g2_meas = 1 + rho^2 (g2_true - 1); the corrector
    must undo it to machine precision for any rho and emitter number."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(1, 5))
        rho = float(rng.uniform(0.2, 1.0))
        gm = g2_zero(20.0, 300.0, 0.5, n_emitters=n, rho=rho)
        gt = g2_zero(20.0, 300.0, 0.5, n_emitters=n, rho=1.0)
        corr = background_corrected_g2(gm, rho)
        assert abs(corr["g2_corrected"] - gt) < 1e-12


def test_background_correction_maps_ci_endpoints():
    """The map is affine increasing, so interval endpoints transform
    individually; the transformed interval still brackets the
    transformed point estimate."""
    out = background_corrected_g2(0.6, 0.8, ci=(0.5, 0.7))
    lo, hi = out["ci"]
    assert lo <= out["g2_corrected"] <= hi
    assert abs(lo - (1.0 + (0.5 - 1.0) / 0.64)) < 1e-12
    assert abs(hi - (1.0 + (0.7 - 1.0) / 0.64)) < 1e-12


def test_background_correction_truncates_at_the_physical_floor():
    out = background_corrected_g2(0.05, 0.3)
    assert out["g2_corrected"] == 0.0
    assert out["g2_uncorrected_inverse"] < 0.0


def test_background_correction_refusals():
    with pytest.raises(ValueError):
        background_corrected_g2(0.5, 0.0)
    with pytest.raises(ValueError):
        background_corrected_g2(0.5, 1.2)
    with pytest.raises(ValueError):
        background_corrected_g2(0.5, 0.8, ci=(0.7, 0.5))


def test_signal_fraction():
    assert signal_fraction(3.0, 1.0) == 0.75
    assert signal_fraction(1.0, 0.0) == 1.0
    with pytest.raises(ValueError):
        signal_fraction(-1.0, 1.0)
    with pytest.raises(ValueError):
        signal_fraction(0.0, 0.0)


def test_deadtime_round_trip_is_exact():
    for r in (1e3, 1e5, 5e6):
        rm = r / (1.0 + r * 45e-9)
        assert abs(deadtime_corrected_rate(rm, 45.0) - r) / r < 1e-12
    # zero dead time is the identity
    assert deadtime_corrected_rate(1e5, 0.0) == 1e5


def test_deadtime_saturation_is_refused():
    with pytest.raises(ValueError):
        deadtime_corrected_rate(1.0 / 45e-9, 45.0)
    with pytest.raises(ValueError):
        deadtime_corrected_rate(-1.0, 45.0)


def test_deadtime_forward_formula_matches_the_monte_carlo_chain():
    """r_meas = r/(1 + r tau_d) is the exact renewal-theory throughput
    of the greedy non-paralyzable pass the detector chain implements;
    one long Poisson stream must land within statistical error."""
    rng = np.random.default_rng(3)
    T_ns, r_true = 2e8, 3e5
    n = rng.poisson(r_true * T_ns * 1e-9)
    t = np.sort(rng.uniform(0.0, T_ns, n))
    imp = DetectorImpairments(dead_time_ns=45.0, afterpulse_p=0.0,
                              sigma_irf_ns=0.0)
    r_mc = len(_detector_chain(t, imp, rng)) / (T_ns * 1e-9)
    pred = r_true / (1.0 + r_true * 45e-9)
    assert abs(r_mc - pred) / pred < 0.02
