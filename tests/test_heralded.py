"""Heralded-source anchors: both closed forms verified against their
defining quadratics (an independent algebraic path); both inversions
are exact round trips to machine precision; the boundary CAR = 1
gives exactly g2 = 1 in both conventions; the large-CAR falloffs
2/CAR and 4/CAR are asserted as limits; monotonicity throughout; and
refusals for uncertifiable inputs."""
import numpy as np
import pytest

from sparq import car_for_purity, heralded_g2_limit


def test_closed_forms_solve_their_quadratics():
    """Independent check: the returned g2 must satisfy the defining
    relation of each distribution exactly."""
    for c in (1.0, 2.5, 16.07, 2220.0):
        g = heralded_g2_limit(c)
        assert abs(g * c ** 2 - 2.0 * c + 1.0) < 1e-9 * max(c, 1.0)
        gt = heralded_g2_limit(c, "thermal")
        assert abs(gt * (c + 1.0) ** 2 - 4.0 * c - 2.0) \
            < 1e-9 * max(c, 1.0)


def test_boundaries_and_limits():
    assert heralded_g2_limit(1.0) == 1.0
    # the heralded remnant of thermal bunching: exactly 3/2 at CAR=1
    assert heralded_g2_limit(1.0, "thermal") == 1.5
    for c in (1e3, 1e5):
        assert abs(heralded_g2_limit(c) - 2.0 / c) < 2.0 / c ** 2
        assert abs(heralded_g2_limit(c, "thermal") - 4.0 / c) \
            < 20.0 / c ** 2
    cs = np.geomspace(1.0, 1e4, 60)
    g = heralded_g2_limit(cs)
    assert np.all(np.diff(g) < 0.0)          # purity improves with CAR


def test_inversions_exact_round_trips():
    for stats in ("poissonian", "thermal"):
        for g in (1.0, 0.5, 0.01, 1e-3, 1e-6):
            c = car_for_purity(g, stats)
            assert c >= 1.0
            assert abs(heralded_g2_limit(c, stats) - g) < 1e-12 * g \
                + 1e-15
        for c in (2.0, 30.0, 4000.0):
            g = heralded_g2_limit(c, stats)
            assert abs(car_for_purity(g, stats) - c) < 1e-9 * c


def test_thermal_needs_higher_car_than_poissonian():
    """A thermal pair distribution is noisier: for the same target
    purity it always demands a larger CAR (asserted numerically over
    a wide range, consistent with the 4/CAR vs 2/CAR falloffs)."""
    for g in (0.5, 0.1, 1e-3):
        assert car_for_purity(g, "thermal") > car_for_purity(g)


def test_refusals():
    with pytest.raises(ValueError, match=">= 1"):
        heralded_g2_limit(0.5)
    with pytest.raises(ValueError, match="statistics"):
        heralded_g2_limit(10.0, "chaotic")
    with pytest.raises(ValueError, match="0, 1"):
        car_for_purity(0.0)
    with pytest.raises(ValueError, match="0, 1"):
        car_for_purity(1.5)
