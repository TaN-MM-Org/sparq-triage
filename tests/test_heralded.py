"""Heralded-source anchors: both closed forms verified against their
defining quadratics (an independent algebraic path); both inversions
are exact round trips to machine precision; the smallest possible CAR
gives exactly g2 = 1 (Poissonian, CAR = 1) and 3/2 (thermal, raw
CAR = 2); the large-CAR falloffs 2/CAR and 4/CAR are asserted as
limits; monotonicity throughout; and refusals for impossible inputs.
0.10.0: the thermal branch takes the raw CAR C/A like the Poissonian
one; the 0.9.1 values are its net-CAR form."""
import numpy as np
import pytest

from sparq import car_for_purity, heralded_g2_limit


def test_closed_forms_solve_their_quadratics():
    """Independent check: the returned g2 must satisfy the defining
    relation of each distribution exactly."""
    for c in (1.0, 2.5, 16.07, 2220.0):
        g = heralded_g2_limit(c)
        assert abs(g * c ** 2 - 2.0 * c + 1.0) < 1e-9 * max(c, 1.0)
        ct = c + 1.0                       # raw thermal CAR >= 2
        gt = heralded_g2_limit(ct, "thermal")
        assert abs(gt * ct ** 2 - 4.0 * ct + 2.0) < 1e-9 * ct
        # the net-CAR form is the 0.9.1 thermal formula
        gn = heralded_g2_limit(c, "thermal", car_definition="net")
        assert abs(gn - (4.0 * c + 2.0) / (c + 1.0) ** 2) < 1e-15


def test_boundaries_and_limits():
    assert heralded_g2_limit(1.0) == 1.0
    # the heralded remnant of thermal bunching: exactly 3/2 at the
    # smallest raw CAR a single-mode source gives, 2
    assert heralded_g2_limit(2.0, "thermal") == 1.5
    with pytest.raises(ValueError, match=">= 2"):
        heralded_g2_limit(1.5, "thermal")
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
            if stats == "thermal":
                g = min(1.5 * g, 1.5)
            c = car_for_purity(g, stats)
            assert c >= (2.0 if stats == "thermal" else 1.0)
            assert abs(heralded_g2_limit(c, stats) - g) < 1e-12 * g \
                + 1e-15
        for c in (2.5, 30.0, 4000.0):
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
