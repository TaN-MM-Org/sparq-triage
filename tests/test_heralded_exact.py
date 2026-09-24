"""Exact heralded-source model (new in 0.10.0): its low-efficiency limit
is the closed form, a photon-level Monte Carlo agrees with it, and it
shows that the closed form is not a lower limit."""
import numpy as np
import pytest

from sparq.heralded import (car_for_purity, heralded_from_car,
                            heralded_g2_limit, heralded_source)


@pytest.mark.parametrize("K", [1.0, 3.0, np.inf])
def test_low_efficiency_limit_is_the_closed_form(K):
    for mu in (0.01, 0.1, 1.0):
        e = heralded_source(mu, K, 1e-6, 1e-6)
        want_car = 1.0 + (0.0 if np.isinf(K) else 1.0 / K) + 1.0 / mu
        assert abs(e["car"] / want_car - 1.0) < 1e-5
        assert abs(e["g2_h"] / heralded_g2_limit(e["car"], modes=K)
                   - 1.0) < 1e-5
    # thermal and Poissonian are K = 1 and K = inf
    assert heralded_g2_limit(7.0, "thermal") == heralded_g2_limit(7.0,
                                                                 modes=1)
    for g in (0.3, 0.01):
        assert abs(heralded_g2_limit(car_for_purity(g, modes=K), modes=K)
                   - g) < 1e-12


def _photons(mu, K, es, ei, ds, di, N, rng):
    n = rng.poisson(mu, N) if np.isinf(K) else \
        rng.negative_binomial(K, K / (K + mu), N)
    H = (rng.binomial(n, ei) > 0) | (rng.random(N) < di)
    nA = rng.binomial(n, es / 2)
    nB = rng.binomial(n - nA, (es / 2) / (1 - es / 2))
    A = (nA > 0) | (rng.random(N) < ds)
    B = (nB > 0) | (rng.random(N) < ds)
    S = ((nA + nB) > 0) | (rng.random(N) < ds)
    return H, A, B, S


@pytest.mark.parametrize("pars", [
    (0.3, np.inf, 0.5, 0.6, 0.0, 0.0),
    (0.3, 1.0, 0.5, 0.6, 0.0, 0.0),
    (0.1, 2.0, 0.3, 0.9, 1e-3, 5e-3),
    (1.0, np.inf, 0.8, 0.8, 0.0, 0.0)])
def test_photon_monte_carlo_agrees(pars):
    rng = np.random.default_rng(0)
    cars, g2s = [], []
    for _ in range(20):
        H, A, B, S = _photons(*pars, 200_000, rng)
        ph = H.mean()
        cars.append((S & H).mean() / (S.mean() * ph))
        g2s.append((A & B & H).mean() * ph / ((A & H).mean()
                                              * (B & H).mean()))
    e = heralded_source(*pars)
    for est, want in ((cars, e["car"]), (g2s, e["g2_h"])):
        m, se = np.mean(est), np.std(est, ddof=1) / np.sqrt(len(est))
        assert abs(m - want) < 4 * se + 1e-3 * want


def test_closed_form_is_not_a_lower_limit():
    """At the same measured CAR: an efficient herald detector gives a
    much lower g2_h than the low-efficiency formula; dark counts give a
    higher one."""
    e = heralded_source(0.05, np.inf, eta_s=0.05, eta_i=0.95)
    assert e["g2_h"] < 0.6 * heralded_g2_limit(e["car"])
    e = heralded_source(0.01, np.inf, 0.05, 0.05, dark_s=1e-4,
                        dark_i=1e-4)
    assert e["g2_h"] > 1.1 * heralded_g2_limit(e["car"])


def test_inverting_a_measured_car():
    e = heralded_source(0.02, 2.0, 0.2, 0.4)
    sols = heralded_from_car(e["car"], 2.0, 0.2, 0.4)
    assert len(sols) == 1
    assert abs(sols[0]["mu"] / 0.02 - 1) < 1e-8
    assert abs(sols[0]["g2_h"] - e["g2_h"]) < 1e-10
    # with dark counts the same CAR has two solutions; the herald
    # probability tells them apart
    sols = heralded_from_car(20.0, np.inf, 0.05, 0.05, 1e-4, 1e-4)
    assert len(sols) == 2 and sols[0]["mu"] < sols[1]["mu"]
    assert sols[0]["p_herald"] < sols[1]["p_herald"]
    for s in sols:
        assert abs(s["car"] - 20.0) < 1e-8
    assert heralded_from_car(1e6, np.inf, 0.05, 0.05, 1e-4, 1e-4) == []


def test_perfect_signal_efficiency_is_finite():
    a = heralded_source(0.1, np.inf, eta_s=1.0, eta_i=0.5)
    b = heralded_source(0.1, np.inf, eta_s=1.0 - 1e-9, eta_i=0.5)
    for k in ("car", "g2_h"):
        assert np.isfinite(a[k]) and abs(a[k] - b[k]) < 1e-6 * abs(b[k])


def test_refusals():
    with pytest.raises(ValueError):
        heralded_source(0.1, np.inf, eta_s=0.0)
    with pytest.raises(ValueError):
        heralded_source(0.1, np.inf, dark_i=1.0)
    with pytest.raises(ValueError):
        heralded_source(-0.1)
    with pytest.raises(ValueError):
        heralded_g2_limit(10.0, car_definition="gross")
    with pytest.raises(ValueError):
        heralded_g2_limit(10.0, modes=0)
