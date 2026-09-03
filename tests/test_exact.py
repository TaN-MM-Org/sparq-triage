"""The master-equation reference against the analytic two-exponential form."""
import numpy as np

from sparq.exact import (
    effective_params,
    g2_exact,
    liouvillian,
    rates_from_site,
    steady_state,
)
from sparq.physics import g2_three_level


def test_steady_state_is_a_null_vector_and_normalized():
    M = liouvillian(0.3, 0.5, 0.05, 0.01)
    p = steady_state(M)
    assert np.abs(M @ p).max() < 1e-12
    assert np.isclose(p.sum(), 1.0)
    assert (p > 0).all()


def test_g2_exact_vanishes_at_zero_and_approaches_one():
    args = (0.3, 0.5, 0.05, 0.01)
    assert abs(g2_exact(np.array([0.0]), *args)[0]) < 1e-12
    assert abs(g2_exact(np.array([1e5]), *args)[0] - 1.0) < 1e-9


def test_exact_g2_is_the_two_exponential_form():
    """For real Liouvillian spectra (the photophysical regime), g2_exact must
    equal g2_three_level with the eigen-decomposed (tau1, tau2, a) to
    machine precision, as the module docstring states."""
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(30):
        tau1 = rng.uniform(1, 30)
        tau2 = rng.uniform(50, 500)
        a = rng.uniform(0.05, 1.5)
        k = rates_from_site(tau1, tau2, a)
        w = np.linalg.eigvals(liouvillian(*k))
        if np.abs(w.imag).max() > 1e-12:
            continue                       # oscillatory regime, form differs
        t1, t2, ae = effective_params(*k)
        tau = np.linspace(0, 5 * tau2, 400)
        d = np.abs(g2_exact(tau, *k) - g2_three_level(tau, t1, t2, ae)).max()
        assert d < 1e-10
        checked += 1
    assert checked >= 25


def test_effective_params_orders_the_timescales():
    k = rates_from_site(12.0, 250.0, 0.4)
    t1, t2, a = effective_params(*k)
    assert 0 < t1 < t2
    assert a > 0
