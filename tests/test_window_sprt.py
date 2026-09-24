"""The model-free sequential test (new in 0.10.0): its error bounds hold
over whole ranges of g and whatever the count rate does."""
import numpy as np
import pytest

from sparq import EmitterSite, HBTConfig, expected_histogram
from sparq.sequential import ACCEPT, CONTINUE, REJECT, WindowSPRT

CFG = HBTConfig()


def _runs(g, n, seed, rate):
    """n runs on a histogram whose central bin sits at g times the flat
    level; `rate` draws the flat level of each increment."""
    rng = np.random.default_rng(seed)
    shape = np.ones(CFG.n_bins)
    shape[CFG.n_bins // 2] = g
    out = []
    for _ in range(n):
        t = WindowSPRT(0.25, 0.5, CFG)
        for _ in range(100000):
            if t.update(rng.poisson(rate(rng) * shape)) != CONTINUE:
                break
        out.append(t.decision)
    return np.array(out)


@pytest.mark.parametrize("rate", [lambda r: 2.0,
                                  lambda r: 0.5 + 3.0 * r.random()],
                         ids=["steady", "changing"])
def test_error_bounds_hold_over_the_ranges(rate):
    t = WindowSPRT(0.25, 0.5, CFG)
    b_acc, b_rej = t.error_bounds
    n = 500
    for g in (0.5, 0.7):
        d = _runs(g, n, int(100 * g), rate)
        assert np.all(d != CONTINUE)
        assert np.mean(d == ACCEPT) <= b_acc + 3 * np.sqrt(b_acc / n)
    for g in (0.25, 0.1):
        d = _runs(g, n, int(1000 * g), rate)
        assert np.mean(d == REJECT) <= b_rej + 3 * np.sqrt(b_rej / n)
    # far from the boundary the wrong decision is rare
    assert np.mean(_runs(0.9, 200, 7, rate) == ACCEPT) < 0.01


def test_wald_counts_and_additivity():
    t = WindowSPRT(0.25, 0.5, CFG, alpha=1e-6, beta=1e-6)
    rng = np.random.default_rng(3)
    x = rng.poisson(np.full(CFG.n_bins, 3.0)).astype(float)
    t.update(x)
    u = WindowSPRT(0.25, 0.5, CFG, alpha=1e-6, beta=1e-6)
    u.update(x / 2)
    u.update(x / 2)
    assert np.isclose(t.llr, u.llr, rtol=1e-12)
    n_acc, n_rej = WindowSPRT(0.25, 0.5, CFG).expected_window_counts()
    assert n_acc > 0 and n_rej > 0
    # on a real site's histogram increments it decides
    site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                            rho=0.95, blinking=False), 1)
    mu = expected_histogram(site, 0.05, CFG)
    w = WindowSPRT(0.25, 0.5, CFG)
    while w.decision == CONTINUE:
        w.update(rng.poisson(mu))
    assert w.decision == ACCEPT


def test_window_sprt_refusals():
    with pytest.raises(ValueError):
        WindowSPRT(0.5, 0.25)
    with pytest.raises(ValueError):
        WindowSPRT(0.25, 0.5, alpha=0.0)
    with pytest.raises(ValueError):
        WindowSPRT(0.25, 0.5, HBTConfig(n_bins=120))
    t = WindowSPRT(0.25, 0.5)
    with pytest.raises(ValueError):
        t.update(np.zeros(5))
    with pytest.raises(ValueError):
        t.update(-np.ones(CFG.n_bins))
