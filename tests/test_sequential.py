"""Sequential certification and the profile-likelihood interval, validated
empirically against the twin (no torch required)."""
import numpy as np
import pytest

from sparq.analysis import profile_likelihood_ci
from sparq.physics import EmitterSite, HBTConfig, expected_histogram
from sparq.sequential import ACCEPT, CONTINUE, REJECT, SPRTCertifier

PARAMS = dict(platform="NV", tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
              rho=0.95, blinking=False, t_on_ms=10.0, t_off_ms=1.0)


def _sites():
    return EmitterSite(dict(PARAMS), 1), EmitterSite(dict(PARAMS), 2)


def _run(cert_args, site, seed, dt=0.05, tmax=120.0):
    s1, s0, cfg = cert_args
    cert = SPRTCertifier(s1, s0, cfg)
    rng = np.random.default_rng(seed)
    mu = expected_histogram(site, dt, cfg)
    while cert.decision == CONTINUE and cert.T_total < tmax:
        cert.update(rng.poisson(mu), dt)
    return cert.decision, cert.T_total


def test_llr_is_additive_in_increments():
    s1, s0 = _sites()
    cfg = HBTConfig()
    rng = np.random.default_rng(3)
    counts = rng.poisson(expected_histogram(s1, 0.2, cfg)).astype(float)
    a = SPRTCertifier(s1, s0, cfg, alpha=1e-6, beta=1e-6)
    a.update(counts, 0.2)
    b = SPRTCertifier(s1, s0, cfg, alpha=1e-6, beta=1e-6)
    b.update(counts / 2, 0.1)
    b.update(counts / 2, 0.1)
    assert np.isclose(a.llr, b.llr, rtol=1e-12)


def test_error_rates_within_nominal_bounds():
    """Empirical error rates over 80 + 80 twin runs stay near the nominal
    alpha = beta = 0.05 (binomial fluctuation allowed for)."""
    s1, s0 = _sites()
    args = (s1, s0, HBTConfig())
    acc = [_run(args, s1, 10_000 + i) for i in range(80)]
    rej = [_run(args, s0, 20_000 + i) for i in range(80)]
    false_reject = sum(d == REJECT for d, _ in acc) / len(acc)
    false_accept = sum(d == ACCEPT for d, _ in rej) / len(rej)
    assert false_reject <= 0.125
    assert false_accept <= 0.125
    assert all(d != CONTINUE for d, _ in acc + rej)     # always decides


def test_stopping_times_near_wald_prediction():
    """Mean decision times land within a factor of two of Wald's
    approximation (which ignores threshold overshoot), and orders of
    magnitude below a fixed 30 s acquisition."""
    s1, s0 = _sites()
    cert = SPRTCertifier(s1, s0, HBTConfig())
    t1_pred, t0_pred = cert.expected_times()
    args = (s1, s0, HBTConfig())
    t_acc = np.mean([t for d, t in (_run(args, s1, 30_000 + i) for i in range(40))
                     if d == ACCEPT])
    t_rej = np.mean([t for d, t in (_run(args, s0, 40_000 + i) for i in range(40))
                     if d == REJECT])
    assert t1_pred <= t_acc <= 2.0 * t1_pred
    assert t0_pred <= t_rej <= 2.0 * t0_pred
    assert t_acc < 2.0 and t_rej < 2.0                  # vs 30 s fixed dwell


def test_certifier_input_validation():
    s1, s0 = _sites()
    with pytest.raises(ValueError):
        SPRTCertifier(s1, s0, alpha=0.0)
    cert = SPRTCertifier(s1, s0)
    with pytest.raises(ValueError):
        cert.update(np.zeros(5), 0.1)                   # wrong grid
    with pytest.raises(ValueError):
        cert.update(np.zeros(cert.cfg.n_bins), 0.0)     # non-positive time


def test_profile_ci_covers_truth_and_shrinks():
    cfg = HBTConfig()
    rng = np.random.default_rng(0)
    site = EmitterSite(dict(PARAMS), 1)
    widths = []
    for T in (10.0, 60.0):
        hist = rng.poisson(expected_histogram(site, T, cfg)).astype(float)
        r = profile_likelihood_ci(hist, T, PARAMS["rate_kcps"] * 1e3, cfg)
        assert r["lo"] <= site.g2_0 <= r["hi"]
        widths.append(r["hi"] - r["lo"])
    assert widths[1] < 0.5 * widths[0]                  # more data, tighter CI


def test_profile_ci_separates_single_from_pair():
    cfg = HBTConfig()
    rng = np.random.default_rng(2)
    single, pair = _sites()
    h1 = rng.poisson(expected_histogram(single, 30.0, cfg)).astype(float)
    h2 = rng.poisson(expected_histogram(pair, 30.0, cfg)).astype(float)
    r1 = profile_likelihood_ci(h1, 30.0, PARAMS["rate_kcps"] * 1e3, cfg)
    r2 = profile_likelihood_ci(h2, 30.0, PARAMS["rate_kcps"] * 1e3, cfg)
    assert r1["hi"] < 0.5                               # certified single
    assert r2["lo"] > 0.4                               # excluded as single
    assert r2["lo"] <= pair.g2_0 <= r2["hi"]
