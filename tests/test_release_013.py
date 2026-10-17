"""Tests of the 0.13.0 changes: the pulsed simulator's comb now keeps
the tails of far peaks, the correlators and the detector dead time run
without a Python loop over the clicks (same results), the saturation
fit, and the exact interval and test for the window g2."""
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq, minimize
from scipy.stats import binom

import sparq
from sparq import (EmitterSite, HBTConfig, bayesian_g2, count_ratio_ci,
                   expected_hist_pulsed, expected_histogram, fit_pulsed,
                   fit_saturation, saturation_curve, saturation_from_rates,
                   window_g2_ci)
from sparq.exact import liouvillian, steady_state
from sparq.pulsed import peak_shape

SITE = dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0, rho=0.95,
            blinking=False)


# ----------------------------------------------------------------------
# pulsed simulator: the comb keeps the tails of far peaks
# ----------------------------------------------------------------------

def _comb_reference(cfg, T_s, rate, g2_0, tau_e, sigma_pair, t_rep=12.5,
                    n_peaks=80):
    """The model of expected_hist_pulsed summed over 2 n_peaks + 1
    peaks, written here from the public peak_shape."""
    A = (0.5 * rate) ** 2 * t_rep * 1e-9 * T_s
    mu = np.zeros(cfg.n_bins)
    for k in range(-n_peaks, n_peaks + 1):
        amp = g2_0 if k == 0 else 1.0
        mu += amp * peak_shape(cfg.bin_centers - k * t_rep, tau_e,
                               sigma_pair)
    return mu * A * cfg.bin_width


def test_pulsed_simulator_keeps_far_peak_tails():
    """Up to 0.12.1 the comb stopped one period beyond the record: for an
    8 ns lifetime at 80 MHz the edge bins were 3.2 % low, and the exact
    comb fit of the noise-free simulated histogram gave g2(0) = 0.1031
    instead of 0.12 (fitted lifetime 7.65 ns)."""
    cfg = HBTConfig(tau_max=60.5, n_bins=484)
    for tau_e in (1.0, 5.0, 8.0):
        mu = expected_hist_pulsed(cfg, 10.0, 2e5, g2_0=0.12, tau_e=tau_e,
                                  sigma_pair=0.5)
        ref = _comb_reference(cfg, 10.0, 2e5, 0.12, tau_e, 0.5)
        assert np.max(np.abs(mu / ref - 1.0)) < 1e-12
    fit = fit_pulsed(cfg.bin_centers, mu, interval=False)
    assert abs(fit["g2_0"] - 0.12) < 1e-4
    assert abs(fit["params"]["tau_e"] - 8.0) < 1e-3


def test_pulsed_simulator_refuses_bad_shapes():
    cfg = HBTConfig(tau_max=60.5, n_bins=484)
    for kw in (dict(tau_e=0.0), dict(tau_e=np.inf), dict(t_rep=0.0),
               dict(sigma_pair=-0.1), dict(sigma_pair=np.nan),
               dict(center_off=np.nan)):
        args = dict(tau_e=1.0)
        args.update(kw)
        with pytest.raises(ValueError):
            expected_hist_pulsed(cfg, 10.0, 2e5, 0.12, **args)


# ----------------------------------------------------------------------
# correlators and dead time: the same results as the 0.12.1 loops
# ----------------------------------------------------------------------

def _correlate_0121(t_a, t_b, cfg):
    """sparq.physics.correlate as it was up to 0.12.1."""
    hist = np.zeros(cfg.n_bins)
    if len(t_a) == 0 or len(t_b) == 0:
        return hist
    lo = np.searchsorted(t_b, t_a - cfg.tau_max)
    hi = np.searchsorted(t_b, t_a + cfg.tau_max)
    diffs = []
    for i in range(len(t_a)):
        if hi[i] > lo[i]:
            diffs.append(t_b[lo[i]:hi[i]] - t_a[i])
    if not diffs:
        return hist
    d = np.concatenate(diffs)
    hist, _ = np.histogram(d, bins=cfg.n_bins,
                           range=(-cfg.tau_max, cfg.tau_max))
    return hist.astype(np.float32)


def _start_stop_0121(t_a, t_b, cfg):
    """sparq.timetags.correlate_start_stop as it was up to 0.12.1."""
    hist = np.zeros(cfg.n_bins, dtype=np.float32)
    if len(t_a) == 0 or len(t_b) == 0:
        return hist
    lo = np.searchsorted(t_b, t_a - cfg.tau_max)
    hi = np.searchsorted(t_b, t_a + cfg.tau_max)
    first = []
    for i in range(len(t_a)):
        if hi[i] > lo[i]:
            first.append(t_b[lo[i]] - t_a[i])
    if not first:
        return hist
    h, _ = np.histogram(np.asarray(first), bins=cfg.n_bins,
                        range=(-cfg.tau_max, cfg.tau_max))
    return h.astype(np.float32)


def _deadtime_0121(t, td):
    """The dead-time pass of sparq.physics._detector_chain up to 0.12.1."""
    keep = np.ones(len(t), bool)
    last = -np.inf
    for i in range(len(t)):
        if t[i] - last >= td:
            last = t[i]
        else:
            keep[i] = False
    return keep


def _streams():
    rng = np.random.default_rng(11)
    out = []
    site = EmitterSite(dict(SITE), n_emitters=1)
    out.append(sparq.simulate_photon_stream(site, 0.2, rng))
    for n, span in ((3000, 1e4), (4000, 2e5), (50, 1e3)):
        a = np.sort(rng.uniform(0, span, n))
        b = np.sort(rng.uniform(0, span, n))
        a[::5] = np.round(a[::5])          # delays on bin edges and
        b[::7] = np.round(b[::7])          # duplicate times
        out.append((np.sort(a), np.sort(b)))
    out.append((np.array([0.0, 60.5, 121.0]), np.array([60.5, 121.0])))
    out.append((np.array([]), np.array([1.0])))
    out.append((np.array([0.0]), np.array([1000.0])))
    return out


def test_correlators_equal_the_0121_loops():
    for cfg in (HBTConfig(), HBTConfig(tau_max=10.0125, n_bins=801)):
        for t_a, t_b in _streams():
            new = sparq.correlate(t_a, t_b, cfg)
            old = _correlate_0121(t_a, t_b, cfg)
            assert new.dtype == old.dtype and np.array_equal(new, old)
            new = sparq.correlate_start_stop(t_a, t_b, cfg)
            old = _start_stop_0121(t_a, t_b, cfg)
            assert new.dtype == old.dtype and np.array_equal(new, old)
    # t_a in any order, and many pairs (several blocks of pairs)
    rng = np.random.default_rng(3)
    t_a = rng.uniform(0, 3e4, 20000)
    t_b = np.sort(rng.uniform(0, 3e4, 20000))
    old_block = sparq.physics._PAIR_BLOCK
    try:
        sparq.physics._PAIR_BLOCK = 50_000
        new = sparq.correlate(t_a, t_b, HBTConfig())
    finally:
        sparq.physics._PAIR_BLOCK = old_block
    assert new.sum() > 4 * 50_000
    assert np.array_equal(new, _correlate_0121(t_a, t_b, HBTConfig()))


def test_deadtime_equals_the_0121_loop():
    rng = np.random.default_rng(5)
    for mean_gap, td in ((30.0, 45.0), (5e3, 45.0), (30.0, 1000.0),
                         (30.0, 0.0), (10.0, 10.0)):
        t = np.sort(np.cumsum(rng.exponential(mean_gap, 20000)))
        t[100:110] = t[100]                 # equal times
        t = np.round(t, 1)                  # gaps of exactly td
        t = np.sort(t)
        new = sparq.physics._deadtime_keep(t, td)
        assert np.array_equal(new, _deadtime_0121(t, td))
    for t in (np.array([]), np.array([3.0])):
        assert np.array_equal(sparq.physics._deadtime_keep(t, 45.0),
                              _deadtime_0121(t, 45.0))
    with pytest.raises(ValueError):
        sparq.physics._deadtime_keep(np.arange(5.0), np.nan)


def test_correlators_refuse_non_finite_tags():
    cfg = HBTConfig()
    good = np.array([0.0, 10.0, 20.0])
    for bad in (np.array([0.0, 10.0, np.nan]), np.array([0.0, np.inf])):
        for f in (sparq.correlate, sparq.correlate_start_stop):
            with pytest.raises(ValueError):
                f(bad, good, cfg)
            with pytest.raises(ValueError):
                f(good, bad, cfg)


# ----------------------------------------------------------------------
# saturation curve
# ----------------------------------------------------------------------

def test_saturation_law_is_the_three_level_steady_state():
    """R_inf k/(k + k_sat) against k_r p_e from the steady state of the
    rate matrix (an eigenvector, sparq.exact), for random rates."""
    rng = np.random.default_rng(0)
    for _ in range(30):
        k_r, k_es, k_se = np.exp(rng.uniform(np.log([0.01, 1e-4, 1e-4]),
                                             np.log([1.0, 0.5, 0.5])))
        s = saturation_from_rates(k_r, k_es, k_se)
        for k_exc in np.geomspace(1e-3, 1e2, 7) * k_r:
            pe = steady_state(liouvillian(k_exc, k_r, k_es, k_se))[1]
            law = s["R_inf"] * k_exc / (k_exc + s["k_sat"])
            assert abs(law / (k_r * pe) - 1.0) < 1e-12
    with pytest.raises(ValueError):
        saturation_from_rates(0.0, 0.1, 0.1)


P_SAT = np.array([10, 20, 40, 70, 100, 150, 200, 300, 450, 650, 900, 1300.0])
TRUE = dict(R_inf=1.2e6, P_sat=300.0, bg_slope=150.0, dark=300.0)


def _nll_full(P, n, t, th):
    """Poisson NLL of the curve in (log R_inf, log P_sat, b, d), written
    independently of sparq.saturation."""
    R, Ps, b, d = np.exp(th[0]), np.exp(th[1]), th[2], th[3]
    if b < 0 or d < 0:
        return 1e30
    mu = t * (R * P / (P + Ps) + b * P + d)
    return float(np.sum(mu - n * np.log(mu)))


def _nm(f, x0):
    best = None
    x = np.array(x0, float)
    for _ in range(3):
        r = minimize(f, x, method="Nelder-Mead",
                     options=dict(xatol=1e-10, fatol=1e-10, maxiter=40000,
                                  maxfev=40000))
        x = r.x
        if best is None or r.fun < best.fun:
            best = r
    return best


def test_saturation_fit_recovers_exact_rates():
    rate = saturation_curve(P_SAT, **TRUE)
    t = 1e4                                  # long counts: nearly exact
    r = fit_saturation(P_SAT, np.round(rate * t), t)
    assert abs(r["R_inf"] / TRUE["R_inf"] - 1) < 1e-6
    assert abs(r["P_sat"] / TRUE["P_sat"] - 1) < 1e-6
    assert abs(r["bg_slope"] / TRUE["bg_slope"] - 1) < 1e-4
    assert abs(r["dark"] / TRUE["dark"] - 1) < 1e-3
    assert r["saturated"]


def test_saturation_fit_and_interval_against_independent_profile():
    rng = np.random.default_rng(1)
    t = 0.5
    n = rng.poisson(saturation_curve(P_SAT, **TRUE) * t).astype(float)
    r = fit_saturation(P_SAT, n, t)
    # the best fit: an independent Nelder-Mead minimum is not lower
    full = _nm(lambda th: _nll_full(P_SAT, n, t, th),
               [np.log(1e6), np.log(250.0), 100.0, 200.0])
    th_hat = [np.log(r["R_inf"]), np.log(r["P_sat"]), r["bg_slope"],
              r["dark"]]
    assert _nll_full(P_SAT, n, t, th_hat) <= full.fun + 1e-6
    # at the ends of both intervals the profile deviance, computed
    # independently, is the chi-square quantile 3.84
    for name, i in (("P_sat", 1), ("R_inf", 0)):
        for end in (r[name + "_lo"], r[name + "_hi"]):
            def prof(rest):
                th = np.insert(rest, i, np.log(end))
                return _nll_full(P_SAT, n, t, th)
            x0 = np.delete(np.array(th_hat), i)
            d = 2.0 * (_nm(prof, x0).fun - full.fun)
            assert abs(d - 3.8415) < 0.01, (name, end, d)
    assert 0.01 < r["p_value"] <= 1.0
    # the true values lie inside
    assert r["P_sat_lo"] < TRUE["P_sat"] < r["P_sat_hi"]
    assert r["R_inf_lo"] < TRUE["R_inf"] < r["R_inf_hi"]
    assert np.allclose(r["rho"], r["signal_rate"]
                       / saturation_curve(P_SAT, r["R_inf"], r["P_sat"],
                                          r["bg_slope"], r["dark"]))


def test_saturation_interval_coverage():
    """95 % intervals for P_sat over 40 simulated curves (dim emitter,
    160 to 4600 counts per point): at least 34 cover the truth
    (expected 38; 34 is three standard errors below). Done once outside
    the suite with 200 curves each of this emitter and a bright one:
    see the README."""
    true = dict(R_inf=2e4, P_sat=300.0, bg_slope=5.0, dark=100.0)
    rng = np.random.default_rng(7)
    hits = 0
    for _ in range(40):
        n = rng.poisson(saturation_curve(P_SAT, **true) * 0.2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r = fit_saturation(P_SAT, n, 0.2, interval="P_sat")
        hi = r["P_sat_hi"] if np.isfinite(r["P_sat_hi"]) else np.inf
        hits += r["P_sat_lo"] <= true["P_sat"] <= hi
    assert hits >= 34


def test_saturation_fit_flags_unsaturated_data_and_wrong_shape():
    rng = np.random.default_rng(2)
    P = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])      # far below P_sat
    n = rng.poisson(saturation_curve(P, 2e4, 300.0, 5.0, 100.0) * 0.2)
    with pytest.warns(RuntimeWarning, match="saturation"):
        r = fit_saturation(P, n, 0.2)
    assert not r["saturated"]
    assert np.isnan(r["P_sat_hi"]) or np.isnan(r["R_inf_hi"])
    # a pulsed-like curve R (1 - exp(-P/P0)) is not this law: rejected
    rate = 1.2e6 * (1 - np.exp(-P_SAT / 300.0)) + 300.0
    r = fit_saturation(P_SAT, rng.poisson(rate * 1.0), 1.0)
    assert r["p_value"] < 1e-6


def test_saturation_fit_refusals():
    P, n = P_SAT[:6], np.full(6, 100.0)
    bad = [dict(power=-P), dict(counts=n + 0.5), dict(counts=-n),
           dict(power=np.r_[P[:5], np.nan]), dict(t_s=0.0),
           dict(background="quadratic"), dict(level=1.0),
           dict(power=np.r_[P[:3], P[:3]]), dict(counts=n[:5]),
           dict(counts=0 * n), dict(power=0 * P), dict(interval="tau")]
    for kw in bad:
        args = dict(power=P, counts=n, t_s=1.0)
        args.update(kw)
        with pytest.raises(ValueError):
            fit_saturation(**args)
    # the simpler curves need fewer powers
    fit_saturation(P[:3], n[:3], 1.0, background="none", interval=None)


# ----------------------------------------------------------------------
# exact interval and test for the window g2
# ----------------------------------------------------------------------

def _cp_by_root(k0, k, level):
    """Clopper-Pearson limits solved from binomial tail sums."""
    a = 0.5 * (1 - level)
    lo = 0.0 if k0 == 0 else brentq(
        lambda p: binom.sf(k0 - 1, k, p) - a, 1e-15, 1 - 1e-15, xtol=1e-15)
    hi = 1.0 if k0 == k else brentq(
        lambda p: binom.cdf(k0, k, p) - a, 1e-15, 1 - 1e-15, xtol=1e-15)
    return lo, hi


def test_count_ratio_ci_equals_clopper_pearson_from_tail_sums():
    for k0, kr, n0, nr, level in ((3, 853, 1, 36, 0.95), (0, 40, 1, 42, 0.9),
                                  (25, 0, 3, 10, 0.95), (120, 900, 1, 42,
                                                        0.68),
                                  (7, 7, 2, 2, 0.99)):
        r = count_ratio_ci(k0, kr, n0, nr, level=level)
        p_lo, p_hi = _cp_by_root(k0, k0 + kr, level)
        g = lambda p: nr * p / (n0 * (1 - p)) if p < 1 else np.inf
        for got, want in ((r["lo"], g(p_lo)), (r["hi"], g(p_hi))):
            assert (got == want == np.inf) or abs(got - want) <= 1e-9 * max(
                1.0, want)
        # the one-sided p-value is the binomial tail at g = 0.5
        p_t = n0 * 0.5 / (n0 * 0.5 + nr)
        tail = sum(binom.pmf(j, k0 + kr, p_t) for j in range(k0 + 1))
        assert abs(r["p_value"] - tail) < 1e-12


def test_window_g2_ci_coverage_and_test_size():
    """On simulated window counts with a rate that changes from run to
    run (2 to 40 counts per reference bin): the 95 % interval covers the
    true window g2 in at least 95 % of 2000 runs (less three standard
    errors), and the test of g >= 0.5 at alpha = 0.05 rejects a site
    with g = 0.5 in at most 5 % of them (plus three standard errors)
    and a site with g = 0.1 in most of them."""
    cfg = HBTConfig()
    rng = np.random.default_rng(9)
    m0 = np.abs(cfg.bin_centers) < 0.5 * cfg.bin_width
    mr = np.abs(cfg.bin_centers) >= 0.65 * cfg.tau_max
    n0, nr = m0.sum(), mr.sum()
    for g_true in (0.1, 0.5):
        cover = reject = 0
        N = 2000
        for _ in range(N):
            lam = rng.uniform(2.0, 40.0)             # counts per far bin
            k0 = rng.poisson(g_true * lam * n0)
            kr = rng.poisson(lam * nr)
            if k0 + kr == 0:
                cover += 1
                continue
            r = count_ratio_ci(k0, kr, n0, nr, level=0.95)
            cover += r["lo"] <= g_true <= r["hi"]
            reject += r["p_value"] <= 0.05
        se = np.sqrt(0.95 * 0.05 / N)
        assert cover / N >= 0.95 - 3 * se
        if g_true == 0.5:
            assert reject / N <= 0.05 + 3 * se
        else:
            assert reject / N > 0.5                    # the test has power


def test_window_g2_ci_agrees_with_bayesian_g2_at_many_counts():
    cfg = HBTConfig()
    site = EmitterSite(dict(SITE), n_emitters=1)
    h = np.random.default_rng(4).poisson(expected_histogram(site, 300.0,
                                                            cfg))
    r = window_g2_ci(h, cfg)
    post = bayesian_g2(h, cfg)
    lo, hi = post.credible_interval(0.95)
    assert abs(r["lo"] / lo - 1) < 0.03 and abs(r["hi"] / hi - 1) < 0.03
    assert r["k0"] == post.k0 and r["kr"] == post.kr
    assert r["p_value"] < 1e-6


def test_window_g2_ci_refusals():
    cfg = HBTConfig()
    with pytest.raises(ValueError):
        window_g2_ci(np.zeros(121), cfg)              # no counts
    with pytest.raises(ValueError):
        window_g2_ci(np.full(121, 1.5), cfg)          # not whole counts
    with pytest.raises(ValueError):
        window_g2_ci(np.ones(120), cfg)
    with pytest.raises(ValueError):
        window_g2_ci(np.ones(121), HBTConfig(n_bins=120))
    for kw in (dict(k0=-1), dict(kr=np.nan), dict(n0=0), dict(level=0.0),
               dict(threshold=-0.5)):
        args = dict(k0=3, kr=10)
        args.update(kw)
        with pytest.raises(ValueError):
            count_ratio_ci(**args)
