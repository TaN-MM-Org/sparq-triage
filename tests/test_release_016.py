"""Tests of the 0.16.0 changes: intervals of the saturation fit that
allow for counts scattering more than Poisson counts
(`fit_saturation(overdispersion=True)`), p-values of
`check_stationarity` by simulation for small counts
(`method="monte_carlo"`), the faster jitter blur and profile-likelihood
interval (the same results as 0.15.0), and the fixes (inputs that gave
silent nonsense, and correlator counts above 2^24).
"""
import itertools
import warnings
from math import comb, factorial, prod

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import erfcx
from scipy.stats import f as f_dist
from scipy.stats import multinomial

import sparq
from sparq import (EmitterSite, HBTConfig, WindowSPRT, bayesian_g2,
                   certification_probability, check_stationarity, correlate,
                   correlate_start_stop, deadtime_throughput,
                   expected_hist_pulsed, expected_histogram,
                   expected_posterior, fit_model, fit_pulsed,
                   fit_saturation, g2_three_level, goodness_of_fit,
                   normalize_g2, profile_likelihood_ci,
                   required_acquisition_time, saturation_curve,
                   simulate_photon_stream, site_from_numbers)
from sparq.physics import _exp_conv_gauss, _exp_or_zero
from sparq.sequential import _homogeneity, _rate_test

SITE = dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0, rho=0.95,
            blinking=False)


# ----------------------------------------------------------------------
# the jitter blur: the same numbers as 0.15.0, in less time
# ----------------------------------------------------------------------

def _blur_0150(tau, T, s):
    """sparq.physics._exp_conv_gauss as it was in 0.15.0."""
    tau = np.asarray(tau, dtype=float)
    if s <= 1e-12:
        return np.exp(-np.abs(tau) / T)
    T = np.asarray(T, dtype=float)
    z = s / T
    g = np.exp(-0.5 * (tau / s) ** 2)
    out = 0.0
    for sg in (1.0, -1.0):
        u = (z - sg * tau / s) / np.sqrt(2.0)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            pos = g * erfcx(u)
            neg = (2.0 * np.exp(0.5 * z ** 2 - sg * tau / T)
                   - g * erfcx(-u))
        out = out + 0.5 * np.where(u >= 0.0, pos, neg)
    return np.clip(out, 0.0, 1.0)


def test_blur_equals_the_0150_code_bit_for_bit():
    """From 0.16.0 erfcx is evaluated once per term and exponentials that
    underflow are not evaluated; the numbers are the same bit for bit,
    for scalar and array delays and lifetimes (also lifetimes far below
    the jitter, where 0.9.1 failed), and with broadcasting as the
    posterior sampler uses it."""
    rng = np.random.default_rng(1)
    tau = np.linspace(-80.0, 80.0, 2001)
    cases = [(tau, T, s) for T in (0.01, 0.05, 0.3, 1.0, 15.0, 250.0, 1e4)
             for s in (0.0, 1e-13, 0.05, 0.49, 3.0)]
    cases += [(0.0, 15.0, 0.49), (3.0, 0.02, 0.49), (-3.0, 0.02, 0.49),
              (np.abs(rng.normal(0.0, 30.0, (61, 8)))[None],
               np.exp(rng.uniform(-3.0, 7.0, 50))[:, None, None], 0.49),
              (tau, np.array([0.1, 1.0, 10.0])[:, None], 0.49)]
    for t, T, s in cases:
        new, old = _exp_conv_gauss(t, T, s), _blur_0150(t, T, s)
        assert np.shape(new) == np.shape(old)
        assert np.array_equal(new, old)
    x = np.r_[np.linspace(-760.0, -700.0, 100001), 0.0, -1.0, np.nan]
    assert np.array_equal(_exp_or_zero(x), np.exp(x), equal_nan=True)


# ----------------------------------------------------------------------
# profile_likelihood_ci: the flat level found exactly, same interval
# ----------------------------------------------------------------------

@pytest.mark.parametrize("prior, g_old, lo_old, hi_old", [
    (None, 0.08188559434847857, 0.06392625551453596, 0.29483850132298506),
    ((1.0, 0.01), 0.10075038648138766, 0.0783543827846084,
     0.12429891160687276)])
def test_profile_interval_equals_the_0150_result(prior, g_old, lo_old,
                                                 hi_old):
    """README example 3. From 0.16.0 the flat-level scale c0 is set to
    its best value for each shape (a closed form) instead of being
    searched, and the search gets the gradient; the result is the same
    as with the 0.15.0 code (the numbers here) to within the 1e-4
    refinement of the ends (measured: at most 3.5e-6), and the ends sit
    at the chi-square quantile (test_release_012 checks that against an
    independent Nelder-Mead profile)."""
    cfg = HBTConfig()
    h = np.random.default_rng(0).poisson(
        expected_histogram(EmitterSite(dict(SITE), 1), 60.0, cfg)
    ).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # tau2 on its limit
        r = profile_likelihood_ci(h, 60.0, 150e3, cfg, c0_prior=prior)
    for new, old in ((r["g2_hat"], g_old), (r["lo"], lo_old),
                     (r["hi"], hi_old)):
        assert abs(new - old) < 2e-5                     # measured 3.5e-6


def test_profile_gradient_matches_central_differences(monkeypatch):
    """The gradient that profile_likelihood_ci gives its search (the
    likelihood at the best flat-level scale c0, differentiated at fixed
    c0) equals central differences of the same objective, at 20 random
    points, with and without the flat-level prior (relative error below
    1e-4; measured 3e-6, the precision of the differences)."""
    import sparq.analysis as A
    captured = []
    real = A.minimize

    def spy(fun, x0, *args, **kw):
        if kw.get("jac") is True and not captured:
            captured.append(fun)
        return real(fun, x0, *args, **kw)

    monkeypatch.setattr(A, "minimize", spy)
    cfg = HBTConfig()
    h = np.random.default_rng(0).poisson(
        expected_histogram(EmitterSite(dict(SITE), 1), 60.0, cfg)
    ).astype(float)
    for prior in (None, (1.0, 0.01)):
        captured.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            profile_likelihood_ci(h, 60.0, 150e3, cfg, c0_prior=prior,
                                  n_grid=3)
        f = captured[0]
        rng = np.random.default_rng(1)
        for _ in range(20):
            y = np.array([np.log(rng.uniform(1.0, 60.0)),
                          np.log(rng.uniform(60.0, 700.0)),
                          rng.uniform(0.05, 2.5)])
            g = f(y)[1]
            for i in range(3):
                e = np.zeros(3)
                e[i] = 1e-6
                fd = (f(y + e)[0] - f(y - e)[0]) / 2e-6
                assert abs(g[i] - fd) < 1e-4 * max(abs(fd), 1e-3)


# ----------------------------------------------------------------------
# correlator counts above 2^24
# ----------------------------------------------------------------------

def test_correlator_keeps_counts_above_float32_precision():
    """4097 x 4097 pairs, all 0.1 ns apart, fall in one bin: 16 785 409
    pairs. 0.15.0 returned float32, which rounds this to 16 785 408. Now
    the histogram is float64 when a bin holds more than 2^24 pairs, and
    float32 as before otherwise."""
    cfg = HBTConfig()
    h = correlate(np.zeros(4097), np.full(4097, 0.1), cfg)
    assert h.dtype == np.float64
    assert h.max() == 4097 ** 2 == 16_785_409
    assert h.sum() == 4097 ** 2
    assert np.float32(4097 ** 2) == 16_785_408           # the old rounding
    small = correlate(np.zeros(5), np.full(5, 0.1), cfg)
    assert small.dtype == np.float32 and small.max() == 25
    ss = correlate_start_stop(np.zeros(1), np.full(1, 0.1), cfg)
    assert ss.dtype == np.float32 and ss.sum() == 1
    big = sparq.physics._exact_counts(np.array([0, 2 ** 24 + 1]))
    assert big.dtype == np.float64 and big[1] == 2 ** 24 + 1


# ----------------------------------------------------------------------
# saturation fit with overdispersed counts
# ----------------------------------------------------------------------

POWER = np.array([0, 50, 100, 200, 300, 450, 600, 900, 1300, 2000.0])
TRUTH = dict(R_inf=600e3, P_sat=350.0, bg_slope=60.0, dark=300.0)


def _nll(P, n, t, th):
    """Poisson NLL of the curve in (log R_inf, log P_sat, b, d), written
    independently of sparq.saturation."""
    R, Ps, b, d = np.exp(th[0]), np.exp(th[1]), th[2], th[3]
    if b < 0 or d < 0:
        return 1e30
    mu = t * (R * P / (P + Ps) + b * P + d)
    return float(np.sum(mu - n * np.log(mu)))


def _nm(f, x0):
    best, x = None, np.array(x0, float)
    for _ in range(3):
        r = minimize(f, x, method="Nelder-Mead",
                     options=dict(xatol=1e-10, fatol=1e-10, maxiter=40000,
                                  maxfev=40000))
        x = r.x
        if best is None or r.fun < best.fun:
            best = r
    return best


def test_overdispersion_estimate_and_interval_ends():
    """Counts that scatter 4 times more than Poisson counts (4 times a
    Poisson count of a quarter of the mean). The dispersion is the
    Pearson sum over the degrees of freedom, recomputed here from the
    returned means (to 1e-12); the fit itself does not change (to
    1e-9: the last linear solve starts from where the interval search
    ended, measured 6e-14); and at
    both ends of the widened P_sat interval the profile deviance,
    computed independently by Nelder-Mead, is phi times the F(1, 6)
    quantile (within 0.01 phi)."""
    rng = np.random.default_rng(3)
    t = 0.5
    n = 4 * rng.poisson(saturation_curve(POWER, **TRUTH) * t / 4)
    plain = fit_saturation(POWER, n, t, interval="P_sat")
    r = fit_saturation(POWER, n, t, interval="P_sat", overdispersion=True)
    mu = r["mu"]
    phi = float(np.sum((n - mu) ** 2 / mu)) / (POWER.size - 4)
    assert abs(r["dispersion"] - phi) < 1e-12 * phi        # measured 5e-14
    assert abs(plain["dispersion"] - phi) < 1e-12 * phi    # measured 5e-14
    assert r["dispersion_used"] == r["dispersion"] and phi > 1.5
    assert plain["dispersion_used"] == 1.0
    for k in ("R_inf", "P_sat", "bg_slope", "dark", "deviance"):
        assert abs(r[k] - plain[k]) <= 1e-9 * abs(plain[k])   # measured 6e-14
    crit = r["dispersion_used"] * f_dist.ppf(0.95, 1, POWER.size - 4)
    full = _nm(lambda th: _nll(POWER, n, t, th),
               [np.log(r["R_inf"]), np.log(r["P_sat"]), r["bg_slope"],
                r["dark"]])
    th_hat = np.array([np.log(r["R_inf"]), np.log(r["P_sat"]),
                       r["bg_slope"], r["dark"]])
    for end in (r["P_sat_lo"], r["P_sat_hi"]):
        prof = _nm(lambda rest: _nll(POWER, n, t,
                                     np.insert(rest, 1, np.log(end))),
                   np.delete(th_hat, 1))
        d = 2.0 * (prof.fun - full.fun)
        # measured 1.4e-7 (relative to phi 4e-8)
        assert abs(d - crit) < 0.01 * r["dispersion_used"], (end, d, crit)
    # the widened interval contains the Poisson one
    assert r["P_sat_lo"] < plain["P_sat_lo"] < plain["P_sat_hi"] \
        < r["P_sat_hi"]


def test_overdispersion_interval_coverage():
    """95 % intervals for P_sat over 100 simulated scans whose counts
    scatter 4 times more than Poisson counts: the Poisson intervals are
    too narrow (measured: 65 of 100 cover the truth; 263 of 400 in a
    longer run outside the suite), the widened ones cover at about the
    stated rate (measured 93 of 100; 381 of 400 outside the suite).
    With counts that drift by 1 % from point to point (a focus that
    moves), 40 scans: Poisson intervals 19, widened 38 (148 and 391 of
    400 outside the suite)."""
    mean = saturation_curve(POWER, **TRUTH) * 0.5
    rng = np.random.default_rng(21)
    hits = {False: 0, True: 0}
    for _ in range(100):
        n = 4 * rng.poisson(mean / 4)
        for od in (False, True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                r = fit_saturation(POWER, n, 0.5, interval="P_sat",
                                   overdispersion=od)
            hi = r["P_sat_hi"] if np.isfinite(r["P_sat_hi"]) else np.inf
            hits[od] += r["P_sat_lo"] <= TRUTH["P_sat"] <= hi
    assert hits[True] >= 89     # measured 93; 95 % less three std. errors
    assert hits[False] <= 80    # measured 65
    hits = {False: 0, True: 0}
    for _ in range(40):
        n = rng.poisson(mean * (1.0 + 0.01 * rng.standard_normal(mean.size)))
        for od in (False, True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                r = fit_saturation(POWER, n, 0.5, interval="P_sat",
                                   overdispersion=od)
            hi = r["P_sat_hi"] if np.isfinite(r["P_sat_hi"]) else np.inf
            hits[od] += r["P_sat_lo"] <= TRUTH["P_sat"] <= hi
    assert hits[True] >= 34     # measured 38
    assert hits[False] <= 25    # measured 19


def test_overdispersion_refusal():
    n = np.random.default_rng(0).poisson(saturation_curve(POWER, **TRUTH))
    with pytest.raises(ValueError, match="overdispersion"):
        fit_saturation(POWER, n, 1.0, overdispersion="yes")


# ----------------------------------------------------------------------
# check_stationarity with p-values by simulation
# ----------------------------------------------------------------------

def test_monte_carlo_rate_p_value_equals_exact_multinomial_tail():
    """Snapshot totals (5, 1, 2) in times (1, 1, 2): the exact p-value,
    summed over every way 8 counts can fall in 3 snapshots with
    multinomial probabilities, against 40 000 simulated series."""
    n, t = np.array([5.0, 1.0, 2.0]), np.array([1.0, 1.0, 2.0])
    N = int(n.sum())
    e = N * t / t.sum()
    stat = float(np.sum((n - e) ** 2 / e))
    tail = 0.0
    for a in range(N + 1):
        for b in range(N + 1 - a):
            x = np.array([a, b, N - a - b])
            if np.sum((x - e) ** 2 / e) >= stat - 1e-9:
                tail += multinomial.pmf(x, N, t / t.sum())
    s, p = _rate_test(n, t, 5.0, (40000, np.random.default_rng(1)))
    se = np.sqrt(tail * (1 - tail) / 40000)
    assert abs(s - stat) < 1e-12
    assert abs(p - tail) < 4 * se          # measured 0.19 standard errors
    with pytest.raises(ValueError, match="monte_carlo"):
        _rate_test(n, t, 5.0)                 # the chi-square law refuses


def test_monte_carlo_table_p_values_equal_exact_enumeration():
    """The window table (2 x J) and the shape table (J x K): p-values
    from 40 000 simulated tables with the same sums agree with exact
    sums over every table with those sums, weighted by the multiple
    hypergeometric law (measured: 0.69 and 0.25 standard errors)."""
    def law(tab, R, C, N):
        num = prod(factorial(int(v)) for v in R) * \
            prod(factorial(int(v)) for v in C)
        return num / (factorial(int(N)) *
                      prod(factorial(int(v)) for v in tab.ravel()))

    for tab in (np.array([[1, 9], [4, 6], [0, 7]]),
                np.array([[2, 0, 3], [1, 4, 0], [0, 1, 2]])):
        R, C, N = tab.sum(1), tab.sum(0), tab.sum()
        E = np.outer(R, C) / N
        stat = float(np.sum((tab - E) ** 2 / E))
        tail = total = 0.0
        rows = [np.array(x) for x in itertools.product(
            *[range(c + 1) for c in C])]
        for r0 in rows:
            if r0.sum() != R[0]:
                continue
            rem = C - r0
            for r1 in itertools.product(*[range(c + 1) for c in rem]):
                r1 = np.array(r1)
                if r1.sum() != R[1]:
                    continue
                r2 = rem - r1
                if r2.sum() != R[2]:
                    continue
                full = np.array([r0, r1, r2])
                pr = law(full, R, C, N)
                total += pr
                if np.sum((full - E) ** 2 / E) >= stat - 1e-9:
                    tail += pr
        assert abs(total - 1.0) < 1e-12
        _, _, p = _homogeneity(tab.astype(float), 5.0, "test",
                               (40000, np.random.default_rng(2)))
        se = np.sqrt(tail * (1 - tail) / 40000)
        assert abs(p - tail) < 4 * se        # measured 0.69 and 0.25 se
    # a 2 x 3 table checked with the hypergeometric law in closed form
    tab = np.array([[1, 9], [4, 6], [0, 7]])
    R, C = tab.sum(1), tab.sum(0)
    p_sum = sum(prod(comb(int(r), int(x)) for r, x in zip(R, xs))
                for xs in itertools.product(*[range(int(r) + 1) for r in R])
                if sum(xs) == C[0]) / comb(int(R.sum()), int(C[0]))
    assert abs(p_sum - 1.0) < 1e-12


def test_monte_carlo_stationarity_at_small_counts():
    """Five snapshots of 0.3 s (about 0.2 counts expected in the
    zero-delay bin of each): the chi-square method refuses every series;
    by simulation, over 200 steady series, each test calls at most 5 %
    plus three standard errors inconsistent at 0.05 (measured: rate
    0.07, window 0.005, shape 0.05; the window test is discrete and
    cautious). A pair that takes part from the fourth of six 2 s
    snapshots on: the chi-square method refuses 39 of 40 series, the
    simulated window test flags 20 of them at p < 0.01 (measured)."""
    cfg = HBTConfig()
    mu1 = expected_histogram(EmitterSite(dict(SITE), 1), 1.0, cfg)
    mu2 = expected_histogram(EmitterSite(dict(SITE), 2), 1.0, cfg)
    rng = np.random.default_rng(9)
    t = np.full(5, 0.3)
    ps, refused = [], 0
    for i in range(200):
        S = rng.poisson(np.outer(t, mu1))
        try:
            check_stationarity(S, t, cfg)
        except ValueError:
            refused += 1
        r = check_stationarity(S, t, cfg, method="monte_carlo", n_sim=200,
                               seed=i)
        assert r["method"] == "monte_carlo"
        ps.append([r["rate_p"], r["window_p"], r["shape_p"]])
    ps = np.array(ps)
    assert refused == 200
    assert np.all((ps < 0.05).mean(axis=0) <= 0.05 + 3 * np.sqrt(
        0.05 * 0.95 / 200))
    assert ps.min() >= 1.0 / 201
    rng = np.random.default_rng(5)
    t = np.full(6, 2.0)
    flagged = refused = 0
    for i in range(40):
        S = np.vstack([rng.poisson(np.outer(t[:3], mu1)),
                       rng.poisson(np.outer(t[3:], mu2))])
        try:
            check_stationarity(S, t, cfg)
        except ValueError:
            refused += 1
        r = check_stationarity(S, t, cfg, method="monte_carlo", n_sim=500,
                               seed=i)
        flagged += r["window_p"] < 0.01
    assert refused >= 38        # measured 39
    assert flagged >= 10        # measured 20


def test_monte_carlo_agrees_with_chi_square_at_many_counts():
    """README example 19's steady series (20 s snapshots): the p-values
    by simulation (4000 series) agree with those of the chi-square law
    within 0.02 (measured 0.0036 at most)."""
    cfg = HBTConfig()
    mu1 = expected_histogram(EmitterSite(dict(SITE), 1), 1.0, cfg)
    t = np.full(8, 20.0)
    S = np.random.default_rng(4).poisson(np.outer(t, mu1))
    a = check_stationarity(S, t, cfg)
    b = check_stationarity(S, t, cfg, method="monte_carlo", n_sim=4000)
    for k in ("rate_p", "window_p", "shape_p"):
        assert abs(a[k] - b[k]) < 0.02                 # measured 0.0036
    for k in ("rate_chi2", "window_chi2", "shape_chi2", "shape_bins"):
        assert a[k] == b[k]
    assert a["method"] == "chi2"


def test_monte_carlo_refusals():
    cfg = HBTConfig()
    S = np.full((3, cfg.n_bins), 4.0)
    with pytest.raises(ValueError, match="method"):
        check_stationarity(S, cfg=cfg, method="exact")
    with pytest.raises(ValueError, match="n_sim"):
        check_stationarity(S, cfg=cfg, method="monte_carlo", n_sim=0)
    with pytest.raises(ValueError, match="whole"):
        check_stationarity(S + 0.5, cfg=cfg, method="monte_carlo")


# ----------------------------------------------------------------------
# inputs that gave silent nonsense up to 0.15.0
# ----------------------------------------------------------------------

def test_window_sprt_refuses_an_infinite_reject_value():
    """0.15.0 took g_reject = inf: the evidence became NaN and the test
    never decided."""
    with pytest.raises(ValueError, match="finite"):
        WindowSPRT(0.25, np.inf)
    with pytest.raises(ValueError, match="finite"):
        WindowSPRT(np.nan, 0.5)


def test_bayesian_prior_and_threshold_must_be_finite():
    """0.15.0: a NaN prior gave NaN for every result, and a NaN threshold
    a verdict probability of 0 (and a planner refusal that blamed the
    confidence)."""
    cfg = HBTConfig()
    site = EmitterSite(dict(SITE), 1)
    h = np.random.default_rng(0).poisson(expected_histogram(site, 30.0, cfg))
    with pytest.raises(ValueError, match="prior"):
        bayesian_g2(h, cfg, prior=(np.nan, 0.0))
    with pytest.raises(ValueError, match="prior"):
        bayesian_g2(h, cfg, prior=(0.5, np.inf))
    with pytest.raises(ValueError, match="threshold"):
        bayesian_g2(h, cfg).prob_below(np.nan)
    with pytest.raises(ValueError, match="prior"):
        certification_probability(site, 1.0, prior=(np.nan, 0.0))
    plan = site_from_numbers(12.0, 200.0, 0.3, 120.0, 0.97)
    for bad in (np.nan, 0.0, -0.5, np.inf):
        with pytest.raises(ValueError, match="threshold"):
            expected_posterior(plan, 1.0, threshold=bad)
        with pytest.raises(ValueError, match="threshold"):
            certification_probability(plan, 1.0, threshold=bad)
        with pytest.raises(ValueError, match="threshold"):
            required_acquisition_time(plan, threshold=bad)
    with pytest.raises(ValueError, match="n_emitters"):
        site_from_numbers(12.0, 200.0, 0.3, 120.0, 0.97, n_emitters=True)


def test_analysis_thresholds_and_profile_grid_are_checked():
    """0.15.0: a NaN threshold called every site not single, and
    g2_max = 0 gave a grid of zeros and an interval of NaN."""
    cfg = HBTConfig()
    h = np.random.default_rng(0).poisson(
        expected_histogram(EmitterSite(dict(SITE), 1), 30.0, cfg))
    with pytest.raises(ValueError, match="threshold"):
        sparq.analyze_histogram(cfg.bin_centers, h, 30.0, n_bootstrap=0,
                                threshold=np.nan, center=0.0)
    with pytest.raises(ValueError, match="threshold"):
        sparq.analyze_pulsed(cfg.bin_centers, h, threshold=np.nan,
                             method="area", n_bootstrap=0)
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="g2_max"):
            profile_likelihood_ci(h, 30.0, 150e3, cfg, g2_max=bad)


def test_grid_and_simulator_inputs_are_checked():
    """0.15.0: HBTConfig took a negative window (bins of negative width)
    or jitter (a model without jitter); deadtime_throughput raised the
    rate for a negative dead time; expected_histogram gave negative
    means for a negative time, and simulate_photon_stream an unrelated
    error."""
    for kw in (dict(tau_max=-60.5), dict(tau_max=np.nan), dict(n_bins=0),
               dict(n_bins=120.5), dict(n_bins=True),
               dict(sigma_irf=-0.3), dict(sigma_irf=np.inf)):
        with pytest.raises(ValueError):
            HBTConfig(**kw)
    assert HBTConfig(n_bins=121.0).n_bins == 121
    assert isinstance(HBTConfig(n_bins=np.int64(61)).bin_width, float)
    site = EmitterSite(dict(SITE), 1)
    assert deadtime_throughput(site, 1e5, 0.0) == 1e5
    for td, r in ((-45.0, 1e5), (np.nan, 1e5), (45.0, -1.0)):
        with pytest.raises(ValueError):
            deadtime_throughput(site, r, td)
    with pytest.raises(ValueError, match="T_s"):
        expected_histogram(site, -1.0, HBTConfig())
    assert np.all(expected_histogram(site, 0.0, HBTConfig()) == 0.0)
    with pytest.raises(ValueError, match="T_s"):
        simulate_photon_stream(site, -1.0, np.random.default_rng(0))


def test_g2_three_level_takes_an_array_of_shelving_times():
    """0.15.0 refused an array of tau2 (the built-in max); each entry now
    equals the scalar result exactly."""
    t2 = np.array([50.0, 250.0, 800.0])
    tau = 12.0
    out = g2_three_level(tau, 15.0, t2, 0.3)
    for v, T in zip(out, t2):
        assert v == g2_three_level(tau, 15.0, float(T), 0.3)


def test_normalization_and_pulsed_inputs_are_checked():
    """0.15.0: an infinite click count gave g2 = 0 everywhere; the pulsed
    simulator gave negative means for g2_0 = -1 (a bin of -64) or
    a negative rate; fit_pulsed failed with an unrelated error for a NaN
    center and fitted the wrong peak for a center outside the record
    (g2(0) 0.29 for a true 0.12)."""
    cfg = HBTConfig()
    with pytest.raises(ValueError, match="finite"):
        normalize_g2(np.ones(cfg.n_bins), np.inf, 1e5, 1.0, cfg)
    pc = HBTConfig(tau_max=60.5, n_bins=484)
    good = dict(T_s=10.0, rate_cps=2e5, g2_0=0.12, tau_e=2.0)
    for k, v in (("g2_0", -1.0), ("rate_cps", -2e5), ("T_s", 0.0),
                 ("bg_frac", -0.1), ("a", -2.0), ("tau2", 0.0),
                 ("g2_0", np.nan)):
        with pytest.raises(ValueError, match=k):
            expected_hist_pulsed(pc, **{**good, k: v})
    mean = expected_hist_pulsed(pc, **good, sigma_pair=0.5, center_off=1.7)
    counts = np.random.default_rng(0).poisson(mean)
    for c in (np.nan, 500.0, -61.0):
        with pytest.raises(ValueError, match="center"):
            fit_pulsed(pc.bin_centers, counts, center=c, interval=False)
    r = fit_pulsed(pc.bin_centers, counts, center=1.7, interval=False)
    assert abs(r["g2_0"] - 0.12) < 0.05          # measured 0.0046 (0.1154)


def test_model_fit_options_are_checked():
    """0.15.0: goodness_of_fit with n_boot = 0 returned the p-value 1.0,
    and fit_model took a negative number of shoulders as 0."""
    cfg = HBTConfig()
    h = np.random.default_rng(0).poisson(
        expected_histogram(EmitterSite(dict(SITE), 1), 30.0, cfg))
    f = fit_model(h, 30.0, 150e3, cfg, c0_prior=(1.0, 0.01))
    for bad in (0, -5, 2.5, True):
        with pytest.raises(ValueError, match="n_boot"):
            goodness_of_fit(f, n_boot=bad)
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="n_shoulders"):
            fit_model(h, 30.0, 150e3, cfg, "multi_exponential",
                      n_shoulders=bad)
