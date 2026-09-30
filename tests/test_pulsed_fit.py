"""Exact comb fit of pulsed histograms (new in 0.11.0).

References used here, all independent of the code under test:
* the peak's distribution function against a numerical integral of
  `peak_shape` (the erfcx jitter blur of `sparq.physics`) and, without
  jitter, against the Laplace law 1 - exp(-x/b)/2;
* noise-free data made by summing `expected_hist_pulsed` on a grid 16
  times finer than the bins (a midpoint-rule bin integral that does
  not use `peak_cdf`), whose parameters the fit must return;
* the profile interval's ends checked with a separate Nelder-Mead
  minimization of the Poisson likelihood over all other parameters;
* the deviance moments against simulated Poisson counts.
"""
import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import chi2

from sparq import analyze_pulsed, fit_pulsed, peak_cdf
from sparq.physics import HBTConfig
from sparq.pulsed import (_CombBasis, _deviance_moments, _edges_from_centers,
                          calibrate_comb, expected_hist_pulsed, g2_peak_area,
                          peak_shape)

CFG = HBTConfig(tau_max=60.5, n_bins=484, sigma_irf=0.35)   # 0.25 ns bins


def _binned(cfg, T_s, g2_0, tau_e, a=0.0, tau2=200.0, bg=0.0, sp=0.5,
            off=1.7, t_rep=12.5, rate=2e5):
    """Bin integrals by the midpoint rule on 16 sub-bins per bin."""
    fine = HBTConfig(tau_max=cfg.tau_max, n_bins=cfg.n_bins * 16)
    mu = expected_hist_pulsed(fine, T_s, rate, g2_0, tau_e, a=a, tau2=tau2,
                              bg_frac=bg, sigma_pair=sp, center_off=off,
                              t_rep=t_rep)
    return mu.reshape(-1, 16).sum(1)


def test_peak_cdf_matches_the_integral_of_the_peak():
    t = np.linspace(-80.0, 80.0, 800001)
    for b, s in ((1.0, 0.5), (0.05, 0.5), (3.0, 0.1), (0.4, 1.2)):
        f = peak_shape(t, b, s)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1])
                                               * np.diff(t))])
        for x in (-7.0, -2.0, -0.3, 0.0, 0.7, 4.0, 9.0):
            assert abs(peak_cdf(x, b, s) - np.interp(x, t, cum)) < 1e-7
    # no jitter: the Laplace law
    x = np.array([-3.0, -0.2, 0.0, 0.5, 6.0])
    lap = np.where(x < 0, 0.5 * np.exp(x / 1.3), 1 - 0.5 * np.exp(-x / 1.3))
    assert np.max(np.abs(peak_cdf(x, 1.3, 0.0) - lap)) < 1e-15
    # far tails keep their relative precision (no 1 - 1 cancellation)
    far = peak_cdf(-40.0, 1.0, 0.5)
    assert far > 0 and abs(far / (0.5 * np.exp(0.125 - 40.0)) - 1) < 1e-9


def test_comb_basis_equals_the_fine_grid_sum():
    """The exact bin integrals equal a fine midpoint-rule sum of the
    simulator's own peaks, to 2e-5 of the largest bin (the midpoint
    rule's own error on 1/64 ns steps is about 1e-5 here)."""
    mu = _binned(CFG, 10.0, 0.12, 1.0, a=0.4, tau2=40.0, bg=0.2)
    X = _CombBasis(_edges_from_centers(CFG.bin_centers), 12.5).columns(
        1.7, 1.0, 0.5, 40.0)
    A = (1e5) ** 2 * 12.5e-9 * 10.0
    model = X @ np.array([0.12 * A, A, 0.4 * A, 0.2 * A])
    assert np.max(np.abs(model - mu)) < 2e-5 * mu.max()


@pytest.mark.parametrize("case", [
    dict(g2_0=0.12, tau_e=1.0),
    dict(g2_0=0.12, tau_e=2.0),
    dict(g2_0=0.05, tau_e=1.0, bg=0.3),
])
def test_fit_recovers_noise_free_combs(case):
    mu = _binned(CFG, 10.0, **case)
    r = fit_pulsed(CFG.bin_centers, mu, interval=False)
    assert r["ok"]
    assert abs(r["g2_0"] - case["g2_0"]) < 1e-4
    assert abs(r["params"]["tau_e"] - case["tau_e"]) < 1e-3
    assert abs(r["params"]["center"] - 1.7) < 1e-3
    assert abs(r["params"]["bg_frac"] - case.get("bg", 0.0)) < 1e-4
    assert np.max(np.abs(r["mu"] - mu)) < 1e-4 * mu.max()


def test_the_area_method_is_biased_where_the_fit_is_not():
    """Noise-free, 80 MHz, true g2(0) = 0.12: the 0.10.0 default (peak
    areas minus the floor between peaks) gives 0.104 for a 1 ns lifetime
    and 0 for 2 ns; the new default gives 0.120."""
    for tau_e, area_max in ((1.0, 0.11), (2.0, 0.01)):
        mu = _binned(CFG, 10.0, 0.12, tau_e)
        old = analyze_pulsed(CFG.bin_centers, mu, method="area",
                             n_bootstrap=0)
        new = analyze_pulsed(CFG.bin_centers, mu, n_bootstrap=0)
        assert old["g2_0"] < area_max
        assert abs(new["g2_0"] - 0.12) < 1e-4


def test_fit_works_where_the_area_method_cannot():
    """160 MHz (6.25 ns period): no delay is 4.5 ns from every peak, so
    the area method is refused; the fit recovers g2(0)."""
    cfg = HBTConfig(tau_max=40.0, n_bins=320)
    mu = _binned(cfg, 5.0, 0.1, 0.8, t_rep=6.25, off=3.0)
    with pytest.raises(ValueError, match="4.5 ns"):
        analyze_pulsed(cfg.bin_centers, mu, t_rep=6.25, method="area",
                       n_bootstrap=0)
    r = fit_pulsed(cfg.bin_centers, mu, t_rep=6.25, interval=False)
    assert abs(r["g2_0"] - 0.1) < 1e-4


def test_envelope_needs_the_singles_rates_to_close_the_interval():
    """With a blinking envelope the far level is not reached within
    +-44 ns; the fit still finds the truth on noise-free data, but its
    interval stays open above unless the singles rates pin the level."""
    cfg = HBTConfig(tau_max=43.75, n_bins=175)               # 0.5 ns bins
    mu = _binned(cfg, 10.0, 0.12, 1.0, a=0.5, tau2=40.0)
    free = fit_pulsed(cfg.bin_centers, mu, envelope=True, sigma_pair=0.5)
    assert abs(free["g2_0"] - 0.12) < 1e-3
    assert abs(free["params"]["a"] - 0.5) < 1e-2
    assert free["lo"] < 0.12 and np.isnan(free["hi"])
    pinned = fit_pulsed(cfg.bin_centers, mu, envelope=True, sigma_pair=0.5,
                        T_s=10.0, singles_cps=2e5)
    assert pinned["lo"] < 0.12 < pinned["hi"] < 0.2


def test_profile_interval_ends_by_an_independent_minimization():
    rng = np.random.default_rng(123)
    cfg = HBTConfig(tau_max=50.0, n_bins=200)
    mu = _binned(cfg, 10.0, 0.12, 1.0)
    h = rng.poisson(mu).astype(float)
    r = fit_pulsed(cfg.bin_centers, h, level=0.95)
    p = r["params"]
    basis = _CombBasis(_edges_from_centers(cfg.bin_centers), 12.5)
    pos = h > 0

    def nll(x, g):
        c, lte, ls, lA, lbg = x
        m = basis.columns(c, np.exp(lte), np.exp(ls)) @ np.array(
            [g * np.exp(lA), np.exp(lA), np.exp(lbg)])
        return np.sum(m - h) - np.sum(h[pos] * np.log(m[pos] / h[pos]))

    f_min = r["deviance"] / 2.0
    for g in (r["lo"], r["hi"]):
        best = np.inf
        for bg0 in (1e-6, 1e-2):
            x0 = [p["center"], np.log(p["tau_e"]), np.log(p["sigma_pair"]),
                  np.log(p["area"]), np.log(bg0 * p["area"])]
            res = minimize(nll, x0, args=(g,), method="Nelder-Mead",
                           options=dict(xatol=1e-7, fatol=1e-9,
                                        maxiter=20000, maxfev=20000))
            best = min(best, res.fun)
        assert abs(2.0 * (best - f_min) - chi2.ppf(0.95, 1)) < 0.03
    assert r["lo"] < r["g2_0"] < r["hi"]


def test_deviance_moments_match_simulation():
    mu = np.array([0.01, 0.3, 1.0, 2.5, 8.0, 40.0, 400.0])
    m, v = _deviance_moments(mu)
    rng = np.random.default_rng(3)
    k = rng.poisson(mu, size=(200000, mu.size)).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 2 * (mu - k + np.where(k > 0, k * np.log(k / mu), 0.0))
    se_m = d.std(0) / np.sqrt(k.shape[0])
    assert np.all(np.abs(d.mean(0) - m) < 5 * se_m + 1e-12)
    assert np.all(np.abs(d.var(0) / v - 1) < 0.05)
    # many counts: the chi-square(1) moments
    assert abs(m[-1] - 1) < 0.01 and abs(v[-1] - 2) < 0.05


def test_short_record_calibration_and_refusals():
    # 3.5 periods: up to 0.10.0 every peak was excluded and the first
    # one returned as the center
    cfg = HBTConfig(tau_max=21.875, n_bins=175)
    mu = expected_hist_pulsed(cfg, 10.0, 2e5, 0.1, 0.8, center_off=1.0)
    center, _, _ = calibrate_comb(cfg.bin_centers, mu)
    assert abs(center - 1.0) < 0.2
    with pytest.raises(ValueError, match="three periods"):
        fit_pulsed(cfg.bin_centers[:80], mu[:80])
    with pytest.raises(ValueError, match="both"):
        fit_pulsed(cfg.bin_centers, mu, T_s=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        fit_pulsed(cfg.bin_centers, -mu)
    with pytest.raises(ValueError, match="method"):
        analyze_pulsed(cfg.bin_centers, mu, method="peaks")


def test_area_method_is_unchanged():
    """method='area' is the 0.10.0 analysis: the same calls, same
    numbers."""
    rng = np.random.default_rng(4)
    mu = expected_hist_pulsed(CFG, 20.0, 2e5, 0.12, 1.0, center_off=1.7)
    counts = rng.poisson(mu).astype(float)
    res = analyze_pulsed(CFG.bin_centers, counts, method="area",
                         n_bootstrap=20, seed=5)
    c, _, _ = calibrate_comb(CFG.bin_centers, counts)
    assert res["g2_0"] == g2_peak_area(CFG.bin_centers, counts, c)
    rng = np.random.default_rng(5)
    draws = [g2_peak_area(CFG.bin_centers, rng.poisson(counts).astype(float),
                          c) for _ in range(20)]
    assert res["g2_0_std"] == float(np.std(draws, ddof=1))


def _moments_by_direct_sum(mu):
    """Independent reference: sum over k = 0 .. mu + 40 sqrt(mu) + 100."""
    from scipy.stats import poisson
    k = np.arange(0, int(mu + 40 * np.sqrt(mu) + 100))
    p = poisson.pmf(k, mu)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 2 * (mu - k + np.where(k > 0, k * np.log(np.maximum(k, 1) / mu),
                                   0.0))
    e = float(np.sum(p * d))
    return e, float(np.sum(p * d * d)) - e * e


def test_deviance_moments_exact_sums_and_large_mean_expansion():
    """Blocked exact sums (m <= 1000) and the expansion (m > 1000)
    against a direct sum per bin, to 1e-8."""
    mus = np.array([1e-3, 0.05, 0.7, 3.0, 17.0, 150.0, 999.0, 1000.5,
                    2000.0, 1e4, 1e5])
    m, v = _deviance_moments(mus)
    for mu, a, b in zip(mus, m, v):
        e, vv = _moments_by_direct_sum(mu)
        assert abs(a - e) < 1e-8 and abs(b - vv) < 1e-8


def test_deviance_moments_of_a_large_bright_histogram_use_little_memory():
    """65 536 bins with peaks of 1e5 counts (a raw PicoQuant-sized
    histogram): 0.11.0 as first written needed over 1 GB here."""
    import tracemalloc
    t = np.linspace(-8192.0, 8192.0, 65536)
    mu = 5.0 + 1e5 * np.exp(-0.5 * ((t % 12.5) - 6.25) ** 2 / 0.25)
    tracemalloc.start()
    m, v = _deviance_moments(mu)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 100e6
    assert np.all(np.isfinite(m)) and np.all(v > 0)
    for i in (0, int(np.argmax(mu)), int(np.argmin(np.abs(mu - 500.0)))):
        e, vv = _moments_by_direct_sum(mu[i])
        assert abs(m[i] - e) < 1e-8 and abs(v[i] - vv) < 1e-8
