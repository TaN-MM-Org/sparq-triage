"""Tests of the 0.15.0 changes: lifetime fits with a measured instrument
response (`fit_lifetime(irf=...)`, `expected_decay(irf=...)`), a
profile-likelihood interval for g2(0) from `fit_model(level=...)`, and
the fixes (the comb calibration at high laser rates, a decreasing delay
axis in the peak-area analysis, the bootstrap level, the rate ceiling of
the photon-by-photon simulator, and the stopping point of `fit_model`).
"""
import warnings

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import chi2, norm

from sparq import (EmitterSite, HBTConfig, analyze_histogram, analyze_pulsed,
                   calibrate_comb, expected_decay, expected_hist_pulsed,
                   expected_histogram, fit_lifetime, fit_model, fit_pulsed,
                   g2_model, g2_peak_area, profile_likelihood_ci,
                   simulate_photon_stream)

T_REP = 12.5
CRIT = float(chi2.ppf(0.95, 1))


# ----------------------------------------------------------------------
# a measured instrument response
# ----------------------------------------------------------------------

def _tailed_response(rng, n):
    """Delays of n photons of an illustrative instrument response with a
    slow tail: a Gaussian core (60 ps) at 1 ns, and a quarter of the
    photons delayed further by an exponential of mean 0.5 ns."""
    t = 1.0 + rng.normal(0.0, 0.06, n)
    tail = rng.random(n) < 0.25
    t[tail] += rng.exponential(0.5, int(tail.sum()))
    return t


def _photon_decay(rng, edges, n_photons, tau, n_dark, n_irf=1_000_000):
    """(response histogram, decay histogram), both made photon by photon:
    the response from its own photons, the decay from photons delayed by
    the response plus an exponential, folded onto the laser period, plus
    dark counts spread evenly over the period."""
    irf = np.histogram(_tailed_response(rng, n_irf) % T_REP, bins=edges)[0]
    t = _tailed_response(rng, n_photons) + rng.exponential(tau, n_photons)
    dt = np.r_[t % T_REP, rng.uniform(0.0, T_REP, n_dark)]
    return irf, np.histogram(dt, bins=edges)[0]


def test_measured_response_model_matches_quadrature():
    """Each bin's share of the decay equals the convolution integral of the
    response (its counts spread evenly over each bin) with the exponential,
    computed bin by bin with SciPy's quad, to 1e-12, for lifetimes from a
    fiftieth of a bin to many bins and shifts of either sign."""
    rng = np.random.default_rng(3)
    n, w = 24, 0.1
    d = (np.arange(n) + 0.5) * w
    q = rng.random(n) ** 3
    q /= q.sum()
    for tau, shift in ((0.5, 0.0), (0.5, 0.037), (0.02, -0.013),
                       (3.0, 0.21), (0.002, 0.05)):
        mu = expected_decay(d, 1.0, tau, shift, irf=q)
        ref = np.zeros(n)
        for i in range(n):
            c, e = i * w, (i + 1) * w
            for j in range(n):
                a, b = j * w + shift, (j + 1) * w + shift
                if a >= e:
                    continue

                def inner(u):
                    return (np.exp(-max(c - u, 0.0) / tau)
                            - np.exp(-max(e - u, 0.0) / tau))

                pts = [p for p in (c, e) if a < p < b]
                ref[i] += q[j] / w * quad(inner, a, b, points=pts or None,
                                          epsabs=1e-15, epsrel=1e-13)[0]
        assert np.max(np.abs(mu - ref)) < 1e-12, (tau, shift)


def test_measured_response_periodic_sum_equals_direct_sum():
    """With a laser period the closed geometric sum over earlier pulses
    equals the direct sum over the pulses (back to where less than 1e-18
    of a decay is left) to 1e-12 of the largest bin, and the bins of one
    period hold one pulse's photons (to 1e-12)."""
    n = 250
    d = (np.arange(n) + 0.5) * T_REP / n
    q = norm.pdf(d, 2.0, 0.08) + 0.2 * (d > 2.0) * np.exp(-(d - 2.0) / 0.4)
    for tau, shift in ((0.5, 0.1), (4.0, -0.3), (20.0, 0.05), (0.05, 5.0)):
        per = expected_decay(d, 1.0, tau, shift, t_rep=T_REP, irf=q)
        M = int(np.ceil(42 * tau / T_REP)) + 3
        direct = sum(expected_decay(d, 1.0, tau, shift - m * T_REP, irf=q)
                     for m in range(-3, M + 1))
        assert np.max(np.abs(per - direct)) < 1e-12 * per.max(), tau
        assert abs(per.sum() - 1.0) < 1e-12, tau


def test_measured_gaussian_response_converges_to_the_gaussian_model():
    """A response histogram taken from a Gaussian (its exact bin integrals)
    gives the decay of the Gaussian model of 0.14.0 (an independent closed
    form) up to the error of spreading each bin's counts evenly, which is
    of second order: it falls by a factor of 4 (3.5 to 4.5) each time the
    bins are halved, to below 1e-4 of the peak at 10 ps bins."""
    for tau, sig in ((1.0, 0.2), (0.3, 0.1)):
        errs = []
        for w in (0.04, 0.02, 0.01):
            n = int(round(T_REP / w))
            d = (np.arange(n) + 0.5) * w
            q = np.diff(norm.cdf(np.arange(n + 1) * w, 1.0, sig))
            a = expected_decay(d, 1.0, tau, 0.0, t_rep=T_REP, irf=q)
            g = expected_decay(d, 1.0, tau, 1.0, sigma_irf=sig, t_rep=T_REP)
            errs.append(np.max(np.abs(a - g)) / g.max())
        for k in range(2):
            assert 3.5 < errs[k] / errs[k + 1] < 4.5, (tau, errs)
        assert errs[-1] < 1e-4 * (1.0 if sig > 0.15 else 4.0), (tau, errs)


def test_reconvolution_fit_recovers_noise_free_decays():
    """On noise-free histograms made with the measured-response model the
    fit returns the lifetimes, areas, shift and background to 1e-6
    (relative), with one and with two lifetimes, and with the shift
    given."""
    rng = np.random.default_rng(1)
    n = 400
    edges = np.linspace(0.0, T_REP, n + 1)
    d = 0.5 * (edges[1:] + edges[:-1])
    irf = np.histogram(_tailed_response(rng, 2_000_000) % T_REP,
                       bins=edges)[0].astype(float)
    for areas, taus, shift, bg in (([5e4], [2.3], 0.031, 40.0),
                                   ([3e4, 2e4], [0.6, 4.0], -0.02, 25.0)):
        mu = expected_decay(d, areas, taus, shift, t_rep=T_REP,
                            background=bg, irf=irf)
        for t0 in (None, shift):
            r = fit_lifetime(d, mu, t_rep=T_REP, n_components=len(taus),
                             irf=irf, t0=t0, interval=False)
            assert np.allclose(r["tau"], taus, rtol=1e-6), r["tau"]
            assert np.allclose(r["area"], areas, rtol=1e-6), r["area"]
            assert abs(r["t0"] - shift) < 1e-6
            assert abs(r["background"] - bg) < 1e-6 * bg
            assert np.isnan(r["sigma_irf"])


def test_reconvolution_fit_on_photons_with_a_tailed_response():
    """Photons whose response has a slow tail (made photon by photon, the
    response histogram from its own 1e6 photons): the Gaussian fit of
    0.14.0 is rejected (p < 1e-6) and its interval misses the true 1.5 ns;
    the fit with the measured response holds it, with p > 0.01. An
    independent Nelder-Mead minimization of the Poisson deviance finds no
    better fit (within 1e-3), and at both ends of the 95 % interval the
    profile deviance, minimized independently, is 3.84 within 0.01."""
    rng = np.random.default_rng(7)
    n, tau = 500, 1.5
    edges = np.linspace(0.0, T_REP, n + 1)
    d = 0.5 * (edges[1:] + edges[:-1])
    irf, h = _photon_decay(rng, edges, 100_000, tau, 2000, n_irf=2_000_000)
    g = fit_lifetime(d, h, t_rep=T_REP)
    assert g["p_value"] < 1e-6
    assert not (g["tau_lo"][0] <= tau <= g["tau_hi"][0])
    r = fit_lifetime(d, h, t_rep=T_REP, irf=irf)
    assert r["tau_lo"][0] <= tau <= r["tau_hi"][0]
    assert r["p_value"] > 0.01 and r["resolved"]
    assert abs(r["background"] - 2000 / T_REP) < 0.1 * 2000 / T_REP

    pos = h > 0

    def deviance(shift, tau_, la, lb):
        mu = expected_decay(d, np.exp(la), tau_, shift, t_rep=T_REP,
                            background=np.exp(lb), irf=irf)
        return 2.0 * float(np.sum(mu - h) - np.sum(h[pos]
                                                  * np.log(mu[pos] / h[pos])))

    la0, lb0 = np.log(r["area"][0]), np.log(r["background"])
    nm = minimize(lambda x: deviance(x[0], np.exp(x[1]), x[2], x[3]),
                  [r["t0"] + 0.01, np.log(tau * 1.05), la0, lb0],
                  method="Nelder-Mead",
                  options=dict(xatol=1e-9, fatol=1e-10, maxiter=20000))
    assert nm.fun >= r["deviance"] - 1e-3
    for end in (r["tau_lo"][0], r["tau_hi"][0]):
        p = minimize(lambda x: deviance(x[0], end, x[1], x[2]),
                     [r["t0"], la0, lb0], method="Nelder-Mead",
                     options=dict(xatol=1e-9, fatol=1e-10, maxiter=20000))
        assert abs(p.fun - min(nm.fun, r["deviance"]) - CRIT) < 0.01, end


def test_reconvolution_interval_coverage_on_dim_decays():
    """Of 20 simulated dim decays (2 ns lifetime, 3000 photons in 125
    bins, 100 dark counts, the tailed response), at least 16 of the 95 %
    intervals hold the true lifetime. (Done once outside the test suite:
    186 of 200 for this case, and 196 of 200 for a bright one, 6 ns and
    1e5 photons in 500 bins; 4.5 % of the p-values fell below 0.05 in
    both.)"""
    edges = np.linspace(0.0, T_REP, 126)
    d = 0.5 * (edges[1:] + edges[:-1])
    hit = 0
    for k in range(20):
        rng = np.random.default_rng(1000 + k)
        irf, h = _photon_decay(rng, edges, 3000, 2.0, 100)
        r = fit_lifetime(d, h, t_rep=T_REP, irf=irf)
        hit += bool(r["tau_lo"][0] <= 2.0 <= r["tau_hi"][0])
    assert hit >= 16, hit


def test_reconvolution_refusals_and_decreasing_axis():
    d = (np.arange(100) + 0.5) * 0.125
    irf = norm.pdf(d, 1.0, 0.1)
    mu = expected_decay(d, 1e4, 2.0, 0.0, t_rep=T_REP, irf=irf)
    with pytest.raises(ValueError, match="same delays"):
        fit_lifetime(d, mu, t_rep=T_REP, irf=irf[:-1])
    with pytest.raises(ValueError, match="non-negative"):
        fit_lifetime(d, mu, t_rep=T_REP, irf=np.r_[-1.0, irf[1:]])
    with pytest.raises(ValueError, match="no counts"):
        fit_lifetime(d, mu, t_rep=T_REP, irf=np.zeros(d.size))
    with pytest.raises(ValueError, match="not both"):
        fit_lifetime(d, mu, t_rep=T_REP, irf=irf, sigma_irf=0.1)
    with pytest.raises(ValueError, match="not both"):
        expected_decay(d, 1.0, 2.0, 0.0, sigma_irf=0.1, irf=irf)
    uneven = np.r_[d[:50], d[50:] + 0.01 * np.arange(50)]
    with pytest.raises(ValueError, match="equal width"):
        expected_decay(uneven, 1.0, 2.0, 0.0, irf=irf)
    # a decreasing delay axis (response in the same order) gives the same
    # result as an increasing one
    a = fit_lifetime(d, mu, t_rep=T_REP, irf=irf, interval=False)
    b = fit_lifetime(d[::-1], mu[::-1], t_rep=T_REP, irf=irf[::-1],
                     interval=False)
    assert abs(a["tau"][0] - b["tau"][0]) < 1e-9
    assert np.allclose(b["mu"][::-1], a["mu"], rtol=1e-9)


# ----------------------------------------------------------------------
# an interval for g2(0) from fit_model
# ----------------------------------------------------------------------

_COH = dict(omega=3.0, gamma=1.4, gamma_deph=0.1, delta=0.0)


def _coherent_mean(cfg, T, rate, rho2=0.9):
    nodes, wts = cfg.bin_nodes()
    flat = (rate / 2) ** 2 * cfg.bin_width * 1e-9 * T
    return flat * (g2_model(nodes, "coherent", _COH, cfg.sigma_irf,
                            rho2=rho2) @ wts)


def test_fit_model_interval_agrees_with_profile_likelihood_ci():
    """For the three-level model with the flat level pinned, the interval
    of `fit_model(level=0.95)` equals that of `profile_likelihood_ci` (a
    separate implementation: another parameterization, search and
    refinement) within 1e-3 at both ends, and so do the best values, on
    data where no parameter of either ends on a search limit (two runs;
    the second is the one of `test_fit_model_reaches_its_best_fit`)."""
    site = EmitterSite(dict(tau1=12.0, tau2=150.0, a=0.5, rate_kcps=200.0,
                            rho=0.93, blinking=False), 1)
    cfg = HBTConfig()
    for seed in (1, 2):
        h = np.random.default_rng(seed).poisson(
            expected_histogram(site, 30.0, cfg)).astype(float)
        f = fit_model(h, 30.0, 200e3, cfg, "three_level",
                      c0_prior=(1.0, 0.01), level=0.95)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            p = profile_likelihood_ci(h, 30.0, 200e3, cfg,
                                      c0_prior=(1.0, 0.01))
        assert not any("search bound" in str(w.message) for w in caught)
        assert p["at_bound"] == []
        assert abs(f["g2_0"] - p["g2_hat"]) < 1e-3, seed
        assert abs(f["g2_0_lo"] - p["lo"]) < 1e-3, seed
        assert abs(f["g2_0_hi"] - p["hi"]) < 1e-3, seed
        assert f["profile_improved"] is False


def test_fit_model_interval_ends_by_an_independent_profile():
    """Coherently driven emitter (README example 10): at both ends of the
    95 % interval of the coherent model, an independent Nelder-Mead
    minimization over the other parameters of the Poisson deviance (built
    from `g2_model`, with the prior term) is the chi-square quantile 3.84
    above the minimum within 0.02, and that minimization finds no better
    fit than `fit_model` (within 1e-3)."""
    cfg = HBTConfig(tau_max=10.05, n_bins=201, sigma_irf=0.05)
    T, rate = 120.0, 2e5
    h = np.random.default_rng(0).poisson(_coherent_mean(cfg, T, rate))
    f = fit_model(h, T, rate, cfg, "coherent", fixed=dict(delta=0.0),
                  c0_prior=(1.0, 0.01), level=0.95)
    assert f["g2_0_lo"] < 0.1 < f["g2_0_hi"]
    nodes, wts = cfg.bin_nodes()
    flat = (rate / 2) ** 2 * cfg.bin_width * 1e-9 * T
    pos = h > 0

    def dev(rho2, y):
        c0 = np.exp(y[0])
        p = dict(omega=np.exp(y[1]), gamma=np.exp(y[2]),
                 gamma_deph=np.exp(y[3]), delta=0.0)
        mu = flat * c0 * (g2_model(nodes, "coherent", p, cfg.sigma_irf,
                                   rho2=rho2) @ wts)
        if np.any(mu <= 0):
            return 1e300
        return (2.0 * float(np.sum(mu - h)
                            - np.sum(h[pos] * np.log(mu[pos] / h[pos])))
                + ((c0 - 1.0) / 0.01) ** 2)

    y0 = [np.log(f["c0"]), np.log(f["params"]["omega"]),
          np.log(f["params"]["gamma"]), np.log(f["params"]["gamma_deph"])]
    opts = dict(xatol=1e-9, fatol=1e-10, maxiter=40000, maxfev=40000)
    best = minimize(lambda x: dev(x[0], x[1:]), [f["rho2"]] + y0,
                    method="Nelder-Mead", options=opts)
    d_fit = dev(f["rho2"], y0)
    assert best.fun >= d_fit - 1e-3
    for end in (f["g2_0_lo"], f["g2_0_hi"]):
        p = minimize(lambda y: dev(1.0 - end, y), y0, method="Nelder-Mead",
                     options=opts)
        assert abs(p.fun - min(best.fun, d_fit) - CRIT) < 0.02, end


def test_fit_model_interval_coverage_for_a_coherent_emitter():
    """Of 10 simulated runs of the coherently driven emitter (20 s: 20
    counts per 0.1 ns bin far from the dip, 2.5 in the zero-delay bin),
    at least 8 of the 95 % intervals of the coherent fit hold the true
    g2(0) = 0.1. (Done once outside the test suite: see the
    CHANGELOG.)"""
    cfg = HBTConfig(tau_max=10.05, n_bins=201, sigma_irf=0.05)
    mean = _coherent_mean(cfg, 20.0, 2e5)
    hit = 0
    for k in range(10):
        h = np.random.default_rng(5000 + k).poisson(mean)
        f = fit_model(h, 20.0, 2e5, cfg, "coherent", fixed=dict(delta=0.0),
                      c0_prior=(1.0, 0.01), level=0.95)
        hit += bool(f["g2_0_lo"] <= 0.1 <= f["g2_0_hi"])
    assert hit >= 8, hit


def test_fit_model_level_is_optional_and_checked():
    cfg = HBTConfig()
    site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                            rho=0.95, blinking=False), 1)
    h = np.random.default_rng(0).poisson(expected_histogram(site, 30.0, cfg))
    f = fit_model(h, 30.0, 150e3, cfg, c0_prior=(1.0, 0.01))
    assert "g2_0_lo" not in f and "level" not in f
    for bad in (0.0, 1.0, 95, float("nan"), "high"):
        with pytest.raises(ValueError, match="level"):
            fit_model(h, 30.0, 150e3, cfg, level=bad)


# ----------------------------------------------------------------------
# fixes
# ----------------------------------------------------------------------

def test_fit_model_reaches_its_best_fit():
    """Fixed in 0.15.0: `fit_model` minimized the negative log-likelihood
    sum(mu - h log mu), a number of the order of the total counts, on
    which L-BFGS-B's relative stopping rule stopped early. On these data
    it returned g2(0) = 0.1751 with a deviance 0.047 above its minimum;
    an independent Nelder-Mead minimization of the deviance (built from
    `g2_model`) finds the best fit at 0.1733, as `profile_likelihood_ci`
    does. Now the fit is within 2e-4 of it and its deviance within 1e-3.
    The coherent fit of README example 10 is unchanged (g2(0) 0.1097 to
    1e-4, deviance 200.48 to 0.01). (Its three-level fit, which the
    p-value rejects, is a local search among several minima, and where it
    stops still depends on the SciPy version: deviance 345.74 with SciPy
    1.18 and 348.81 with SciPy 1.10; in 0.14.0 350.16 and 345.75. It is
    not tested here.)"""
    site = EmitterSite(dict(tau1=12.0, tau2=150.0, a=0.5, rate_kcps=200.0,
                            rho=0.93, blinking=False), 1)
    cfg = HBTConfig()
    h = np.random.default_rng(2).poisson(
        expected_histogram(site, 30.0, cfg)).astype(float)
    f = fit_model(h, 30.0, 200e3, cfg, "three_level", c0_prior=(1.0, 0.01))
    nodes, wts = cfg.bin_nodes()
    flat = (1e5) ** 2 * cfg.bin_width * 1e-9 * 30.0
    pos = h > 0

    def dev(x):
        rho2, c0 = x[0], np.exp(x[1])
        p = dict(tau1=np.exp(x[2]), a=np.exp(x[3]), tau2=np.exp(x[4]))
        mu = flat * c0 * (g2_model(nodes, "three_level", p, cfg.sigma_irf,
                                   rho2) @ wts)
        if np.any(mu <= 0) or not 0 <= rho2 <= 1:
            return 1e300
        return (2.0 * float(np.sum(mu - h)
                            - np.sum(h[pos] * np.log(mu[pos] / h[pos])))
                + ((c0 - 1.0) / 0.01) ** 2)

    p = f["params"]
    x0 = np.array([f["rho2"], np.log(f["c0"]), np.log(p["tau1"]),
                   np.log(p["a"]), np.log(p["tau2"])])
    best = None
    for k in range(2):
        st = x0 + (np.random.default_rng(k).normal(0.0, 0.05, 5) if k else 0)
        r = minimize(dev, st, method="Nelder-Mead",
                     options=dict(xatol=1e-10, fatol=1e-12, maxiter=40000,
                                  maxfev=40000))
        if best is None or r.fun < best.fun:
            best = r
    assert abs(f["g2_0"] - (1.0 - best.x[0])) < 2e-4
    assert dev(x0) - best.fun < 1e-3

    cfg = HBTConfig(tau_max=10.05, n_bins=201, sigma_irf=0.05)
    h = np.random.default_rng(0).poisson(_coherent_mean(cfg, 120.0, 2e5))
    fc = fit_model(h, 120.0, 2e5, cfg, "coherent", fixed=dict(delta=0.0),
                   c0_prior=(1.0, 0.01))
    assert abs(fc["g2_0"] - 0.1097) < 1e-4
    assert abs(fc["deviance"] - 200.48) < 0.01


def _comb(t_rep, tau_e, sp, off, g2_0=0.12, T=10.0, rate=2e5, periods=15):
    n = int(round(2 * periods * 40)) | 1        # 40 bins per period
    cfg = HBTConfig(tau_max=periods * t_rep, n_bins=n)
    return cfg, expected_hist_pulsed(cfg, T, rate, g2_0=g2_0, tau_e=tau_e,
                                     sigma_pair=sp, center_off=off,
                                     t_rep=t_rep)


def test_comb_fit_at_high_laser_rates():
    """Fixed in 0.15.0: `calibrate_comb` folded the comb within +-1.5 ns
    and summed the peaks within +-3 ns for every period, which at periods
    below about 3 ns covers a whole period or more. At 400 MHz (2.5 ns)
    and 1 GHz the zero-delay peak was then placed on a side peak or
    between peaks, and `fit_pulsed` returned g2(0) = 0.000 instead of
    0.12 on noise-free data; on a simulated 400 MHz run with a true g2(0)
    of 0.6, `analyze_pulsed` said 0.000 and "single emitter", with no
    warning. The windows now shrink with the period: the noise-free fits
    return 0.12 to 1e-4 and the peak position to 1e-3 ns, and the 400 MHz
    run gives an interval that holds 0.6 and lies above 0.5."""
    for t_rep, tau_e, sp, off in ((2.5, 0.15, 0.05, 0.3),
                                  (2.5, 0.15, 0.05, 1.05),
                                  (1.0, 0.05, 0.02, 0.3),
                                  (1.0, 0.05, 0.02, -0.45),
                                  (4.0, 0.3, 0.1, 0.9)):
        cfg, mean = _comb(t_rep, tau_e, sp, off)
        f = fit_pulsed(cfg.bin_centers, mean, t_rep=t_rep, interval=False)
        assert abs(f["g2_0"] - 0.12) < 1e-4, (t_rep, off, f["g2_0"])
        assert abs(f["params"]["center"] - off) < 1e-3, (t_rep, off)
        c, _, _ = calibrate_comb(cfg.bin_centers, mean, t_rep=t_rep)
        assert abs(c - off) < 0.05 * t_rep, (t_rep, off, c)
    t_rep = 2.5
    n = 601
    cfg = HBTConfig(tau_max=15 * t_rep, n_bins=n)
    mean = expected_hist_pulsed(cfg, 10.0, 2e5, g2_0=0.6, tau_e=0.15,
                                sigma_pair=0.05, center_off=1.05,
                                t_rep=t_rep)
    counts = np.random.default_rng(0).poisson(mean)
    r = analyze_pulsed(cfg.bin_centers, counts, t_rep=t_rep, ci=0.95)
    assert r["g2_0_low"] <= 0.6 <= r["g2_0_high"]
    assert r["g2_0_low"] > 0.5 and not r["single_emitter"]
    assert r["fit_p_value"] > 0.01


def _calibrate_comb_014(delay, hist, t_rep):
    """`calibrate_comb` as it was up to 0.14.0, kept to show that the
    result is unchanged at periods of 6 ns and more."""
    nphase = 250
    phases = np.linspace(0, t_rep, nphase, endpoint=False)
    folded = np.zeros(nphase)
    for i, ph in enumerate(phases):
        m = np.abs(((delay - ph + t_rep / 2) % t_rep) - t_rep / 2) < 1.5
        folded[i] = hist[m].mean()
    phase = phases[int(np.argmax(folded))]
    kmin = int(np.ceil((delay[0] - phase) / t_rep))
    kmax = int(np.floor((delay[-1] - phase) / t_rep))
    pos = phase + np.arange(kmin, kmax + 1) * t_rep
    areas = np.array([hist[(delay >= p - 3) & (delay <= p + 3)].sum()
                      for p in pos])
    v = areas.copy().astype(float)
    if v.size > 4:
        v[:2] = v[-2:] = np.inf
    else:
        inside = (pos - 3 >= delay[0]) & (pos + 3 <= delay[-1])
        if inside.any():
            v[~inside] = np.inf
    return float(pos[int(np.argmin(v))]), float(phase), t_rep


def test_comb_calibration_unchanged_at_periods_of_6_ns_and_more():
    """The same computation as up to 0.14.0 where its windows fit in a
    period: the results are identical (80 MHz, 160 MHz and 6 ns, noise-free
    and noisy, records of 3.5 to 10 periods)."""
    rng = np.random.default_rng(4)
    for t_rep, tau_e, off, half_width in ((12.5, 1.0, 1.7, 60.5),
                                          (12.5, 2.0, -3.3, 22.0),
                                          (6.25, 0.5, 1.1, 30.0),
                                          (6.0, 0.6, -0.7, 30.0)):
        n = int(round(2 * half_width / 0.25)) | 1
        cfg = HBTConfig(tau_max=half_width, n_bins=n)
        mean = expected_hist_pulsed(cfg, 10.0, 2e5, g2_0=0.12, tau_e=tau_e,
                                    sigma_pair=0.4, center_off=off,
                                    t_rep=t_rep)
        for hist in (mean, rng.poisson(mean)):
            assert calibrate_comb(cfg.bin_centers, hist, t_rep) == \
                _calibrate_comb_014(cfg.bin_centers, hist, t_rep)


def test_peak_area_analysis_accepts_a_decreasing_delay_axis():
    """Fixed in 0.15.0: `analyze_pulsed(method="area")` failed with
    "attempt to get argmin of an empty sequence" on a decreasing delay
    axis (which `fit_pulsed` and `analyze_histogram` accept), and
    `g2_peak_area` returned NaN for it. Both now give what the increasing
    axis gives (the same computation, so exactly)."""
    cfg = HBTConfig(tau_max=60.5, n_bins=484)
    mean = expected_hist_pulsed(cfg, 10.0, 2e5, g2_0=0.12, tau_e=1.0,
                                sigma_pair=0.5, center_off=1.7)
    counts = np.random.default_rng(0).poisson(mean)
    d = cfg.bin_centers
    up = analyze_pulsed(d, counts, method="area", n_bootstrap=20, seed=3)
    down = analyze_pulsed(d[::-1], counts[::-1], method="area",
                          n_bootstrap=20, seed=3)
    assert up == down
    assert abs(up["g2_0"] - 0.12) < 0.03
    a = g2_peak_area(d, counts, up["center"])
    b = g2_peak_area(d[::-1], counts[::-1], up["center"])
    assert a == b and np.isfinite(b)
    assert calibrate_comb(d[::-1], counts[::-1]) == calibrate_comb(d, counts)


def test_bootstrap_level_is_checked():
    """Fixed in 0.15.0: `analyze_histogram` and `analyze_pulsed(method=
    "area")` did not check `ci`. With ci = -0.5 the interval's low end
    came out above its high end (0.069 and 0.057 with 10 redraws of these
    data), and ci = 0 gave an interval of zero width, both without a
    word. A ci
    outside (0, 1) is now refused when an interval is asked for; without
    the bootstrap `ci` is not used and not checked."""
    site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                            rho=0.95, blinking=False), 1)
    cfg = HBTConfig()
    h = np.random.default_rng(1).poisson(expected_histogram(site, 30.0, cfg))
    for bad in (-0.5, 0.0, 1.0, 95, float("nan")):
        with pytest.raises(ValueError, match="ci must"):
            analyze_histogram(cfg.bin_centers, h, 30.0, n_bootstrap=5, ci=bad,
                              singles_cps=150e3)
    r = analyze_histogram(cfg.bin_centers, h, 30.0, n_bootstrap=0, ci=95,
                          singles_cps=150e3)
    assert "g2_0_low" not in r
    r = analyze_histogram(cfg.bin_centers, h, 30.0, n_bootstrap=10, ci=0.68,
                          singles_cps=150e3)
    assert r["g2_0_low"] < r["g2_0_high"]
    pc = HBTConfig(tau_max=60.5, n_bins=484)
    hp = np.random.default_rng(0).poisson(expected_hist_pulsed(
        pc, 10.0, 2e5, g2_0=0.12, tau_e=1.0, sigma_pair=0.5))
    with pytest.raises(ValueError, match="ci must"):
        analyze_pulsed(pc.bin_centers, hp, method="area", ci=-0.5,
                       n_bootstrap=5)
    with pytest.raises(ValueError, match="level"):
        analyze_pulsed(pc.bin_centers, hp, ci=95)          # fit: as before


def test_photon_simulator_refuses_a_rate_it_cannot_give():
    """Fixed in 0.15.0: the photon-by-photon simulator capped its
    collection efficiency at 1 without a word, so a site whose signal
    rate is above what its emitters give out got fewer photons than
    asked for (for tau1 = 15 ns, tau2 = 250 ns, a = 0.3: 1.2e7 counts/s
    instead of 5e7), while `expected_histogram` used the full rate. It
    is now refused. Below the ceiling the counted rate still matches
    `rate_kcps` (within 5 standard errors)."""
    rng = np.random.default_rng(0)
    p = dict(tau1=15.0, tau2=250.0, a=0.3, rho=1.0, blinking=False)
    with pytest.raises(ValueError, match="give out"):
        simulate_photon_stream(EmitterSite(dict(p, rate_kcps=5e4), 1),
                               0.002, rng)
    # two emitters give out twice as much
    t_a, t_b = simulate_photon_stream(EmitterSite(dict(p, rate_kcps=2e4), 2),
                                      0.002, rng)
    n = t_a.size + t_b.size
    assert abs(n - 2e7 * 0.002) < 5 * np.sqrt(2e7 * 0.002)
    with pytest.raises(ValueError, match="give out"):
        simulate_photon_stream(EmitterSite(dict(p, rate_kcps=2e4), 1),
                               0.002, rng)
