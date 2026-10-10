"""Tests of the 0.12.0 changes: the corrected profile-likelihood
interval, the stationarity check of a series of histograms, the signal
fraction measured from counts and its propagation into the background
correction, and the sharper input checks."""
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq, minimize
from scipy.stats import beta as beta_dist
from scipy.stats import binom, chi2, chi2_contingency, chisquare, kstest

import sparq
from sparq import (EmitterSite, HBTConfig, expected_histogram, g2_measured,
                   profile_likelihood_ci)

SITE = dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0, rho=0.95,
            blinking=False)


# ----------------------------------------------------------------------
# profile_likelihood_ci: the profile is now minimized and the ends found
# ----------------------------------------------------------------------

def _independent_profile(h, T, r_hat, cfg, prior_sd):
    """The profile NLL of the three-level model written independently of
    sparq.analysis: the model from the public g2_measured (rho^2 = 1 - g
    for one emitter), bin-averaged with cfg.bin_nodes, and minimized
    over the nuisance parameters by Nelder-Mead from several starts."""
    nodes, w = cfg.bin_nodes()
    flat = (0.5 * r_hat) ** 2 * cfg.bin_width * 1e-9 * T
    box = [(np.log(0.3), np.log(80.0)), (np.log(50.0), np.log(800.0)),
           (0.0, 3.0), (0.01, 10.0)]

    def nll(g, th):
        if not all(lo <= v <= hi for v, (lo, hi) in zip(th, box)):
            return 1e30
        m = flat * th[3] * (g2_measured(nodes, np.exp(th[0]), np.exp(th[1]),
                                        th[2], 1, np.sqrt(1.0 - g),
                                        cfg.sigma_irf) @ w)
        v = float(np.sum(m - h * np.log(m)))
        if prior_sd is not None:
            v += 0.5 * ((th[3] - 1.0) / prior_sd) ** 2
        return v

    def profile(g):
        best = np.inf
        for st in ([np.log(15.0), np.log(250.0), 0.3, 1.0],
                   [np.log(15.0), np.log(700.0), 1.0, 1.0]):
            x = np.array(st)
            for _ in range(2):              # restart once from the end
                r = minimize(lambda th: nll(g, th), x, method="Nelder-Mead",
                             options=dict(xatol=1e-8, fatol=1e-10,
                                          maxiter=20000, maxfev=20000))
                x = r.x
            best = min(best, r.fun)
        return best

    return profile


@pytest.mark.parametrize("prior_sd, old_lo, old_hi", [
    (0.01, 0.08387011427635446, 0.12614144470734584),
    (None, 0.06698087194684033, 0.1815332396141158)])
def test_profile_interval_ends_sit_at_the_chi_square_quantile(
        prior_sd, old_lo, old_hi):
    """README example 3. Up to 0.11.0 the nuisance parameters were not
    fully minimized (tau2 hardly moved from its start), the deviance was
    taken from the smallest grid value and the ends were interpolated
    linearly: the ends returned then were [0.0839, 0.1261] with the 1 %
    flat-level prior and [0.0670, 0.1815] without it, where an
    independent minimization puts the deviance at 2.15 / 4.45 and
    2.63 / 1.16 instead of 3.84. Now both ends sit at 3.84 and g2_hat at
    the independent minimum."""
    cfg = HBTConfig()
    T, r_hat = 60.0, 150e3
    h = np.random.default_rng(0).poisson(
        expected_histogram(EmitterSite(dict(SITE), 1), T, cfg)).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # tau2 on its limit
        res = profile_likelihood_ci(
            h, T, r_hat, cfg,
            c0_prior=None if prior_sd is None else (1.0, prior_sd))
    prof = _independent_profile(h, T, r_hat, cfg, prior_sd)
    # the independent minimum over g, near the returned one
    g_grid = res["g2_hat"] + np.array([-2e-3, 0.0, 2e-3])
    p_grid = np.array([prof(g) for g in g_grid])
    # a parabola through three points: its vertex is the minimum
    c2, c1, c0 = np.polyfit(g_grid - res["g2_hat"], p_grid, 2)
    p_min = c0 - c1 * c1 / (4.0 * c2)
    assert abs(-c1 / (2.0 * c2)) < 2e-4              # g2_hat at the minimum
    crit = chi2.ppf(0.95, 1)
    for g in (res["lo"], res["hi"]):
        assert abs(2.0 * (prof(g) - p_min) - crit) < 0.03
    # the 0.11.0 ends were far from the quantile on at least one side
    worst = max(abs(2.0 * (prof(g) - p_min) - crit) for g in (old_lo, old_hi))
    assert worst > 0.5


# ----------------------------------------------------------------------
# input checks
# ----------------------------------------------------------------------

def test_profile_refuses_bad_level_and_grid():
    cfg = HBTConfig()
    h = np.full(cfg.n_bins, 50.0)
    with pytest.raises(ValueError, match="level"):
        profile_likelihood_ci(h, 1.0, 1e5, cfg, level=95)
    with pytest.raises(ValueError, match="n_grid"):
        profile_likelihood_ci(h, 1.0, 1e5, cfg, n_grid=2)
    with pytest.raises(ValueError, match="c0_prior"):
        profile_likelihood_ci(h, 1.0, 1e5, cfg, c0_prior=0.01)


# ----------------------------------------------------------------------
# check_stationarity
# ----------------------------------------------------------------------

def _steady_series(rng, t, site=None, cfg=None):
    cfg = cfg or HBTConfig()
    mu = expected_histogram(site or EmitterSite(dict(SITE), 1), 1.0, cfg)
    return rng.poisson(np.outer(t, mu)).astype(float)


T_SNAP = np.array([20.0, 30.0, 20.0, 50.0, 40.0, 40.0])


def test_stationarity_statistics_match_scipy():
    """The three statistics against scipy's own chi-square routines
    (an independent implementation), also when bins must be merged."""
    # (the shape statistic without merging is checked in the last part
    # with merging, which covers the general case)
    from sparq import check_stationarity
    cfg = HBTConfig()
    rng = np.random.default_rng(1)
    for t in (T_SNAP, 2.0 * T_SNAP):
        S = _steady_series(rng, t, cfg=cfg)
        r = check_stationarity(S, t, cfg)
        n = S.sum(axis=1)
        ref = chisquare(n, n.sum() * t / t.sum())
        assert np.isclose(r["rate_chi2"], ref.statistic, rtol=1e-12)
        assert np.isclose(r["rate_p"], ref.pvalue, rtol=1e-9)
        c = cfg.n_bins // 2
        mr = np.abs(cfg.bin_centers) >= 0.65 * cfg.tau_max
        tab = np.column_stack([S[:, c], S[:, mr].sum(axis=1)])
        st, p, dof, _ = chi2_contingency(tab, correction=False)
        assert np.isclose(r["window_chi2"], st, rtol=1e-12)
        assert r["window_dof"] == dof and np.isclose(r["window_p"], p)
    # a series with few counts: bins are merged and the shape test is
    # the contingency test of the merged table
    t = np.full(4, 0.5)
    S = _steady_series(np.random.default_rng(2), t, cfg=cfg)
    S[:, cfg.n_bins // 2] += 6.0          # keep the window test allowed
    r = check_stationarity(S, t, cfg)
    assert 2 <= r["shape_bins"] < cfg.n_bins
    need = 5.0 * S.sum() / S.sum(axis=1).min()
    groups, cur, acc = [], [], 0.0
    for k, ck in enumerate(S.sum(axis=0)):
        cur.append(k)
        acc += ck
        if acc >= need:
            groups.append(cur)
            cur, acc = [], 0.0
    if cur:
        groups[-1].extend(cur)
    tab = np.column_stack([S[:, g].sum(axis=1) for g in groups])
    st, p, dof, exp = chi2_contingency(tab, correction=False)
    assert exp.min() >= 5.0
    assert np.isclose(r["shape_chi2"], st, rtol=1e-12)
    assert r["shape_dof"] == dof and np.isclose(r["shape_p"], p)


def test_stationarity_p_values_are_uniform_for_a_steady_source():
    from sparq import check_stationarity
    rng = np.random.default_rng(11)
    ps = {k: [] for k in ("rate_p", "window_p", "shape_p")}
    n_cons = 0
    for _ in range(200):
        r = check_stationarity(_steady_series(rng, T_SNAP), T_SNAP)
        for k in ps:
            ps[k].append(r[k])
        n_cons += r["consistent"]
    for k, v in ps.items():
        v = np.array(v)
        assert kstest(v, "uniform").pvalue > 0.01, k
        assert (v < 0.05).mean() <= 0.09, k
    assert n_cons >= 194            # alpha = 0.01: at most ~2 false alarms


def test_stationarity_finds_a_second_emitter_and_a_dimmer_source():
    from sparq import check_stationarity
    rng = np.random.default_rng(5)
    cfg = HBTConfig()
    one = EmitterSite(dict(SITE), 1)
    two = EmitterSite(dict(SITE), 2)
    dim = EmitterSite(dict(SITE, rate_kcps=120.0), 1)
    a = _steady_series(rng, T_SNAP[:3], one, cfg)
    r = check_stationarity(np.vstack([a, _steady_series(rng, T_SNAP[3:],
                                                        two, cfg)]),
                           T_SNAP, cfg)
    assert r["window_p"] < 1e-6 and not r["consistent"]
    assert np.all(r["g2_window"][3:] > 0.4) and np.all(r["g2_window"][:3] < 0.3)
    r = check_stationarity(np.vstack([a, _steady_series(rng, T_SNAP[3:],
                                                        dim, cfg)]),
                           T_SNAP, cfg)
    # the rate drops by (120/150)^2; the shape does not change
    assert r["rate_p"] < 1e-6 and not r["consistent"]
    assert r["window_p"] > 1e-3 and r["shape_p"] > 1e-3
    assert np.isclose(r["rate_cps"][3:].mean() / r["rate_cps"][:3].mean(),
                      0.64, rtol=0.03)


def test_stationarity_refusals():
    from sparq import check_stationarity
    cfg = HBTConfig()
    S = _steady_series(np.random.default_rng(0), T_SNAP, cfg=cfg)
    with pytest.raises(ValueError, match="J >= 2"):
        check_stationarity(S[:1], T_SNAP[:1], cfg)
    with pytest.raises(ValueError, match="shape"):
        check_stationarity(S[:, :-1], T_SNAP, cfg)
    with pytest.raises(ValueError, match="non-negative"):
        check_stationarity(-S, T_SNAP, cfg)
    with pytest.raises(ValueError, match="durations"):
        check_stationarity(S, -T_SNAP, cfg)
    with pytest.raises(ValueError, match="window test"):
        check_stationarity(S * 0.0 + (S > 0), T_SNAP, cfg)
    with pytest.raises(ValueError, match="alpha"):
        check_stationarity(S, T_SNAP, cfg, alpha=1.5)


# ----------------------------------------------------------------------
# signal fraction from counts, and its propagation
# ----------------------------------------------------------------------

def _clopper_pearson_by_root(k, n, level):
    """Clopper-Pearson limits solved from binomial tail sums (an
    independent route to the beta quantiles used by the package)."""
    a = 0.5 * (1.0 - level)
    lo = 0.0 if k == 0 else brentq(lambda p: binom.sf(k - 1, n, p) - a,
                                   1e-15, 1 - 1e-15, xtol=1e-15)
    hi = 1.0 if k == n else brentq(lambda p: binom.cdf(k, n, p) - a,
                                   1e-15, 1 - 1e-15, xtol=1e-15)
    return lo, hi


def test_signal_fraction_interval_is_clopper_pearson():
    from sparq import signal_fraction_from_counts
    for n_on, t_on, n_off, t_off in ((9000, 10.0, 450, 5.0),
                                     (120, 1.0, 7, 2.0),
                                     (50, 1.0, 0, 1.0)):
        r = signal_fraction_from_counts(n_on, t_on, n_off, t_off, 0.95)
        assert np.isclose(r["rho"], 1.0 - (n_off / t_off) / (n_on / t_on),
                          rtol=1e-14)
        p_lo, p_hi = _clopper_pearson_by_root(n_off, n_on + n_off, 0.95)
        q = lambda p: p * t_on / ((1.0 - p) * t_off)
        assert np.isclose(r["hi"], 1.0 - q(p_lo), rtol=1e-9, atol=1e-12)
        assert np.isclose(r["lo"], 1.0 - q(p_hi), rtol=1e-9)
        assert r["lo"] < r["rho"] <= r["hi"] <= 1.0


def test_signal_fraction_interval_covers():
    from sparq import signal_fraction_from_counts
    rng = np.random.default_rng(3)
    for S, B, t_on, t_off in ((2000.0, 300.0, 1.0, 1.0),
                              (60.0, 20.0, 1.0, 3.0)):
        rho = S / (S + B)
        hit, runs = 0, 2000
        for _ in range(runs):
            n_on = rng.poisson((S + B) * t_on)
            n_off = rng.poisson(B * t_off)
            try:
                r = signal_fraction_from_counts(n_on, t_on, n_off, t_off,
                                                0.9)
            except ValueError:              # background >= on-spot rate
                continue
            hit += r["lo"] <= rho <= r["hi"]
        # Clopper-Pearson is conservative: at least the nominal 0.9
        assert hit / runs >= 0.9 - 3.0 * np.sqrt(0.09 / runs)


def test_corrected_interval_with_rho_interval_is_the_box_extreme():
    from sparq import background_corrected_g2
    gs = np.linspace(0.0, 1.0, 201)
    for ci, rho, rho_ci in (((0.2, 0.3), 0.8, (0.75, 0.85)),
                            ((0.9, 1.2), 0.7, (0.6, 0.8)),
                            ((1.1, 1.3), 0.9, (0.85, 0.95)),
                            ((0.05, 0.15), 0.5, (0.45, 0.6))):
        out = background_corrected_g2(0.5 * (ci[0] + ci[1]), rho, ci=ci,
                                      rho_ci=rho_ci)
        G = ci[0] + gs[:, None] * (ci[1] - ci[0])
        R = rho_ci[0] + gs[None, :] * (rho_ci[1] - rho_ci[0])
        V = np.maximum(1.0 + (G - 1.0) / R ** 2, 0.0)
        assert np.isclose(out["ci"][0], V.min(), rtol=1e-12, atol=1e-15)
        assert np.isclose(out["ci"][1], V.max(), rtol=1e-12, atol=1e-15)
        # without rho_ci the result is exactly that of 0.11.0
        old = background_corrected_g2(0.5 * (ci[0] + ci[1]), rho, ci=ci)
        assert old["ci"] == (max(1.0 + (ci[0] - 1.0) / rho ** 2, 0.0),
                             max(1.0 + (ci[1] - 1.0) / rho ** 2, 0.0))


def test_combined_interval_covers_with_bonferroni_levels():
    """A measured g2 with a 97.5 % interval and rho from counts with a
    97.5 % interval: the corrected interval covers the true g2 in at
    least 95 % of simulated runs."""
    from sparq import background_corrected_g2, signal_fraction_from_counts
    from scipy.stats import norm
    rng = np.random.default_rng(8)
    g_true, S, B = 0.1, 400.0, 100.0
    rho = S / (S + B)
    g_meas_true = 1.0 + rho ** 2 * (g_true - 1.0)
    sd, z = 0.03, norm.ppf(1.0 - 0.025 / 2.0)
    hit, runs = 0, 2000
    for _ in range(runs):
        g_obs = g_meas_true + sd * rng.standard_normal()
        r = signal_fraction_from_counts(rng.poisson(S + B), 1.0,
                                        rng.poisson(B), 1.0, 0.975)
        out = background_corrected_g2(g_obs, r["rho"],
                                      ci=(g_obs - z * sd, g_obs + z * sd),
                                      rho_ci=(r["lo"], r["hi"]))
        hit += out["ci"][0] <= g_true <= out["ci"][1]
    assert hit / runs >= 0.95 - 3.0 * np.sqrt(0.05 * 0.95 / runs)


def test_signal_fraction_and_correction_refusals():
    from sparq import (background_corrected_g2, signal_fraction,
                       signal_fraction_from_counts)
    with pytest.raises(ValueError, match="not below"):
        signal_fraction_from_counts(100, 1.0, 100, 1.0)
    with pytest.raises(ValueError, match="whole"):
        signal_fraction_from_counts(100.5, 1.0, 3, 1.0)
    with pytest.raises(ValueError, match="time"):
        signal_fraction_from_counts(100, 0.0, 3, 1.0)
    with pytest.raises(ValueError, match="n_on"):
        signal_fraction_from_counts(0, 1.0, 0, 1.0)
    with pytest.raises(ValueError, match="reaches down to 0"):
        background_corrected_g2(0.3, 0.5, rho_ci=(-0.1, 0.8))
    with pytest.raises(ValueError, match="rho_ci"):
        background_corrected_g2(0.3, 0.5, rho_ci=(0.6, 0.8))
    with pytest.raises(ValueError, match="finite"):
        background_corrected_g2(float("nan"), 0.5)
    with pytest.raises(ValueError, match="finite"):
        signal_fraction(float("nan"), 1.0)


def test_other_new_input_checks():
    """Inputs that gave NaN or ran on silently up to 0.11.0."""
    from sparq import SPRTCertifier, deadtime_corrected_rate, fit_model
    with pytest.raises(ValueError, match="finite"):
        deadtime_corrected_rate(float("nan"), 45.0)
    cert = SPRTCertifier(EmitterSite(dict(SITE), 1),
                         EmitterSite(dict(SITE), 2))
    with pytest.raises(ValueError, match="T_s"):
        cert.update(np.zeros(cert.cfg.n_bins), float("nan"))
    with pytest.raises(ValueError, match="non-negative"):
        cert.update(-np.ones(cert.cfg.n_bins), 1.0)
    assert cert.llr == 0.0 and cert.T_total == 0.0
    h = np.full(HBTConfig().n_bins, 10.0)
    with pytest.raises(ValueError, match="finite"):
        fit_model(h, 1.0, float("inf"))
    with pytest.raises(ValueError, match="c0_prior"):
        fit_model(h, 1.0, 1e5, c0_prior=(1.0, float("inf")))
