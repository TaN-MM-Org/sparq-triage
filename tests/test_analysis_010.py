"""Measured-histogram analysis, 0.10.0: proportional re-binning, the dip
center to a fraction of a bin, bin-averaged models, the singles-rate
constraint on the flat level, and fits that sit on a search bound."""
import warnings

import numpy as np
import pytest

from sparq import (EmitterSite, HBTConfig, analyze_histogram, locate_dip,
                   expected_histogram, fit_g2_histogram, g2_measured,
                   profile_likelihood_ci, rebin_real)
from sparq.datasets import robust_flat_rate


def _site(n=1, **kw):
    p = dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0, rho=0.95,
             blinking=False)
    p.update(kw)
    return EmitterSite(p, n)


def _grid(w, half_width=90.0):
    n = 2 * int(round(half_width / w)) + 1      # odd: a bin centered on 0
    return HBTConfig(tau_max=n * w / 2, n_bins=n)


@pytest.mark.parametrize("w", [0.3, 0.7, 0.25, 0.5])
def test_split_rebinning_is_flat_and_keeps_counts(w):
    cfg = HBTConfig()
    g = _grid(w)
    d = g.bin_centers
    flat = np.full(d.size, 5.0)
    out, _ = rebin_real(d, flat, cfg, center=0.0)
    inner = out[3:-3]
    assert np.abs(inner / (5.0 / w) - 1.0).max() < 1e-6   # float32 output
    whole, _ = rebin_real(d, flat, cfg, center=0.0, method="whole")
    ripple = whole[3:-3].max() / whole[3:-3].min() - 1.0
    if abs(1.0 / w - round(1.0 / w)) > 1e-9:
        assert ripple > 0.1                   # e.g. 3 and 4 bins of 0.3 ns
    # counts are kept: the grid lies inside the data, so every count in
    # the covered range is kept
    rng = np.random.default_rng(0)
    c = rng.poisson(20.0, d.size).astype(float)
    out, _ = rebin_real(d, c, cfg, center=0.0)
    edges = np.linspace(-cfg.tau_max, cfg.tau_max, cfg.n_bins + 1)
    in_edges = np.r_[d - w / 2, d[-1] + w / 2]
    cum = np.r_[0.0, np.cumsum(c)]
    want = np.interp(edges[-1], in_edges, cum) - np.interp(edges[0],
                                                           in_edges, cum)
    assert abs(out.sum() - want) < 1e-3 * want


def test_split_rebinning_is_symmetric_about_the_center():
    """0.5 ns input on the 1 ns grid, centered on an input bin: the split
    grid is symmetric; 'whole' puts the input bins at 0 and +0.5 ns into
    the zero-delay bin and is not."""
    g = _grid(0.5)
    site = _site()
    mu = expected_histogram(site, 30.0, g)
    cfg = HBTConfig()
    out, _ = rebin_real(g.bin_centers, mu, cfg, center=0.0)
    assert np.abs(out - out[::-1]).max() < 1e-4 * out.max()
    whole, _ = rebin_real(g.bin_centers, mu, cfg, center=0.0, method="whole")
    assert np.abs(whole - whole[::-1]).max() > 1e-2 * whole.max()


def test_dip_center_is_found_to_a_fraction_of_a_bin():
    site = _site()
    for w in (0.3, 0.5, 0.7):
        g = _grid(w)
        mu = expected_histogram(site, 30.0, g)
        d = g.bin_centers + 3.0
        _, c = rebin_real(d, mu, HBTConfig())
        assert abs(c - 3.0) < 1e-3                         # noise-free
    g = _grid(0.5)
    mu = expected_histogram(site, 30.0, g)
    err_new, err_old = [], []
    for seed in range(15):
        h = np.random.default_rng(100 + seed).poisson(mu).astype(float)
        _, c = rebin_real(g.bin_centers + 3.0, h, HBTConfig())
        _, c_old = rebin_real(g.bin_centers + 3.0, h, HBTConfig(),
                              method="whole")                # 0.9.1 locate
        err_new.append(c - 3.0)
        err_old.append(c_old - 3.0)
    rms_new = np.sqrt(np.mean(np.square(err_new)))
    rms_old = np.sqrt(np.mean(np.square(err_old)))
    assert rms_new < 0.5 and rms_new < 0.6 * rms_old


def test_bin_nodes_average_the_bin():
    cfg = HBTConfig()
    nodes, wts = cfg.bin_nodes()
    assert nodes.shape == (121, 8) and abs(wts.sum() - 1.0) < 1e-15
    sub = (np.arange(20000) + 0.5) / 20000 - 0.5
    for sig, t1 in ((0.35, 15.0), (0.0, 15.0), (0.0, 0.3), (0.35, 0.3)):
        f = lambda t: g2_measured(t, t1, 250.0, 0.3, 1, 0.95, sig)
        dense = np.array([f(c + sub).mean() for c in cfg.bin_centers])
        assert np.abs(f(nodes) @ wts - dense).max() < 1e-7
    c, w1 = HBTConfig(bin_average=False).bin_nodes()
    assert np.array_equal(c[:, 0], cfg.bin_centers) and w1.tolist() == [1.0]


def test_fit_of_bin_averaged_data_needs_the_bin_averaged_model():
    """Noise-free bin averages, as a correlator records them: the
    bin-averaged model recovers g2(0); the center-sampled model of
    0.9.1 is off by more than 0.01."""
    site = _site()
    cfg = HBTConfig()
    mu = expected_histogram(site, 30.0, cfg)                 # averaged
    g_avg, ok = fit_g2_histogram(mu, 30.0, 150e3, cfg)
    assert ok and abs(g_avg - site.g2_0) < 1e-3
    g_ctr, _ = fit_g2_histogram(mu, 30.0, 150e3,
                                HBTConfig(bin_average=False))
    assert abs(g_ctr - site.g2_0) > 0.01


@pytest.mark.parametrize("n", [1, 2])
def test_singles_rates_pin_the_flat_level(n):
    """Over 15 simulated runs each, the histogram-only fit misses by up
    to ~0.2 (the flat level trades against a slow shoulder), and most
    misses larger than 0.1 are flagged (bound_matters, with a warning);
    with the singles rates every run is within 0.06, the rms error is
    below 0.03, and nothing is flagged."""
    site = _site(n)
    T = 30.0 if n == 1 else 60.0
    g = _grid(0.5)
    mu = expected_histogram(site, T, g)
    e_free, e_sing, flagged = [], [], []
    for seed in range(15):
        h = np.random.default_rng(1000 + seed).poisson(mu).astype(float)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always", RuntimeWarning)
            r0 = analyze_histogram(g.bin_centers + 7.0, h, T, n_bootstrap=0)
        assert r0["bound_matters"] == any("search bound" in str(x.message)
                                          for x in rec)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)   # none issued
            r1 = analyze_histogram(g.bin_centers + 7.0, h, T,
                                   n_bootstrap=0, singles_cps=(75e3, 75e3))
        assert not r1["bound_matters"]
        e_free.append(r0["g2_0"] - site.g2_0)
        e_sing.append(r1["g2_0"] - site.g2_0)
        flagged.append(r0["bound_matters"])
    e_free, e_sing = np.abs(e_free), np.abs(e_sing)
    flagged = np.array(flagged)
    assert e_sing.max() < 0.06
    assert np.sqrt(np.mean(e_sing ** 2)) < 0.03
    assert e_free.max() > 0.1                  # the histogram alone ...
    # ... and most such misses are flagged (all of them with current
    # SciPy; with SciPy 1.10 one fit of the one-emitter case stops
    # inside the search range and is not)
    assert np.mean(flagged[e_free > 0.1]) >= 0.5
    with pytest.raises(ValueError):
        analyze_histogram(g.bin_centers, h, T, n_bootstrap=0,
                          singles_cps=(-1.0, 5.0))


def test_fit_on_a_bound_is_reported_and_auto_widens():
    """tau1 = 0.15 ns, tau2 = 5 ns under the NV-scale default window:
    the fit sits on the edges, g2(0) is far off, and this is now
    reported (at_bound, and a warning from analyze_histogram). With
    'auto' the window widens and the fit recovers the truth."""
    cfg = HBTConfig(tau_max=10.0125, n_bins=801, sigma_irf=0.005)
    site = EmitterSite(dict(tau1=0.15, tau2=5.0, a=0.4, rho=0.97,
                            rate_kcps=400.0, blinking=False), 1)
    T = 300.0
    h = np.random.default_rng(0).poisson(expected_histogram(site, T, cfg)
                                         ).astype(float)
    r = robust_flat_rate(h, cfg, T)
    g, ok, det = fit_g2_histogram(h, T, r, cfg, return_details=True)
    assert "tau1_low" in det["at_bound"] and abs(g - site.g2_0) > 0.2
    g, ok, det = fit_g2_histogram(h, T, r, cfg, t1_bounds="auto",
                                  t2_bounds="auto", return_details=True)
    assert det["at_bound"] == [] and abs(g - site.g2_0) < 0.05
    assert det["t1_bounds"][0] < 0.15 < det["t1_bounds"][1]
    with pytest.warns(RuntimeWarning, match="search bound"):
        res = analyze_histogram(cfg.bin_centers, h, T, cfg=cfg,
                                n_bootstrap=0)
    assert res["at_bound"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = analyze_histogram(cfg.bin_centers, h, T, cfg=cfg,
                                n_bootstrap=3, t1_bounds="auto",
                                t2_bounds="auto")
    assert res["at_bound"] == [] and abs(res["g2_0"] - site.g2_0) < 0.05
    prof = profile_likelihood_ci(h, T, r, cfg, t1_bounds="auto",
                                 t2_bounds="auto")
    assert prof["lo"] <= site.g2_0 + 0.05 and prof["g2_hat"] < 0.2
    with pytest.raises(ValueError):
        fit_g2_histogram(h, T, r, cfg, t1_bounds="wide")


def test_fit_p_value_is_roughly_calibrated_and_conservative():
    """The fit's error bars are the re-binned bins' own variance (smaller
    than their counts). Neighbouring split bins share input bins, which
    the chi-square ignores, so the p-value comes out somewhat too large:
    small p-values are not more common than they should be."""
    site = _site()
    for w in (0.5, 0.3):
        g = _grid(w)
        mu = expected_histogram(site, 30.0, g)
        ps = []
        for s in range(20):
            h = np.random.default_rng(s).poisson(mu).astype(float)
            r = analyze_histogram(g.bin_centers, h, 30.0, n_bootstrap=0,
                                  singles_cps=(75e3, 75e3))
            ps.append(r["fit_p_value"])
        ps = np.array(ps)
        assert np.mean(ps < 0.05) <= 0.15
        assert 0.4 < ps.mean() < 0.75
    _, _, var = rebin_real(g.bin_centers, h, HBTConfig(), center=0.1,
                           return_variance=True)
    out, _ = rebin_real(g.bin_centers, h, HBTConfig(), center=0.1)
    assert np.all(var <= out + 1e-9) and np.any(var < 0.9 * out)


def test_window_beyond_the_data_is_refused():
    """A correlator window of +-60.5 ns with the dip at +7 ns: the
    analysis window +-60.5 ns around the dip would reach 7 ns beyond the
    data. It is refused, naming the largest window that fits, and the
    data are analyzed correctly with that window."""
    site = _site()
    big = HBTConfig(tau_max=90.5, n_bins=181)
    d = big.bin_centers + 7.0
    keep = np.abs(d) < 60.5
    mu = expected_histogram(site, 30.0, big)[keep]
    d = d[keep]
    with pytest.raises(ValueError, match="at most 53.5 ns"):
        analyze_histogram(d, mu, 30.0, n_bootstrap=0,
                          singles_cps=(75e3, 75e3))
    with pytest.raises(ValueError, match="at most 53.5 ns"):
        analyze_histogram(d, mu, 30.0, n_bootstrap=0, rebin="whole",
                          center=7.0, singles_cps=(75e3, 75e3))
    cfg = HBTConfig(tau_max=53.5, n_bins=107)
    r = analyze_histogram(d, mu, 30.0, cfg=cfg, n_bootstrap=0,
                          singles_cps=(75e3, 75e3))
    assert abs(r["center"] - 7.0) < 1e-3
    assert abs(r["g2_0"] - site.g2_0) < 5e-3
    with pytest.raises(ValueError, match="beyond the data"):
        rebin_real(d, mu, HBTConfig(), center=3.0)


def test_narrow_dip_inside_a_bunching_peak_is_found():
    """tau1 = 0.2 ns in 1 ns bins with a strong 5 ns shoulder: the
    zero-delay bin sits barely above the flat level, far below its
    neighbours. The 0.9.1 smoothed minimum (still used by
    method="whole") puts the dip tens of ns away (here refused, as its
    window no longer fits the data); the symmetry choice finds it, and
    the fit recovers g2(0)."""
    cfg = HBTConfig()
    site = EmitterSite(dict(tau1=0.2, tau2=5.0, a=0.4, rho=0.97,
                            rate_kcps=1000.0, blinking=False), 1)
    mu = expected_histogram(site, 600.0, cfg)
    big = HBTConfig(tau_max=90.5, n_bins=181)
    mub = expected_histogram(site, 600.0, big)
    with pytest.raises(ValueError, match="dip is at -76 ns"):
        rebin_real(big.bin_centers - 4.0, mub, cfg, method="whole")
    _, c = rebin_real(big.bin_centers - 4.0, mub, cfg)
    assert abs(c + 4.0) < 1e-3
    r = analyze_histogram(cfg.bin_centers, mu, 600.0, n_bootstrap=0,
                          singles_cps=(5e5, 5e5), t1_bounds="auto",
                          t2_bounds="auto")
    assert abs(r["center"]) < 1e-3
    assert abs(r["g2_0"] - site.g2_0) < 1e-3


def test_descending_delay_axis_is_accepted():
    g = _grid(0.5)
    mu = expected_histogram(_site(), 30.0, g)
    up = analyze_histogram(g.bin_centers, mu, 30.0, n_bootstrap=0,
                           singles_cps=(75e3, 75e3))
    down = analyze_histogram(g.bin_centers[::-1], mu[::-1], 30.0,
                             n_bootstrap=0, singles_cps=(75e3, 75e3))
    assert abs(up["g2_0"] - down["g2_0"]) < 1e-9
    with pytest.raises(ValueError, match="increasing"):
        rebin_real(np.r_[g.bin_centers[:5], g.bin_centers[4:]],
                   np.r_[mu[:5], mu[4:]], HBTConfig())


@pytest.mark.parametrize("a", [0.0, 0.3])
def test_dip_found_in_a_wide_record(a):
    """+-500 ns of data, dip at +37 ns: flat stretches far from the dip
    are as symmetric as the dip itself, so the dip must be chosen by
    its significance (in review, a symmetry-first choice put it
    hundreds of ns away in some runs)."""
    site = _site(a=a)
    big = HBTConfig(tau_max=500.5, n_bins=1001)
    mu = expected_histogram(site, 30.0, big)
    for seed in range(6):
        h = np.random.default_rng(100 + seed).poisson(mu).astype(float)
        _, c = rebin_real(big.bin_centers + 37.0, h, HBTConfig())
        assert abs(c - 37.0) < 1.0


def test_dip_just_beyond_the_fitting_range_is_refused():
    """Bins from -65 to +65 ns (edges +-65.5) with the dip at +6 ns: the
    +-60.5 ns window needs the center at <= 5 ns. It is refused, not
    moved to 5."""
    site = _site()
    big = HBTConfig(tau_max=120.5, n_bins=241)
    d = big.bin_centers + 6.0
    keep = np.abs(d) <= 65.0
    for seed in range(3):
        h = np.random.default_rng(seed).poisson(
            expected_histogram(site, 30.0, big)[keep]).astype(float)
        with pytest.raises(ValueError, match="beyond the data"):
            rebin_real(d[keep], h, HBTConfig())


def test_unclear_dips_are_flagged():
    """A clear dip in a +-500 ns record is not flagged; a 0.5 ns dip on
    1 ns bins in the same record often cannot be placed (a noise dip
    elsewhere is about as significant), and every such run is flagged,
    with a warning from analyze_histogram."""
    big = HBTConfig(tau_max=500.5, n_bins=1001)
    d = big.bin_centers + 37.0
    clear = expected_histogram(_site(), 30.0, big)
    fast = _site(tau1=0.5, tau2=5.0, a=0.4)
    faint = expected_histogram(fast, 30.0, big)
    for seed in range(4):
        rng = np.random.default_rng(seed)
        loc = locate_dip(d, rng.poisson(clear).astype(float))
        assert not loc["ambiguous"]
        assert abs(loc["center"] - 37.0) <= loc["width"]   # coarse
        loc = locate_dip(d, rng.poisson(faint).astype(float))
        assert loc["ambiguous"]
    h = np.random.default_rng(0).poisson(faint).astype(float)
    with pytest.warns(RuntimeWarning, match="stand out"):
        r = analyze_histogram(d, h, 30.0, n_bootstrap=0,
                              singles_cps=(75e3, 75e3))
    assert r["dip_ambiguous"] is True
