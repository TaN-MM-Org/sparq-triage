"""Tests of the 0.14.0 changes: the lifetime fit of a TCSPC decay
histogram (`expected_decay`, `fit_lifetime`)."""
import warnings

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import exponnorm, norm

from sparq import expected_decay, fit_lifetime

T_REP = 12.5


def _edges(d):
    mid = 0.5 * (d[1:] + d[:-1])
    return np.concatenate([[2 * d[0] - mid[0]], mid, [2 * d[-1] - mid[-1]]])


def _cdf_ref(x, tau, s):
    """Distribution function of an exponential decay (start 0, lifetime
    tau) blurred by Gaussian jitter s: SciPy's exponnorm, or the pure
    exponential for s = 0."""
    x = np.asarray(x, float)
    if s == 0:
        return np.where(x > 0, -np.expm1(-np.maximum(x, 0) / tau), 0.0)
    return exponnorm.cdf(x, tau / s, loc=0.0, scale=s)


def _decay_ref(d, area, tau, t0, s, t_rep=None, bg=0.0):
    """Mean counts per bin written here from exponnorm: one decay, or with
    t_rep the direct sum over the decays of the pulses m = -3 ... M, M far
    enough back that e^(-M t_rep / tau) < 1e-18."""
    e = _edges(d)
    if t_rep is None:
        return bg * np.diff(e) + area * np.diff(_cdf_ref(e - t0, tau, s))
    M = int(np.ceil(42 * tau / t_rep)) + 3
    m = np.arange(-3, M + 1)[:, None]
    F = _cdf_ref((e - t0)[None, :] + m * t_rep, tau, s)
    return bg * np.diff(e) + area * np.diff(F, axis=1).sum(axis=0)


# ----------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------

def test_decay_model_matches_exponnorm_and_the_convolution_integral():
    """The bin integrals equal differences of SciPy's exponnorm
    distribution function (an independent implementation) to 1e-12 of the
    decay's photons, also where the jitter is 50 times the lifetime and
    where it is 1/2000 of it; the distribution function equals the
    convolution integral int_0^inf (1/tau) e^(-y/tau) Phi((x - y)/s) dy,
    computed by quadrature, to 1e-9."""
    from sparq.lifetime import _lower, _upper
    d = np.linspace(-3.0, 40.0, 431)
    for tau, s in ((1.0, 0.1), (5.0, 0.3), (0.01, 0.5), (0.3, 0.3),
                   (100.0, 0.05), (2.0, 0.0)):
        mu = expected_decay(d, 1.0, tau, 0.7, s)
        ref = _decay_ref(d, 1.0, tau, 0.7, s)
        assert np.max(np.abs(mu - ref)) < 1e-12, (tau, s)
        if s == 0:
            continue
        for x in (-2.0, -0.4, 0.0, 0.05, 0.6, 3.0, 12.0):
            conv = quad(lambda y: np.exp(-y / tau) / tau
                        * norm.cdf((x - y) / s), 0.0, np.inf,
                        epsabs=1e-13, epsrel=1e-12, limit=200)[0]
            F = _lower(min(x, 0.0), tau, s) if x < 0 \
                else 1.0 - _upper(x, tau, s)
            assert abs(F - conv) < 1e-9, (tau, s, x)


def test_periodic_sum_equals_the_direct_sum_over_pulses():
    """With a laser period the decays of earlier pulses add up: the closed
    geometric sum equals the direct sum over the pulses (to 1e-12 of the
    largest bin), and over exactly one period the bins hold all the
    photons of one pulse (sum 1 to 1e-12), for lifetimes from a twentieth
    of the period to more than twice it."""
    d = (np.arange(500) + 0.5) * T_REP / 500
    for tau in (0.5, 3.0, 8.0, 30.0):
        for s in (0.0, 0.05, 0.3):
            mu = expected_decay(d, 1.0, tau, 3.0, s, T_REP)
            ref = _decay_ref(d, 1.0, tau, 3.0, s, T_REP)
            assert np.max(np.abs(mu - ref)) < 1e-12 * ref.max(), (tau, s)
            assert abs(mu.sum() - 1.0) < 1e-12
    # several components and a background add up; a decreasing axis
    # gives the same bins in reverse
    mu = expected_decay(d, [2.0, 3.0], [1.0, 6.0], 2.0, 0.1, T_REP, 4.0)
    ref = (_decay_ref(d, 2.0, 1.0, 2.0, 0.1, T_REP)
           + _decay_ref(d, 3.0, 6.0, 2.0, 0.1, T_REP) + 4.0 * T_REP / 500)
    assert np.max(np.abs(mu - ref)) < 1e-12 * ref.max()
    rev = expected_decay(d[::-1], [2.0, 3.0], [1.0, 6.0], 2.0, 0.1, T_REP,
                         4.0)
    assert np.max(np.abs(rev[::-1] - mu)) < 1e-12 * mu.max()


# ----------------------------------------------------------------------
# the fit
# ----------------------------------------------------------------------

def test_fit_recovers_noise_free_decays():
    """On noise-free histograms built from exponnorm (not from the
    package) the fit returns every parameter: one lifetime, one whose
    decay reaches far into the next periods, two lifetimes, and a single
    decay without a laser period, also with the jitter and t0 given."""
    d = (np.arange(400) + 0.5) * T_REP / 400
    for tau in (3.2, 8.0):
        mu = _decay_ref(d, 2e5, tau, 2.0, 0.15, T_REP, bg=50.0)
        f = fit_lifetime(d, mu, t_rep=T_REP, interval=False)
        assert abs(f["tau"][0] / tau - 1) < 1e-6
        assert abs(f["sigma_irf"] / 0.15 - 1) < 1e-5
        assert abs(f["t0"] - 2.0) < 1e-5
        assert abs(f["area"][0] / 2e5 - 1) < 1e-6
        assert abs(f["background"] / 50.0 - 1) < 1e-4
        assert f["p_value"] > 0.99 and f["at_bound"] == []
    mu = (_decay_ref(d, 1.5e5, 0.8, 2.0, 0.1, T_REP, bg=20.0)
          + _decay_ref(d, 5e4, 4.0, 2.0, 0.1, T_REP))
    f = fit_lifetime(d, mu, t_rep=T_REP, n_components=2, interval=False)
    assert np.allclose(f["tau"], [0.8, 4.0], rtol=1e-5)
    assert np.allclose(f["fraction"], [0.75, 0.25], rtol=1e-5)
    d1 = np.arange(0.025, 60.0, 0.05)
    mu = _decay_ref(d1, 1e5, 2.5, 5.0, 0.2, bg=10.0)
    for kw in ({}, dict(sigma_irf=0.2, t0=5.0)):
        f = fit_lifetime(d1, mu, interval=False, **kw)
        assert abs(f["tau"][0] / 2.5 - 1) < 1e-6
        assert abs(f["area"][0] / 1e5 - 1) < 1e-6


def _nll_full(d, h, th):
    """Poisson NLL of one periodic decay in (log A, log tau, t0, log s,
    B), written independently of sparq.lifetime."""
    A, tau, t0, s, B = np.exp(th[0]), np.exp(th[1]), th[2], \
        np.exp(th[3]), th[4]
    if B < 0:
        return 1e30
    m = _decay_ref(d, A, tau, t0, s, T_REP, bg=B)
    return float(np.sum(m - h * np.log(m)))


def _nm(f, x0):
    best = None
    x = np.array(x0, float)
    for _ in range(2):
        r = minimize(f, x, method="Nelder-Mead",
                     options=dict(xatol=1e-9, fatol=1e-9, maxiter=20000,
                                  maxfev=20000))
        x = r.x
        if best is None or r.fun < best.fun:
            best = r
    return best


def test_fit_and_interval_against_an_independent_minimization():
    """On a noisy decay an independent Nelder-Mead minimization of the
    Poisson likelihood (model from exponnorm) finds no better fit, and at
    both ends of the 95 % interval its profile deviance is the
    chi-square quantile 3.84 within 0.01; the truth lies inside."""
    d = (np.arange(100) + 0.5) * T_REP / 100
    rng = np.random.default_rng(5)
    h = rng.poisson(_decay_ref(d, 2e4, 1.8, 2.0, 0.12, T_REP, bg=40.0))
    f = fit_lifetime(d, h, t_rep=T_REP)
    th = [np.log(f["area"][0]), np.log(f["tau"][0]), f["t0"],
          np.log(f["sigma_irf"]), f["background"]]
    full = _nm(lambda x: _nll_full(d, h, x), th)
    assert _nll_full(d, h, th) <= full.fun + 1e-6
    for end in (f["tau_lo"][0], f["tau_hi"][0]):
        def prof(rest):
            return _nll_full(d, h, np.insert(rest, 1, np.log(end)))
        dev = 2.0 * (_nm(prof, np.delete(th, 1)).fun - full.fun)
        assert abs(dev - 3.8415) < 0.01, (end, dev)
    assert f["tau_lo"][0] < 1.8 < f["tau_hi"][0]
    assert f["resolved"] and 0.01 < f["p_value"] <= 1.0


def test_lifetime_interval_coverage():
    """95 % intervals over 20 simulated dim decays (about 3100 photons in
    125 bins; the jitter given, which keeps the test short): at least 16
    hold the true lifetime (expected 19; 16 is three standard errors
    below). Done once outside the suite with 200 decays each of this
    emitter (jitter fitted) and a bright one: see the README."""
    d = (np.arange(125) + 0.5) * 0.1
    mu = _decay_ref(d, 3000.0, 2.0, 1.5, 0.1, T_REP, bg=10.0)
    rng = np.random.default_rng(11)
    hits = 0
    for _ in range(20):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            f = fit_lifetime(d, rng.poisson(mu), t_rep=T_REP,
                             sigma_irf=0.1)
        hi = f["tau_hi"][0] if np.isfinite(f["tau_hi"][0]) else np.inf
        hits += f["tau_lo"][0] <= 2.0 <= hi
    assert hits >= 16


def test_p_value_flags_a_wrong_model():
    """Two lifetimes fitted with one, and a long lifetime fitted without
    the laser period (so the decays of earlier pulses are taken for
    background), are rejected (p < 1e-6); the right models are not."""
    d = (np.arange(500) + 0.5) * T_REP / 500
    rng = np.random.default_rng(3)
    h = rng.poisson(_decay_ref(d, 1.5e5, 0.8, 2.0, 0.1, T_REP, bg=20.0)
                    + _decay_ref(d, 5e4, 4.0, 2.0, 0.1, T_REP))
    one = fit_lifetime(d, h, t_rep=T_REP, interval=False)
    two = fit_lifetime(d, h, t_rep=T_REP, n_components=2, interval=False)
    assert one["p_value"] < 1e-6 and two["p_value"] > 0.01
    h = rng.poisson(_decay_ref(d, 1e5, 8.0, 2.0, 0.15, T_REP, bg=10.0))
    no_wrap = fit_lifetime(d, h, interval=False)
    wrap = fit_lifetime(d, h, t_rep=T_REP)
    assert no_wrap["p_value"] < 1e-6 and no_wrap["tau"][0] < 6.0
    assert wrap["p_value"] > 0.01
    assert wrap["tau_lo"][0] < 8.0 < wrap["tau_hi"][0]
    # a short lifetime (e^(-12.5/0.8) = 2e-7 of the decay is left at the
    # next pulse): the fits with and without the period agree
    h = rng.poisson(_decay_ref(d, 5e4, 0.8, 2.0, 0.1, T_REP, bg=20.0))
    a = fit_lifetime(d, h, t_rep=T_REP, interval=False)
    b = fit_lifetime(d, h, interval=False)
    assert abs(a["tau"][0] / b["tau"][0] - 1) < 1e-4


def test_unmeasurable_lifetimes_give_open_intervals():
    """A decay much longer than the record (no laser period) leaves the
    upper end open; one much shorter than the known jitter leaves the
    lower end at 0; both warn and set resolved to False."""
    rng = np.random.default_rng(8)
    d = np.arange(0.05, 20.0, 0.1)
    h = rng.poisson(_decay_ref(d, 2e4, 2000.0, 2.0, 0.1, bg=5.0))
    with pytest.warns(RuntimeWarning, match="do not bound"):
        f = fit_lifetime(d, h, sigma_irf=0.1)
    assert np.isnan(f["tau_hi"][0]) and f["resolved"] is False
    h = rng.poisson(_decay_ref(d, 2e4, 0.002, 2.0, 0.5, bg=5.0))
    with pytest.warns(RuntimeWarning, match="do not bound"):
        f = fit_lifetime(d, h, sigma_irf=0.5)
    assert f["tau_lo"][0] == 0.0 and f["resolved"] is False


def test_lifetime_refusals():
    d = (np.arange(50) + 0.5) * 0.25
    h = np.full(50, 10.0)
    bad = [dict(counts=-h), dict(counts=0 * h), dict(counts=h[:40]),
           dict(delay=np.r_[d[:49], np.nan]), dict(counts=np.r_[h[:49],
                                                                np.inf]),
           dict(delay=np.r_[d[:25], d[:25]]), dict(t_rep=0.0),
           dict(t_rep=np.inf), dict(t_rep="80 MHz"), dict(n_components=3),
           dict(n_components=True), dict(sigma_irf=-0.1),
           dict(sigma_irf=np.nan), dict(t0=np.inf), dict(level=1.0),
           dict(delay=d[:4], counts=h[:4])]
    for kw in bad:
        args = dict(delay=d, counts=h)
        args.update(kw)
        with pytest.raises(ValueError):
            fit_lifetime(**args)
    for kw in (dict(tau=0.0), dict(tau=[1.0, 2.0]), dict(area=-1.0),
               dict(sigma_irf=-1.0), dict(t0=np.nan), dict(t_rep=-1.0),
               dict(background=-1.0)):
        args = dict(delay=d, area=1.0, tau=1.0, t0=1.0)
        args.update(kw)
        with pytest.raises(ValueError):
            expected_decay(**args)


# ----------------------------------------------------------------------
# PTU writer without a Python loop: the same file, byte for byte
# ----------------------------------------------------------------------

def _t2_records_0130(ch, ticks, family, wrap, many):
    """The record loop of sparq.ptu.save_ptu_t2 as it was up to 0.13.0."""
    recs = []
    base = 0
    for c, k in zip(ch.tolist(), ticks.tolist()):
        n_ofl = (k - base) // wrap
        while n_ofl > 0:
            m = min(n_ofl, 0x1FFFFFF) if many else 1
            if family == "PT":
                recs.append((15 << 28) | 0)           # overflow
            else:
                recs.append((1 << 31) | (63 << 25) | (m if many else 0))
            base += m * wrap
            n_ofl -= m
        tt = k - base
        if family == "PT":
            recs.append((c << 28) | tt)
        elif c == 0:
            recs.append((1 << 31) | (0 << 25) | tt)   # sync event
        else:
            recs.append(((c - 1) << 25) | tt)
    return np.array(recs, dtype="<u4")


def test_ptu_writer_gives_the_0130_records(tmp_path):
    """For every T2 record type, with sync events, gaps of several wrap
    periods and (where one record may carry many) a gap of more than
    twice the most one record can carry, the records equal those of the
    0.13.0 loop exactly, and the whole file is the same byte for byte."""
    from sparq.ptu import RECORD_TYPES, _t2_records
    rng = np.random.default_rng(0)
    for name, (code, mode, fam) in RECORD_TYPES.items():
        if mode != "T2":
            continue
        wrap, many = ((210698240, False) if fam == "PT" else
                      (33552000, False) if fam == "HH1" else
                      (33554432, True))
        n = 3000
        gaps = rng.exponential(5e6, n).astype(np.int64)
        gaps[rng.integers(0, n, 20)] += rng.integers(1, 5, 20) * wrap
        if many:
            gaps[7] += (2 * 0x1FFFFFF + 5) * wrap
        ticks = np.cumsum(gaps)
        ticks[0] = 0
        ch = rng.integers(0, 5 if fam == "PT" else 65, n).astype(np.int64)
        new = _t2_records(ch, ticks, fam, wrap, many)
        old = _t2_records_0130(ch, ticks, fam, wrap, many)
        assert new.dtype == old.dtype and np.array_equal(new, old), name
        # the file: its records are those of the old loop on the ticks
        # that save_ptu_t2 rounds the times to
        p = tmp_path / f"{name}.ptu"
        t_ns = ticks * 1e-3                         # ns at 1 ps ticks
        from sparq import save_ptu_t2
        save_ptu_t2(p, ch, t_ns, 1e-12, name)
        k = np.rint(t_ns * 1e-9 / 1e-12).astype(np.int64)
        old = _t2_records_0130(ch, k, fam, wrap, many)
        raw = p.read_bytes()
        assert raw.endswith(old.tobytes())
        assert len(raw) - old.nbytes == raw.index(b"Header_End") + 48
    assert _t2_records(np.zeros(0, np.int64), np.zeros(0, np.int64), "HH2",
                       33554432, True).size == 0


def test_bin_nodes_unchanged():
    """The cached Gauss-Legendre nodes are those NumPy gives (exactly:
    the same call), so the bin averages do not change."""
    from sparq import HBTConfig
    cfg = HBTConfig()
    x, wt = np.polynomial.legendre.leggauss(4)
    half = 0.5 * cfg.bin_width
    off = np.concatenate([(x - 1.0) * 0.5 * half, (x + 1.0) * 0.5 * half])
    nodes, w = cfg.bin_nodes()
    assert np.array_equal(nodes, cfg.bin_centers[:, None] + off[None, :])
    assert np.array_equal(w, np.concatenate([wt, wt]) / 4.0)
    nodes[0, 0] = 1e9                    # callers may change their copy
    assert np.array_equal(cfg.bin_nodes()[0][0], (cfg.bin_centers[0]
                                                  + off))


def test_saturation_without_background_refuses_counts_at_zero_power():
    """Up to 0.13.0, counts at zero power with background='none' gave
    R_inf = 429 and P_sat = 0.012 with an infinite deviance without
    intervals (the fit with dark counts, background='constant', gives
    1974 and 190), and a NaN error from inside the interval search with
    them. It is now refused; without counts at zero power the fit
    runs."""
    from sparq import fit_saturation
    P = [0, 10, 20, 40, 80, 160]
    with pytest.raises(ValueError, match="zero power"):
        fit_saturation(P, [5, 100, 190, 350, 600, 900], 1.0,
                       background="none", interval=None)
    with pytest.raises(ValueError, match="zero power"):
        fit_saturation(P, [5, 100, 190, 350, 600, 900], 1.0,
                       background="none")
    r = fit_saturation(P, [0, 100, 190, 350, 600, 900], 1.0,
                       background="none")
    assert np.isfinite(r["deviance"]) and r["P_sat_lo"] < r["P_sat"]
