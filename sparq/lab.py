"""Plan an HBT acquisition before running it.

The package's inference tools answer "is this a single emitter?" from
data that already exists. This module answers the planning question a
lab faces first: how long must the correlator run before the answer
can reach a chosen confidence -- and can it ever?

The planning device is deliberate and stated plainly: evaluate the
exact Bayesian verdict (`sparq.bayes.bayesian_g2`) on the AVERAGE
histogram the site would produce (`sparq.physics.expected_histogram`)
-- typical-data planning. The answer is the acquisition time at which
the posterior computed on the average data reaches the confidence;
individual runs scatter around it, so budget margin on top (the
seeded Monte-Carlo test in the suite shows a 4x margin certifying
more than 90% of runs at the default settings).

One exact fact makes the impossible case refusable rather than
merely slow: the ratio of the central-window mean to the reference
mean does not depend on the acquisition time at all (both scale
linearly with it), so a site whose window-averaged g2 is not below
the threshold can NEVER be certified below it, at any time -- and
`required_acquisition_time` refuses with that number instead of
searching forever.
"""
from __future__ import annotations

import numpy as np
from scipy.special import betainc
from scipy.stats import poisson

from .bayes import bayesian_g2
from .physics import EmitterSite, HBTConfig, expected_histogram

__all__ = ["site_from_numbers", "expected_posterior",
           "required_acquisition_time", "certification_probability",
           "assured_acquisition_time"]


def site_from_numbers(tau1_ns, tau2_ns, a, rate_kcps, rho=1.0,
                      n_emitters=1):
    """An `EmitterSite` from plainly named numbers.

    tau1_ns, tau2_ns : antibunching and shelving timescales (ns).
    a : shelving amplitude.
    rate_kcps : total detected count rate, both arms (kilocounts/s).
    rho : signal fraction (1 = background-free).
    n_emitters : how many emitters the site holds.
    """
    for name, v in (("tau1_ns", tau1_ns), ("tau2_ns", tau2_ns),
                    ("rate_kcps", rate_kcps)):
        if not (np.isfinite(v) and v > 0.0):
            raise ValueError(f"{name} must be finite and positive")
    if not (0.0 <= a and np.isfinite(a)):
        raise ValueError("a must be finite and >= 0")
    if not (0.0 < rho <= 1.0):
        raise ValueError("rho must lie in (0, 1]")
    if not (isinstance(n_emitters, (int, np.integer))
            and n_emitters >= 1):
        raise ValueError("n_emitters must be an integer >= 1")
    return EmitterSite(params=dict(tau1=float(tau1_ns),
                                   tau2=float(tau2_ns), a=float(a),
                                   rate_kcps=float(rate_kcps),
                                   rho=float(rho), blinking=False),
                       n_emitters=int(n_emitters))


def expected_posterior(site, T_s, cfg=None, threshold=0.5,
                       n_center_bins=1, lo_frac=0.65):
    """The verdict typical data would give after T_s seconds.

    Evaluates the exact closed-form posterior on the site's average
    histogram. Returns dict(prob_below, interval, g2_window, k0, kr):
    the single-emitter verdict probability P[g < threshold], the 95%
    credible interval, the time-independent window-averaged g2 the
    posterior concentrates on, and the expected window counts.
    """
    if cfg is None:
        cfg = HBTConfig()
    if not (np.isfinite(T_s) and T_s > 0.0):
        raise ValueError("T_s must be finite and positive (seconds)")
    mu = expected_histogram(site, float(T_s), cfg)
    post = bayesian_g2(mu, cfg, n_center_bins=n_center_bins,
                       lo_frac=lo_frac)
    g_window = (post.k0 / post.n0) / (post.kr / post.nr)
    lo, hi = post.credible_interval(0.95)
    return {"prob_below": post.prob_below(threshold),
            "interval": (float(lo), float(hi)),
            "g2_window": float(g_window),
            "k0": float(post.k0), "kr": float(post.kr)}


def required_acquisition_time(site, confidence=0.95, threshold=0.5,
                              cfg=None, t_max_s=3600.0,
                              n_center_bins=1, lo_frac=0.65):
    """Smallest acquisition time whose typical data certify the site.

    Bisects the time at which `expected_posterior`'s verdict
    probability reaches the confidence. Refuses -- by the exact
    time-independence of the window ratio -- a site whose
    window-averaged g2 is not below the threshold, naming that number:
    no acquisition time can certify it. Refuses a target not reached
    by `t_max_s` and reports the probability there, so the practical
    ceiling is visible instead of silent.

    Returns (T_s, report) with `report` the `expected_posterior`
    result at the returned time.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie in (0, 1)")
    if not (np.isfinite(t_max_s) and t_max_s > 0.0):
        raise ValueError("t_max_s must be finite and positive")
    cfg = cfg if cfg is not None else HBTConfig()

    def prob(t):
        return expected_posterior(site, t, cfg, threshold,
                                  n_center_bins, lo_frac)["prob_below"]

    probe = expected_posterior(site, 1.0, cfg, threshold,
                               n_center_bins, lo_frac)
    if probe["g2_window"] >= threshold:
        raise ValueError(
            f"the site's window-averaged g2 is {probe['g2_window']:.3f}"
            f" >= the threshold {threshold}: the window ratio does not "
            "change with acquisition time, so no run length certifies "
            "this site below the threshold")
    if prob(t_max_s) < confidence:
        raise ValueError(
            f"confidence {confidence} is not reached within t_max_s = "
            f"{t_max_s:.3g} s (probability there: {prob(t_max_s):.3f});"
            " raise t_max_s, improve the count rate, or lower the "
            "confidence")
    lo, hi = 0.0, float(t_max_s)
    if prob(min(1e-6 * t_max_s, 1e-3)) >= confidence:
        lo = 0.0
        hi = min(1e-6 * t_max_s, 1e-3)
    for _ in range(80):
        mid = 0.5 * (lo + hi) if lo > 0.0 else 0.5 * hi
        if prob(mid) >= confidence:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-6 * hi:
            break
    report = expected_posterior(site, hi, cfg, threshold,
                                n_center_bins, lo_frac)
    return float(hi), report


# ----------------------------------------------------------------------
# exact planning over the scatter of real runs (new in 0.10.0)
# ----------------------------------------------------------------------

def _window_means(site, T_s, cfg, n_center_bins, lo_frac):
    """Expected summed counts in the central and reference windows and
    their bin numbers, from the same window definitions as bayesian_g2."""
    mu = expected_histogram(site, float(T_s), cfg)
    post = bayesian_g2(mu, cfg, n_center_bins=n_center_bins,
                       lo_frac=lo_frac)
    return post.k0, post.kr, post.n0, post.nr


def certification_probability(site, T_s, confidence=0.95, threshold=0.5,
                              cfg=None, n_center_bins=1, lo_frac=0.65,
                              prior=(0.5, 0.0), tail=1e-12):
    """The probability that a run of T_s seconds on this site certifies
    it: that `bayesian_g2` on the measured histogram gives
    P[g < threshold] >= confidence (new in 0.10.0).

    Exact over the Poisson scatter of the two window counts that the
    verdict uses. The summed central-window count k0 and reference
    count kr are independent Poisson variables with the means of the
    site's expected histogram. For each kr the verdict holds exactly for
    k0 up to a largest value k0*(kr) (more central counts only lower
    the verdict probability), so
        P = sum_kr Poisson(kr; mu_r) * PoissonCDF(k0*(kr); mu_0),
    summed over all kr outside a Poisson tail of total mass <= `tail`
    (which bounds the error). No simulation is involved; the tests
    compare it with simulated runs.
    """
    if cfg is None:
        cfg = HBTConfig()
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie in (0, 1)")
    if not (np.isfinite(T_s) and T_s > 0.0):
        raise ValueError("T_s must be finite and positive (seconds)")
    a, b = float(prior[0]), float(prior[1])
    if a <= 0.0 or b < 0.0:
        raise ValueError("prior must have shape a > 0 and rate b >= 0")
    mu0, mur, n0, nr = _window_means(site, T_s, cfg, n_center_bins,
                                     lo_frac)
    kr_lo = int(poisson.ppf(0.5 * tail, mur))
    kr_hi = int(poisson.isf(0.5 * tail, mur)) + 1
    kr = np.arange(max(kr_lo, 0), kr_hi + 1)
    beta0, betar = b + n0, b + nr
    r = beta0 * threshold / betar
    x = r / (1.0 + r)

    def verdict(k0):                       # P[g < threshold | k0, kr]
        return betainc(a + k0, a + kr, x)

    # largest k0 with verdict >= confidence, by bisection on integers
    hi_k = np.full(kr.shape, float(max(10.0, 10.0 * mu0 + 100.0)))
    while True:
        bad = verdict(hi_k) >= confidence
        if not bad.any():
            break
        hi_k[bad] *= 2.0
    lo_k = np.full(kr.shape, -1.0)         # verdict(-1) taken as true
    ok0 = verdict(np.zeros_like(hi_k)) >= confidence
    while np.any(hi_k - lo_k > 1):
        mid = np.floor(0.5 * (lo_k + hi_k))
        good = verdict(np.maximum(mid, 0.0)) >= confidence
        good &= mid >= 0
        lo_k = np.where(good, mid, lo_k)
        hi_k = np.where(good, hi_k, mid)
    k0_star = np.where(ok0, lo_k, -1.0)
    p = float(np.sum(poisson.pmf(kr, mur) * poisson.cdf(k0_star, mu0)))
    return p


def assured_acquisition_time(site, assurance=0.9, confidence=0.95,
                             threshold=0.5, cfg=None, t_max_s=3600.0,
                             n_center_bins=1, lo_frac=0.65):
    """Shortest acquisition time at which a run certifies the site with
    probability at least `assurance` (new in 0.10.0).

    `required_acquisition_time` plans on the average histogram, so a
    sizeable share of real runs of that length fall short (41 % for
    the README's example site). This plans on the exact certification
    probability (`certification_probability`) over the Poisson scatter
    of runs instead.

    That probability rises with time but not smoothly: counts are
    whole numbers, and it can dip by about 0.01 just after rising past
    a value (for the README's site it is 0.905 at 0.9 T and 0.896 at
    0.97 T). So the time returned is the one after which it STAYS at or
    above `assurance`: a first estimate by halving from `t_max_s` and
    bisecting, then a scan from half to four times that estimate in
    steps of 0.2 %, returning the first scan time after the last one
    below `assurance`. Beyond four times the estimate the probability
    is not scanned (in the tests it is then far above the assurance).

    Refuses, as `required_acquisition_time` does, a site whose window
    g2 is not below the threshold, and an assurance not reached by
    `t_max_s`.

    Returns (T_s, the probability at T_s, the lowest probability on
    the scan from T_s on).
    """
    if not (0.0 < assurance < 1.0):
        raise ValueError("assurance must lie in (0, 1)")
    if not (np.isfinite(t_max_s) and t_max_s > 0.0):
        raise ValueError("t_max_s must be finite and positive")
    cfg = cfg if cfg is not None else HBTConfig()
    probe = expected_posterior(site, 1.0, cfg, threshold, n_center_bins,
                               lo_frac)
    if probe["g2_window"] >= threshold:
        raise ValueError(
            f"the site's window-averaged g2 is {probe['g2_window']:.3f}"
            f" >= the threshold {threshold}: no run length certifies "
            "this site with high probability")

    def P(t):
        return certification_probability(site, t, confidence, threshold,
                                          cfg, n_center_bins, lo_frac)

    p_max = P(t_max_s)
    if p_max < assurance:
        raise ValueError(
            f"assurance {assurance} is not reached within t_max_s = "
            f"{t_max_s:.3g} s (probability there: {p_max:.3f})")
    # halve until the assurance is missed, then bisect
    t_hi = float(t_max_s)
    t_lo = t_hi
    while t_lo > 1e-9 * t_max_s and P(t_lo) >= assurance:
        t_hi = t_lo
        t_lo = t_lo / 2.0
    for _ in range(60):
        mid = 0.5 * (t_lo + t_hi)
        if P(mid) >= assurance:
            t_hi = mid
        else:
            t_lo = mid
        if t_hi - t_lo <= 1e-4 * t_hi:
            break
    # scan for the time after which the probability stays up
    t_end = min(4.0 * t_hi, float(t_max_s))
    n = int(np.ceil(np.log(t_end / (0.5 * t_hi)) / np.log(1.002))) + 1
    grid = 0.5 * t_hi * 1.002 ** np.arange(n)
    grid = grid[grid <= t_end * (1 + 1e-12)]
    pg = np.array([P(t) for t in grid])
    below = np.nonzero(pg < assurance)[0]
    i = int(below[-1]) + 1 if below.size else 0
    if i >= grid.size:
        return float(t_hi), float(P(t_hi)), float(P(t_hi))
    return float(grid[i]), float(pg[i]), float(pg[i:].min())
