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

from .bayes import bayesian_g2
from .physics import EmitterSite, HBTConfig, expected_histogram

__all__ = ["site_from_numbers", "expected_posterior",
           "required_acquisition_time"]


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
