"""General-purpose analysis of measured HBT data with honest uncertainties.

This module makes the package's conventional analysis pipeline usable on
anyone's data, from any emitter platform, without touching the twin or the
learned estimators: give it a delay axis (ns) and raw coincidence counts,
and it returns the g2(0) estimate together with a parametric-bootstrap
confidence interval and a single-emitter verdict.

The bootstrap treats the measured counts as the per-bin Poisson means,
draws B synthetic histograms, re-runs the full pipeline (dip centering,
re-binning, flat-level normalization, multi-start Levenberg-Marquardt fit)
on each draw, and reports percentiles of the resulting g2(0) sample.  This
propagates shot noise through every step of the analysis rather than
quoting the fit's linearized parameter error.

CW histograms go through :func:`analyze_histogram`; pulsed (comb)
histograms through :func:`analyze_pulsed`, which reuses the comb
calibration and side-peak-area analysis of :mod:`sparq.pulsed`.
"""
from __future__ import annotations

import numpy as np

from .datasets import rebin_real, robust_flat_rate
from .physics import HBTConfig
from .pulsed import T_REP_NS, calibrate_comb, g2_peak_area
from scipy.optimize import curve_fit


def fit_g2_histogram(hist, T_s, r_hat, cfg: HBTConfig, starts=None):
    """Conventional pipeline: normalize by the singles-rate flat level and
    LM-fit the three-level model with multiple starts (best practice);
    returns (g2_0_hat, ok_flag)."""
    flat = (0.5 * r_hat) ** 2 * (cfg.bin_width * 1e-9) * T_s
    if flat <= 0 or hist.sum() < 5:
        return 1.0, False
    y = hist / max(flat, 1e-12)
    tau = cfg.bin_centers
    sd = np.sqrt(np.maximum(hist, 1)) / flat

    def model(t, d, t1, a, t2, c0):
        return c0 * (1.0 - d * np.exp(-np.abs(t) / t1)
                     + a * np.exp(-np.abs(t) / t2))

    if starts is None:
        starts = [(0.7, 8.0, 0.1), (0.7, 15.0, 0.6),
                  (0.7, 25.0, 0.1), (0.3, 15.0, 0.6)]
    best = None
    c0g = max(np.median(y), 0.1)
    bounds = ([0.0, 0.3, 0.0, 50.0, 0.01], [1.0, 80.0, 3.0, 800.0, 10.0])
    for dg, t1g, ag in starts:
        try:
            popt, _ = curve_fit(model, tau, y, p0=(dg, t1g, ag, 250.0, c0g),
                                sigma=sd, bounds=bounds, maxfev=3000)
            r = float(np.sum(((model(tau, *popt) - y) / sd) ** 2))
            if best is None or r < best[0]:
                best = (r, popt)
        except Exception:
            continue
    if best is None:
        return 1.0, False
    d, t1, a, t2, c0 = best[1]
    return float(np.clip(1.0 - d + a, 0, 3)), True




def _fit_once(delay, counts, T_s, cfg, center=None):
    hist, center = rebin_real(np.asarray(delay, float),
                              np.asarray(counts, float), cfg, center=center)
    r_hat = robust_flat_rate(hist, cfg, T_s)
    g2_0, ok = fit_g2_histogram(hist, T_s, r_hat, cfg)
    return g2_0, ok, center, r_hat


def analyze_histogram(delay, counts, T_s, cfg: HBTConfig | None = None,
                      center=None, n_bootstrap: int = 200, ci: float = 0.68,
                      threshold: float = 0.5, seed: int = 0):
    """Estimate g2(0) from a measured CW HBT histogram, with a bootstrap CI.

    Parameters: delay in ns (any uniform grid; an electronic delay offset is
    fine, the dip is located automatically unless ``center`` is given),
    counts raw coincidences per bin, T_s the acquisition time in seconds,
    cfg the analysis grid (defaults to the package's 121-bin +-60.5 ns
    grid), n_bootstrap the number of Poisson resamples (0 disables the CI),
    ci the two-sided confidence level, threshold the single-emitter
    criterion on g2(0).

    Returns a dict with g2_0, ok (fit convergence), center (located dip
    position, ns), rate_cps (flat-level singles-rate estimate),
    single_emitter (g2_0 < threshold), and, when bootstrapping,
    g2_0_low / g2_0_high (CI bounds), g2_0_std, and
    single_emitter_confident (the whole CI on the same side of threshold).
    """
    if cfg is None:
        cfg = HBTConfig()
    counts = np.asarray(counts, float)
    if counts.ndim != 1 or len(counts) != len(delay):
        raise ValueError("delay and counts must be 1-D arrays of equal length")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")
    g2_0, ok, ctr, r_hat = _fit_once(delay, counts, T_s, cfg, center=center)
    out = dict(g2_0=g2_0, ok=ok, center=ctr, rate_cps=r_hat,
               single_emitter=bool(g2_0 < threshold))
    if n_bootstrap and n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(int(n_bootstrap)):
            resampled = rng.poisson(counts).astype(float)
            g2_b, ok_b, _, _ = _fit_once(delay, resampled, T_s, cfg, center=ctr)
            if ok_b:
                draws.append(g2_b)
        if draws:
            draws = np.array(draws)
            lo, hi = np.percentile(draws, [50 * (1 - ci), 50 * (1 + ci)])
            out.update(g2_0_low=float(lo), g2_0_high=float(hi),
                       g2_0_std=float(draws.std(ddof=1)) if len(draws) > 1 else np.nan,
                       n_bootstrap_ok=int(len(draws)),
                       single_emitter_confident=bool(hi < threshold or lo >= threshold))
    return out


def analyze_pulsed(delay, counts, t_rep: float = T_REP_NS,
                   n_bootstrap: int = 200, ci: float = 0.68,
                   threshold: float = 0.5, seed: int = 0):
    """Estimate g2(0) from a measured pulsed HBT histogram (peak-area method).

    The comb phase and the suppressed peak are located with
    :func:`sparq.pulsed.calibrate_comb`; g2(0) is the center-peak area over
    the mean side-peak area; the bootstrap resamples the raw counts as in
    :func:`analyze_histogram`.  ``t_rep`` is the laser repetition period in
    ns (default 12.5, i.e. 80 MHz).

    Returns a dict with g2_0, center, period, single_emitter and, when
    bootstrapping, g2_0_low / g2_0_high / g2_0_std /
    single_emitter_confident.
    """
    delay = np.asarray(delay, float)
    counts = np.asarray(counts, float)
    if counts.ndim != 1 or len(counts) != len(delay):
        raise ValueError("delay and counts must be 1-D arrays of equal length")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")
    center, phase, period = calibrate_comb(delay, counts, t_rep=t_rep)
    g2_0 = g2_peak_area(delay, counts, center, t_rep=t_rep)
    out = dict(g2_0=float(g2_0), center=float(center), period=float(period),
               single_emitter=bool(g2_0 < threshold))
    if n_bootstrap and n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(int(n_bootstrap)):
            resampled = rng.poisson(counts).astype(float)
            draws.append(g2_peak_area(delay, resampled, center, t_rep=t_rep))
        draws = np.array(draws, float)
        lo, hi = np.percentile(draws, [50 * (1 - ci), 50 * (1 + ci)])
        out.update(g2_0_low=float(lo), g2_0_high=float(hi),
                   g2_0_std=float(draws.std(ddof=1)),
                   single_emitter_confident=bool(hi < threshold or lo >= threshold))
    return out
