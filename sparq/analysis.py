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
histograms through :func:`analyze_pulsed`, which fits the whole comb
(:func:`sparq.pulsed.fit_pulsed`, the default from 0.11.0) or uses the
comb calibration and side-peak-area analysis of :mod:`sparq.pulsed`.
"""
from __future__ import annotations

import warnings

import numpy as np

from .datasets import locate_dip, rebin_real, robust_flat_rate
from .physics import HBTConfig
from .physics import _exp_conv_gauss
from .pulsed import T_REP_NS, calibrate_comb, fit_pulsed, g2_peak_area
from scipy.optimize import curve_fit, minimize
from scipy.stats import chi2


DEFAULT_T1_BOUNDS = (0.3, 80.0)     # ns; the NV-scale window of v0.1-v0.5
DEFAULT_T2_BOUNDS = (50.0, 800.0)   # ns


def _check_c0_prior(c0_prior):
    try:
        c0_m, c0_sd = float(c0_prior[0]), float(c0_prior[1])
    except (TypeError, IndexError, ValueError):
        raise ValueError("c0_prior must be a (mean, sd) pair") from None
    if not (np.isfinite(c0_m) and np.isfinite(c0_sd)
            and c0_m > 0.0 and c0_sd > 0.0):
        raise ValueError("c0_prior must be a (mean, sd) pair with "
                         "positive values")
    return c0_m, c0_sd


def _check_time_bounds(name, b):
    try:
        lo, hi = float(b[0]), float(b[1])
    except (TypeError, IndexError, ValueError):
        raise ValueError(f"{name} must be a (low, high) pair in ns") from None
    if not (np.isfinite(lo) and np.isfinite(hi) and 0.0 < lo < hi):
        raise ValueError(f"{name} must satisfy 0 < low < high; got {b}")
    return lo, hi


def _check_hist_inputs(hist, T_s, r_hat, cfg):
    """Refusals shared by the fits (new in 0.11.0): a histogram on the
    grid of cfg with finite, non-negative counts, and a positive finite
    acquisition time and singles rate. Up to 0.10.0 a wrong length or a
    non-positive time gave (1.0, False) or a broadcasting error, and NaN
    counts a TypeError."""
    h = np.asarray(hist)            # dtype kept: float32 re-binned data
    if h.dtype.kind not in "fiu":   # give the same fit as in 0.10.0
        raise ValueError("hist must hold numbers")
    if h.ndim != 1 or h.size != cfg.n_bins:
        raise ValueError(f"hist must have cfg.n_bins = {cfg.n_bins} bins; "
                         f"got shape {h.shape}")
    if not np.all(np.isfinite(h)) or np.any(h < 0):
        raise ValueError("hist must be finite and non-negative")
    for name, v in (("T_s", T_s), ("r_hat", r_hat)):
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a positive number") from None
        if not (np.isfinite(v) and v > 0.0):
            raise ValueError(f"{name} must be positive and finite; got {v}")
    return h


def _near(x, bound, rtol=1e-3):
    return abs(np.log(x) - np.log(bound)) < rtol


_A_MAX = 3.0                      # search limits of the shoulder amplitude
_C0_RANGE = (0.01, 10.0)          # and of the flat-level scale


def _railed(params, t1b, t2b, a_max=_A_MAX, c0_range=_C0_RANGE):
    """Names of fitted parameters that sit on a search bound (new in
    0.10.0). tau2 is only reported when the shoulder is not negligible
    (a > 0.02): with no shoulder, tau2 has no effect. The dip depth
    rho2 is bounded by physics (0 and 1), so it is never reported."""
    out = []
    for name, (lo, hi) in (("tau1", t1b), ("tau2", t2b)):
        if name == "tau2" and params["a"] <= 0.02:
            continue
        v = params[name]
        if _near(v, lo):
            out.append(name + "_low")
        elif _near(v, hi):
            out.append(name + "_high")
    if params["a"] >= a_max * (1 - 1e-3):
        out.append("a_high")
    if _near(params["c0"], c0_range[0]):
        out.append("c0_low")
    elif _near(params["c0"], c0_range[1]):
        out.append("c0_high")
    return out


def _widen(at_bound, b1, b2, a_max, c0r, lim1, lim2,
           which=("tau1", "tau2", "a", "c0")):
    """Tenfold wider limits on every side named in at_bound (within the
    hard limits lim1/lim2 for the lifetimes, 300 for a, 1e-4 and 1e3
    for c0). Returns the new limits and whether anything changed."""
    b1, b2, c0r = list(b1), list(b2), list(c0r)
    changed = False
    for name, b, lim in (("tau1", b1, lim1), ("tau2", b2, lim2)):
        if name not in which:
            continue
        if name + "_low" in at_bound and b[0] > lim[0]:
            b[0] = max(b[0] / 10.0, lim[0])
            changed = True
        if name + "_high" in at_bound and b[1] < lim[1]:
            b[1] = min(b[1] * 10.0, lim[1])
            changed = True
    if "a" in which and "a_high" in at_bound and a_max < 300.0:
        a_max = min(a_max * 10.0, 300.0)
        changed = True
    if "c0" in which and "c0_low" in at_bound and c0r[0] > 1e-4:
        c0r[0] = max(c0r[0] / 10.0, 1e-4)
        changed = True
    if "c0" in which and "c0_high" in at_bound and c0r[1] < 1e3:
        c0r[1] = min(c0r[1] * 10.0, 1e3)
        changed = True
    return tuple(b1), tuple(b2), a_max, tuple(c0r), changed


def _fit_auto(hist, T_s, r_hat, cfg, starts, t1_bounds, t2_bounds,
              max_widen=8, c0_prior=None, check_bounds=True, var=None):
    """Fit; then, if a parameter sits on a search bound:

    * with t1_bounds or t2_bounds "auto": widen the lifetime sides it
      sits on tenfold and refit, up to max_widen times (tau1 within
      1e-3 ns and 10 tau_max, tau2 within 1e-3 ns and 100 tau_max);
    * then, with check_bounds, if any parameter still sits on a bound
      (lifetime, shoulder amplitude a, flat-level scale c0): refit once
      with those limits ten times wider and record that fit's g2(0) as
      details["g2_0_widened"], so the caller can see whether the bound
      matters. (The limits of a and c0 are not widened for good: along
      them the fit can run off into the flat-level/shoulder trade.)
    Returns (g2_0, ok, details)."""
    auto1 = isinstance(t1_bounds, str)
    auto2 = isinstance(t2_bounds, str)
    for flag, name in ((auto1, t1_bounds), (auto2, t2_bounds)):
        if flag and name != "auto":
            raise ValueError("bounds must be a (low, high) pair or 'auto'")
    b1 = tuple(DEFAULT_T1_BOUNDS if auto1 else _check_time_bounds(
        "t1_bounds", t1_bounds))
    b2 = tuple(DEFAULT_T2_BOUNDS if auto2 else _check_time_bounds(
        "t2_bounds", t2_bounds))
    lim1 = (1e-3, 10.0 * cfg.tau_max)
    lim2 = (1e-3, 100.0 * cfg.tau_max)
    if auto1:
        b1 = (max(b1[0], lim1[0]), min(b1[1], lim1[1]))
    if auto2:
        b2 = (max(b2[0], lim2[0]), min(b2[1], lim2[1]))
    a_max, c0r = _A_MAX, _C0_RANGE
    # a fixed window goes to the fit exactly as given (so the starts,
    # which depend on whether it is the default window, are unchanged)
    g2, ok, det = _fit_core(hist, T_s, r_hat, cfg, starts,
                            b1 if auto1 else t1_bounds,
                            b2 if auto2 else t2_bounds,
                            c0_prior=c0_prior, var=var)
    if not (ok and det["at_bound"]):
        return g2, ok, det
    if auto1 or auto2:
        which = tuple(n for n, f in (("tau1", auto1), ("tau2", auto2)) if f)
        for _ in range(max_widen):
            b1, b2, _, _, changed = _widen(det["at_bound"], b1, b2, a_max,
                                           c0r, lim1, lim2, which)
            if not changed:
                break
            g2n, okn, detn = _fit_core(hist, T_s, r_hat, cfg, None, b1, b2,
                                       c0_prior=c0_prior, var=var)
            if not okn:
                break
            g2, ok, det = g2n, okn, detn
            if not det["at_bound"]:
                return g2, ok, det
    if check_bounds and det["at_bound"]:
        w1, w2, wa, wc, changed = _widen(det["at_bound"], det["t1_bounds"],
                                         det["t2_bounds"], a_max, c0r,
                                         (1e-6, np.inf), (1e-6, np.inf))
        if changed:
            g2w, okw, _ = _fit_core(hist, T_s, r_hat, cfg, None, w1, w2,
                                    c0_prior=c0_prior, a_max=wa,
                                    c0_range=wc, var=var)
            if okw:
                det["g2_0_widened"] = float(g2w)
    return g2, ok, det


BOUND_TOL = 0.01    # g2(0) change that makes a search bound "matter"


def fit_g2_histogram(hist, T_s, r_hat, cfg: HBTConfig, starts=None,
                     t1_bounds=DEFAULT_T1_BOUNDS,
                     t2_bounds=DEFAULT_T2_BOUNDS, return_details=False,
                     c0_prior=None, check_bounds=True, var=None):
    """Conventional pipeline: normalize by the singles-rate flat level and
    LM-fit the three-level model with multiple starts (best practice);
    returns (g2_0_hat, ok_flag), and with return_details=True also a
    dict (see below).

    t1_bounds / t2_bounds: the (low, high) windows, in ns, that the
    antibunching and bunching times are fitted within.  The defaults are
    the NV-scale window used since v0.1; emitters outside it (sub-0.3 ns
    quantum dots, second-scale shelving) need their own window.  A
    window that excludes the true value rails the fit: the fitted
    lifetime sits on the window's edge. From 0.10.0 this is detected
    (details["at_bound"]); the fit is then repeated once with the named
    sides widened tenfold (check_bounds=True), and details["g2_0_widened"]
    and details["bound_matters"] (the two g2(0) differ by more than
    BOUND_TOL = 0.01) say whether the bound changes the answer. The
    same check covers the limits of the shoulder amplitude a (3) and
    of the flat-level scale c0 (0.01 to 10). Either window may also be
    "auto": the fit then starts from the default window and keeps
    widening the lifetime sides it sits on (see `_fit_auto`).

    c0_prior: optional (mean, sd) Gaussian constraint on the flat-level
    scale c0 (new in 0.10.0; as in `profile_likelihood_ci`). Without it
    a slow, strong shoulder can trade against a lower flat level, and
    the histogram alone cannot tell them apart; with r_hat from the
    detectors' measured singles rates, c0_prior=(1.0, sd) breaks the
    tie. It assumes no bunching slower than the window other than the
    fitted shoulder (slow blinking raises the level inside the window).

    var: optional per-bin variance of hist, used for the error bars
    (default: the counts, i.e. Poisson; `rebin_real` gives it for a
    split histogram, whose bins vary less than Poisson counts).

    details: params (rho2, tau1, a, tau2, c0 of the best fit), chi2
    (including the prior term, if any) and dof of that fit, the
    t1_bounds / t2_bounds finally used, at_bound (names such as
    "tau1_low", "tau2_high", "a_high", "c0_low"; empty when no
    parameter sits on a bound), bound_matters, and, when at_bound is
    not empty, g2_0_widened. None when the fit failed.
    """
    hist = _check_hist_inputs(hist, T_s, r_hat, cfg)
    g2, ok, det = _fit_auto(hist, T_s, r_hat, cfg, starts, t1_bounds,
                            t2_bounds, c0_prior=c0_prior,
                            check_bounds=check_bounds, var=var)
    if det is not None:
        det["bound_matters"] = bool(
            det["at_bound"] and (
                "g2_0_widened" not in det
                or abs(det["g2_0_widened"] - g2) > BOUND_TOL))
    if return_details:
        return g2, ok, det
    return g2, ok


def _fit_core(hist, T_s, r_hat, cfg: HBTConfig, starts=None,
              t1_bounds=DEFAULT_T1_BOUNDS, t2_bounds=DEFAULT_T2_BOUNDS,
              c0_prior=None, a_max=_A_MAX, c0_range=_C0_RANGE, var=None):
    """One bounded multi-start fit; returns (g2_0, ok, details).

    c0_prior=(mean, sd): a Gaussian constraint on the flat-level scale
    c0, added to the least-squares objective as one extra residual
    (c0 - mean)/sd."""
    t1_lo, t1_hi = _check_time_bounds("t1_bounds", t1_bounds)
    t2_lo, t2_hi = _check_time_bounds("t2_bounds", t2_bounds)
    flat = (0.5 * r_hat) ** 2 * (cfg.bin_width * 1e-9) * T_s
    if flat <= 0 or hist.sum() < 5:
        return 1.0, False, None
    y = hist / max(flat, 1e-12)
    tau = cfg.bin_centers
    sd = np.sqrt(np.maximum(hist if var is None else var, 1)) / flat

    # the physical single-site form: dip depth rho2 = rho^2 in [0, 1]
    # multiplies BOTH exponentials (g2_measured's own parameterization),
    # so g2(0) = 1 - rho2 and the (depth, shoulder) pair is identifiable
    # -- unlike the naive 1 - d e1 + a e2 form, whose (d, a) ridge lets
    # a long-t2 shoulder trade against the constant level.
    # IRF-convolved exponentials (closed form; sigma_pair = sqrt(2)
    # sigma_irf for two detectors) -- with the instrument response in
    # the model, the estimate is the IRF-free g2(0), not the softened
    # dip the raw histogram shows.
    s_pair = np.sqrt(2.0) * float(cfg.sigma_irf)

    nodes, wts = cfg.bin_nodes()          # bin average (cfg.bin_average)
    anodes = np.abs(nodes)

    def model(t, rho2, t1, a, t2, c0):
        # t (the bin centers) only tells curve_fit the data length; the
        # model is evaluated at the bin nodes and averaged
        dip = ((1.0 + a) * _exp_conv_gauss(anodes, t1, s_pair)
               - a * _exp_conv_gauss(anodes, t2, s_pair)) @ wts
        return c0 * (1.0 - rho2 * dip)

    default_window = (t1_bounds == DEFAULT_T1_BOUNDS
                      and t2_bounds == DEFAULT_T2_BOUNDS)
    if starts is None:
        if default_window:
            # the exact start set of v0.1-v0.5 (bitwise regression anchor)
            starts = [(0.7, 8.0, 0.1), (0.7, 15.0, 0.6),
                      (0.7, 25.0, 0.1), (0.3, 15.0, 0.6)]
        else:
            # spread the starts across the user's window (log-spaced)
            tA, tB, tC = np.exp(np.linspace(np.log(t1_lo), np.log(t1_hi),
                                            5))[1:4]
            starts = [(0.7, tA, 0.1), (0.7, tB, 0.6),
                      (0.7, tC, 0.1), (0.3, tB, 0.6)]
    t2_start = 250.0 if default_window else float(np.sqrt(t2_lo * t2_hi))
    best = None
    c0g = max(np.median(y), 0.1)
    bounds = ([0.0, t1_lo, 0.0, t2_lo, c0_range[0]],
              [1.0, t1_hi, a_max, t2_hi, c0_range[1]])
    fit_model, xdata, ydata, sdata = model, tau, y, sd
    if c0_prior is not None:
        c0_m, c0_sd = _check_c0_prior(c0_prior)
        c0g = float(np.clip(c0_m, *c0_range))

        def fit_model(t, rho2, t1, a, t2, c0):
            return np.append(model(None, rho2, t1, a, t2, c0), c0)
        xdata = np.arange(y.size + 1)
        ydata = np.append(y, c0_m)
        sdata = np.append(sd, c0_sd)
    for dg, t1g, ag in starts:
        t1g = float(np.clip(t1g, t1_lo, t1_hi))
        rg = float(np.clip(dg, 0.0, 1.0))          # rho2 start
        try:
            popt, _ = curve_fit(fit_model, xdata, ydata,
                                p0=(rg, t1g, ag, t2_start, c0g),
                                sigma=sdata, bounds=bounds, maxfev=3000)
            r = float(np.sum(((fit_model(xdata, *popt) - ydata)
                              / sdata) ** 2))
            if best is None or r < best[0]:
                best = (r, popt)
        except (RuntimeError, ValueError, FloatingPointError,
                np.linalg.LinAlgError):
            continue
    if best is None:
        return 1.0, False, None
    rho2, t1, a, t2, c0 = best[1]
    params = dict(rho2=float(rho2), tau1=float(t1), a=float(a),
                  tau2=float(t2), c0=float(c0))
    details = dict(params=params, chi2=float(best[0]),
                   dof=int(len(y) - 5 + (c0_prior is not None)),
                   t1_bounds=(t1_lo, t1_hi), t2_bounds=(t2_lo, t2_hi),
                   at_bound=_railed(params, (t1_lo, t1_hi),
                                    (t2_lo, t2_hi), a_max, c0_range))
    return float(np.clip(1.0 - rho2, 0.0, 1.0)), True, details




def _fit_once(delay, counts, T_s, cfg, center=None,
              t1_bounds=DEFAULT_T1_BOUNDS, t2_bounds=DEFAULT_T2_BOUNDS,
              rebin="split", r_singles=None, c0_prior=None,
              check_bounds=True):
    hist, center, var = rebin_real(np.asarray(delay, float),
                                   np.asarray(counts, float), cfg,
                                   center=center, method=rebin,
                                   return_variance=True)
    if rebin == "whole":
        var = None             # Poisson counts: the 0.9.1 error bars
    r_hat = robust_flat_rate(hist, cfg, T_s) if r_singles is None \
        else r_singles
    g2_0, ok, det = fit_g2_histogram(hist, T_s, r_hat, cfg,
                                     t1_bounds=t1_bounds,
                                     t2_bounds=t2_bounds,
                                     return_details=True,
                                     c0_prior=c0_prior,
                                     check_bounds=check_bounds, var=var)
    return g2_0, ok, center, r_hat, det


def analyze_histogram(delay, counts, T_s, cfg: HBTConfig | None = None,
                      center=None, n_bootstrap: int = 200, ci: float = 0.68,
                      threshold: float = 0.5, seed: int = 0,
                      t1_bounds=DEFAULT_T1_BOUNDS,
                      t2_bounds=DEFAULT_T2_BOUNDS, rebin: str = "split",
                      singles_cps=None, singles_rel_sd: float = 0.01):
    """Estimate g2(0) from a measured CW HBT histogram, with a bootstrap CI.

    Parameters: delay in ns (bin centers on any grid no coarser than the
    analysis grid; an electronic delay offset is fine, the dip is
    located automatically unless ``center`` is given), counts raw
    coincidences per bin, T_s the acquisition time in seconds, cfg the
    analysis grid (defaults to the package's 121-bin +-60.5 ns grid),
    n_bootstrap the number of Poisson resamples (0 disables the CI), ci
    the two-sided confidence level, threshold the single-emitter
    criterion on g2(0).  t1_bounds / t2_bounds set the lifetime windows
    (ns) the fit searches -- the defaults are NV-scale; set them from
    your emitter's known timescales, or pass "auto" to have the window
    widened while the fit sits on its edge (see `fit_g2_histogram`).
    rebin: "split" (default from 0.10.0) or "whole", see `rebin_real`.
    singles_cps (new in 0.10.0): the two detectors' measured count
    rates (r_a, r_b) in counts/s, or their sum as one number (taken as
    split evenly). They set the flat level r_a r_b w T directly, and
    the fit is told that level to a relative `singles_rel_sd` (default
    1 %): the relative uncertainty of the product r_a r_b, about
    sqrt(2) times that of each rate when both are equally uncertain. Without them the
    flat level is a free fit parameter, and a slow shoulder can trade
    against it: the histogram alone then pins g2(0) only loosely (see
    the README). This assumes no bunching slower than the window
    except the fitted shoulder; slow blinking breaks it.

    Returns a dict with g2_0, ok (fit convergence), center (located dip
    position, ns), rate_cps (flat-level singles-rate estimate),
    single_emitter (g2_0 < threshold), fit (the fitted parameters),
    fit_p_value (new in 0.10.0: the chi-square tail probability of the
    fit's weighted residuals, with each re-binned bin's own Poisson
    variance as its error; a small value says the three-level model
    does not describe the dip -- try `sparq.models.fit_model` with
    another model. It is approximate: it ignores that neighbouring
    re-binned bins share input bins, and comes out somewhat too large;
    in 300 simulated runs 2 to 3 % fell below 0.05 instead of 5 %),
    dip_ambiguous (new in 0.10.0: True when the automatically located
    dip does not stand out clearly -- see `sparq.locate_dip`; a
    RuntimeWarning is then issued; None when `center` was given or
    rebin="whole"),
    at_bound (fitted parameters that sit on a search bound),
    g2_0_widened (the g2(0) of a refit with those limits ten times
    wider) and bound_matters (they differ by more than 0.01; a
    RuntimeWarning is then issued, because g2_0 is set by where the
    search stops, not by the data), t1_bounds / t2_bounds (the windows
    used), and, when
    bootstrapping, g2_0_low / g2_0_high (CI bounds), g2_0_std,
    n_bootstrap_ok and single_emitter_confident (the whole CI on the
    same side of threshold). The bootstrap refits reuse the windows of
    the main fit and, from 0.10.0, locate the dip anew on every copy
    (unless `center` is given), so the interval includes the
    uncertainty of the dip position; a copy whose dip cannot be placed
    inside the data counts as a failed refit.
    """
    if cfg is None:
        cfg = HBTConfig()
    counts = np.asarray(counts, float)
    delay = np.asarray(delay, float)
    if counts.ndim != 1 or delay.shape != counts.shape:
        raise ValueError("delay and counts must be 1-D arrays of equal length")
    if not (np.all(np.isfinite(delay)) and np.all(np.isfinite(counts))):
        raise ValueError("delay and counts must be finite")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")
    try:
        T_ok = np.isfinite(float(T_s)) and float(T_s) > 0.0
    except (TypeError, ValueError):
        T_ok = False
    if not T_ok:
        raise ValueError(f"T_s must be a positive acquisition time in "
                         f"seconds; got {T_s!r}")
    r_singles, c0_prior = None, None
    if singles_cps is not None:
        sc = np.atleast_1d(np.asarray(singles_cps, dtype=float))
        if sc.size == 1:
            sc = np.array([0.5 * sc[0], 0.5 * sc[0]])
        if sc.size != 2 or not np.all(np.isfinite(sc)) or np.any(sc <= 0):
            raise ValueError("singles_cps must be (r_a, r_b) or their sum, "
                             "positive counts per second")
        if not (np.isfinite(singles_rel_sd) and singles_rel_sd > 0):
            raise ValueError("singles_rel_sd must be positive")
        r_singles = 2.0 * float(np.sqrt(sc[0] * sc[1]))
        c0_prior = (1.0, float(singles_rel_sd))
    g2_0, ok, ctr, r_hat, det = _fit_once(delay, counts, T_s, cfg,
                                          center=center,
                                          t1_bounds=t1_bounds,
                                          t2_bounds=t2_bounds, rebin=rebin,
                                          r_singles=r_singles,
                                          c0_prior=c0_prior)
    at_bound = list(det["at_bound"]) if det else []
    matters = bool(det and det["bound_matters"])
    dip_ambiguous = None
    if center is None and rebin == "split":
        loc = locate_dip(delay, counts)
        dip_ambiguous = bool(loc["ambiguous"])
        if dip_ambiguous:
            warnings.warn(
                "the dip does not stand out clearly from the noise "
                f"(significance {loc['z']:.1f} standard errors; another "
                f"place reaches {loc['z_other']})"
                if loc["z_other"] is not None else
                "the dip does not stand out clearly from the noise",
                RuntimeWarning, stacklevel=2)
            warnings.warn(
                "the automatic dip position may be wrong; pass `center` "
                "if you know where the dip is", RuntimeWarning,
                stacklevel=2)
    out = dict(g2_0=g2_0, ok=ok, center=ctr, rate_cps=r_hat,
               single_emitter=bool(g2_0 < threshold),
               fit=det["params"] if det else None, at_bound=at_bound,
               bound_matters=matters, dip_ambiguous=dip_ambiguous,
               fit_p_value=(float(chi2.sf(det["chi2"], det["dof"]))
                            if det and det["dof"] > 0 else None),
               g2_0_widened=det.get("g2_0_widened") if det else None,
               t1_bounds=det["t1_bounds"] if det else None,
               t2_bounds=det["t2_bounds"] if det else None)
    if matters:
        wid = det.get("g2_0_widened")
        more = (f" (with those limits ten times wider: {wid:.3f})"
                if wid is not None else "")
        warnings.warn(
            "the fit sits on a search bound (" + ", ".join(at_bound)
            + f") and g2_0 = {g2_0:.3f} depends on it{more}; the data "
            "do not pin g2(0) within the searched range: widen "
            "t1_bounds / t2_bounds, pass 'auto', or give singles_cps",
            RuntimeWarning, stacklevel=2)
    if n_bootstrap and n_bootstrap > 0:
        b1 = det["t1_bounds"] if det else t1_bounds
        b2 = det["t2_bounds"] if det else t2_bounds
        if det and not isinstance(t1_bounds, str):
            b1 = t1_bounds
        if det and not isinstance(t2_bounds, str):
            b2 = t2_bounds
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(int(n_bootstrap)):
            resampled = rng.poisson(counts).astype(float)
            # the dip is located again on every copy (unless `center`
            # was given), so the interval includes that uncertainty
            try:
                g2_b, ok_b, _, _, _ = _fit_once(
                    delay, resampled, T_s, cfg, center=center,
                    t1_bounds=b1, t2_bounds=b2, rebin=rebin,
                    r_singles=r_singles, c0_prior=c0_prior,
                    check_bounds=False)
            except ValueError:          # dip not placeable in this copy
                ok_b = False
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
                   threshold: float = 0.5, seed: int = 0,
                   method: str = "fit", envelope: bool = False,
                   T_s=None, singles_cps=None, sigma_pair=None):
    """Estimate g2(0) from a measured pulsed HBT histogram (a comb of
    peaks, one per laser period).

    method="fit" (the default from 0.11.0): `sparq.pulsed.fit_pulsed`,
    a Poisson maximum-likelihood fit of the whole comb in which the
    peaks may overlap; the interval is its profile-likelihood interval
    at level `ci` (computed when n_bootstrap > 0; `seed` is not used).
    `envelope`, `T_s`, `singles_cps` and `sigma_pair` are passed to it.

    method="area" (the only method up to 0.10.0): the comb phase and
    the suppressed peak are located with `calibrate_comb`; g2(0) is the
    center-peak area over the mean side-peak area, both with the floor
    between the peaks subtracted (`g2_peak_area`); the bootstrap
    resamples the raw counts as in `analyze_histogram`. It is biased
    low when the peaks' tails reach the space between them: on
    noise-free data at 80 MHz with true g2(0) = 0.12 it gives 0.104 for
    a 1 ns lifetime and 0 for 2 ns, where the fit gives 0.120.

    ``t_rep`` is the laser repetition period in ns (default 12.5, i.e.
    80 MHz).

    Returns a dict with g2_0, center, period, single_emitter and, when
    n_bootstrap > 0, g2_0_low / g2_0_high / g2_0_std /
    single_emitter_confident (the whole interval on one side of
    threshold). For method="fit" also ok, fit (the fitted parameters),
    fit_p_value and at_bound (see `fit_pulsed`); g2_0_std is then NaN
    (the profile interval has no standard deviation), and g2_0_high is
    NaN when the interval does not close (then
    single_emitter_confident is False unless g2_0_low >= threshold).
    """
    delay = np.asarray(delay, float)
    counts = np.asarray(counts, float)
    if counts.ndim != 1 or len(counts) != len(delay):
        raise ValueError("delay and counts must be 1-D arrays of equal length")
    if not (np.all(np.isfinite(delay)) and np.all(np.isfinite(counts))):
        raise ValueError("delay and counts must be finite")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")
    if method == "fit":
        want = bool(n_bootstrap and n_bootstrap > 0)
        fit = fit_pulsed(delay, counts, t_rep=t_rep, sigma_pair=sigma_pair,
                         envelope=envelope, T_s=T_s,
                         singles_cps=singles_cps, level=ci, interval=want)
        g2_0 = float(fit["g2_0"])
        out = dict(g2_0=g2_0,
                   center=float(fit["params"]["center"]) if fit["ok"]
                   else np.nan,
                   period=float(t_rep), single_emitter=bool(g2_0 < threshold),
                   ok=bool(fit["ok"]), fit=fit.get("params"),
                   fit_p_value=fit.get("p_value"),
                   at_bound=fit.get("at_bound", []))
        if want and fit["ok"]:
            lo, hi = float(fit["lo"]), float(fit["hi"])
            out.update(g2_0_low=lo, g2_0_high=hi, g2_0_std=np.nan,
                       single_emitter_confident=bool(
                           (np.isfinite(hi) and hi < threshold)
                           or lo >= threshold))
        return out
    if method != "area":
        raise ValueError("method must be 'fit' or 'area'")
    if envelope or T_s is not None or singles_cps is not None \
            or sigma_pair is not None:
        raise ValueError("envelope, T_s, singles_cps and sigma_pair are "
                         "options of method='fit'")
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


def _poisson_nll(hist, mu):
    mu = np.maximum(mu, 1e-12)
    return float(np.sum(mu - hist * np.log(mu)))


def profile_likelihood_ci(hist, T_s: float, r_hat: float,
                          cfg: HBTConfig | None = None,
                          level: float = 0.95, n_grid: int = 41,
                          g2_max: float = 1.5,
                          t1_bounds=DEFAULT_T1_BOUNDS,
                          t2_bounds=DEFAULT_T2_BOUNDS,
                          c0_prior=None) -> dict:
    """Profile-likelihood confidence interval for g2(0) (Wilks/likelihood
    ratio), from the exact Poisson likelihood of the histogram.

    For each value of g2(0) on a grid the Poisson negative log-likelihood
    of the three-level model is minimized over all nuisance parameters
    (lifetimes, bunching amplitude, normalization) with the dip depth
    FIXED at rho2 = 1 - g2 -- the physical parameterization, in which
    g2(0) = 1 - rho2 is identifiable (the naive 1 - d e1 + a e2 form
    has a (d, a) ridge when the shoulder is slow, and its intervals
    inherit it); the interval is the set where the profile deviance
    2 [NLL(g2) - NLL_min] stays below the chi-square(1) quantile of
    ``level`` (Wilks' theorem).  Unlike the parametric bootstrap this
    needs no resampling, and unlike the fit's linearized errors it remains
    honest for asymmetric likelihoods at low counts.

    t1_bounds / t2_bounds: the lifetime windows (ns) the nuisance
    parameters are profiled within; defaults are the NV-scale window.

    c0_prior: optional (mean, sd) Gaussian constraint on the flat-level
    normalization c0 (the ratio of the true uncorrelated coincidence
    level to the one implied by ``r_hat``).  With c0 fully free the
    interval can be honestly WIDE: a larger bunching amplitude with a
    slower shoulder trades almost exactly against a lower flat level,
    and the histogram alone cannot break the tie (this near-degeneracy
    is real; a more restrictive model would only hide it).  An
    experiment breaks it with information the histogram does not
    carry: the singles rates measured directly on the counters.  Pass
    ``r_hat`` from those measured rates and ``c0_prior=(1.0, sd)``
    with sd their relative uncertainty, and the interval tightens to
    what the data genuinely support.

    t1_bounds / t2_bounds may also be "auto" (new in 0.10.0): the
    windows are then those an automatically widened fit ends with (see
    `fit_g2_histogram`).

    Returns a dict with g2_hat (profile minimum), lo, hi (interval
    bounds, NaN when unbounded on that side within the grid), level,
    the (grid, deviance) profile for plotting, the windows used, and
    at_bound: nuisance parameters that sit on a search bound at the
    profile minimum (new in 0.10.0; a RuntimeWarning is issued when it
    is not empty).

    Accuracy (corrected in 0.12.0): the profile is minimized over
    log tau1, log tau2, a and log c0, with the objective written as
    half the Poisson deviance; the grid locates the minimum and the
    two crossings, and g2_hat, lo and hi are then refined between grid
    points (to about 1e-4). Up to 0.11.0 tau2 hardly moved from its
    start value, the deviance was measured from the smallest grid value
    and the ends were interpolated linearly, so the ends could be far
    from the chi-square quantile: for README example 3, [0.084, 0.126]
    with c0_prior=(1.0, 0.01) and [0.067, 0.182] without, where the
    true profile gives [0.078, 0.124] and [0.064, 0.295]. The tests
    check the ends against an independent Nelder-Mead profile.
    """
    if cfg is None:
        cfg = HBTConfig()
    hist = np.asarray(_check_hist_inputs(hist, T_s, r_hat, cfg), float)
    flat = (0.5 * r_hat) ** 2 * (cfg.bin_width * 1e-9) * T_s
    if flat <= 0 or hist.sum() < 5:
        raise ValueError("histogram carries too little signal to profile")
    nodes, wts = cfg.bin_nodes()          # bin average (cfg.bin_average)
    tau = np.abs(nodes)

    if c0_prior is not None:
        c0_m, c0_sd = _check_c0_prior(c0_prior)
    if not (0.0 < float(level) < 1.0):
        raise ValueError(f"level must lie in (0, 1); got {level}")
    if int(n_grid) < 3:
        raise ValueError("n_grid must be at least 3")

    s_pair = np.sqrt(2.0) * float(cfg.sigma_irf)
    # sum(h log h - h): subtracting it makes the objective half the
    # Poisson deviance (of order the number of bins) instead of a number
    # of order the total counts, so the optimizer's relative stopping
    # rule acts on the differences that matter
    pos = hist > 0
    nll_offset = float(np.sum(hist[pos] * np.log(hist[pos])) - hist.sum())

    def nll_free(theta, g2):
        """theta = (log tau1, log tau2, a, log c0) (from 0.12.0; up to
        0.11.0 the lifetimes and c0 were searched in natural units, in
        which the optimizer moved tau2 hardly at all from its start)."""
        t1, t2, a, c0 = np.exp(theta[0]), np.exp(theta[1]), theta[2], \
            np.exp(theta[3])
        rho2 = 1.0 - g2                      # fixed by the profiled g2
        mu = flat * c0 * (1.0 - rho2 * (((1.0 + a) * _exp_conv_gauss(tau, t1, s_pair)
                                         - a * _exp_conv_gauss(tau, t2, s_pair)) @ wts))
        nll = _poisson_nll(hist, mu) + nll_offset
        if c0_prior is not None:
            nll += 0.5 * ((c0 - c0_m) / c0_sd) ** 2
        return nll

    if isinstance(t1_bounds, str) or isinstance(t2_bounds, str):
        # "auto": take the windows an automatically widened fit ends with
        _, ok_a, det_a = _fit_auto(hist, T_s, r_hat, cfg, None, t1_bounds,
                                   t2_bounds)
        if not ok_a or det_a is None:
            raise ValueError("the fit that sets the 'auto' windows failed")
        t1_bounds, t2_bounds = det_a["t1_bounds"], det_a["t2_bounds"]
    t1_lo, t1_hi = _check_time_bounds("t1_bounds", t1_bounds)
    t2_lo, t2_hi = _check_time_bounds("t2_bounds", t2_bounds)
    t1_st = float(np.clip(15.0, t1_lo, t1_hi))
    t2_st = float(np.clip(250.0, t2_lo, t2_hi))
    lo_b = [t1_lo, t2_lo, 0.0, 0.01]
    hi_b = [t1_hi, t2_hi, 3.0, 10.0]
    # the single-site family has measured g2(0) = 1 - rho2 in [0, 1]
    grid = np.linspace(0.0, min(float(g2_max), 1.0), int(n_grid))
    prof = np.empty_like(grid)
    xs = np.empty((grid.size, 4))
    warm = None
    c0g = max(float(np.median(hist)) / max(flat, 1e-12), 0.1)
    t1_gm = float(np.sqrt(t1_lo * t1_hi))
    t2_gm = float(np.sqrt(t2_lo * t2_hi))
    log_lo = np.array([np.log(t1_lo), np.log(t2_lo), 0.0, np.log(0.01)])
    log_hi = np.array([np.log(t1_hi), np.log(t2_hi), 3.0, np.log(10.0)])

    def _to_log(x):
        return np.array([np.log(x[0]), np.log(x[1]), x[2], np.log(x[3])])

    def _from_log(y):
        return np.array([np.exp(y[0]), np.exp(y[1]), y[2], np.exp(y[3])])

    def _profile(g2, warm_x):
        """Minimum of the NLL over the nuisance parameters at fixed g2,
        from the warm start and the two fixed starts; returns the
        minimum and its (tau1, tau2, a, c0)."""
        starts = [np.array([t1_st, t2_st, 0.3, c0g]),
                  np.array([t1_gm, t2_gm, 0.3, c0g])]
        if warm_x is not None:
            starts.insert(0, warm_x)
        best = None
        for x0 in starts:
            y0 = np.clip(_to_log(np.clip(x0, lo_b, hi_b)), log_lo, log_hi)
            res = minimize(nll_free, y0, args=(g2,), method="L-BFGS-B",
                           bounds=list(zip(log_lo, log_hi)))
            if best is None or res.fun < best.fun:
                best = res
        return float(best.fun), _from_log(best.x)

    for i, g2 in enumerate(grid):
        prof[i], xs[i] = _profile(g2, warm)
        warm = xs[i]
    i0 = int(np.argmin(prof))
    crit = float(chi2.ppf(level, 1))
    # From 0.12.0 the minimum and the two ends are refined between the
    # grid points. Up to 0.11.0 the deviance was taken relative to the
    # smallest GRID value (grid step 0.025 by default), the ends were
    # interpolated linearly between grid points, and g2_hat was a grid
    # point.
    from scipy.optimize import brentq, minimize_scalar
    pts = {float(g): (prof[i], xs[i]) for i, g in enumerate(grid)}

    def _p_min(g):
        near = min(pts, key=lambda q: abs(q - g))
        v, x = _profile(g, pts[near][1])
        pts[float(g)] = (v, x)
        return v

    minimize_scalar(_p_min, bounds=(float(grid[max(i0 - 1, 0)]),
                                    float(grid[min(i0 + 1, grid.size - 1)])),
                    method="bounded", options=dict(xatol=1e-4))
    gs = np.array(sorted(pts))
    pv = np.array([pts[g][0] for g in gs])
    j0 = int(np.argmin(pv))
    p_min = float(pv[j0])
    g_hat = float(gs[j0])
    x_hat = pts[g_hat][1]
    dev = 2.0 * (prof - p_min)

    def _edge(lo_side: bool):
        dv = 2.0 * (pv - p_min)
        idx = range(j0, -1, -1) if lo_side else range(j0, gs.size)
        prev = j0
        for j in idx:
            if dv[j] > crit:
                # the crossing lies between gs[prev] (inside) and gs[j]
                warm_x = pts[float(gs[prev])][1]

                def excess(g):
                    g = float(g)
                    v = pts[g][0] if g in pts else _profile(g, warm_x)[0]
                    return 2.0 * (v - p_min) - crit

                a_e, b_e = float(gs[prev]), float(gs[j])
                return float(brentq(excess, min(a_e, b_e), max(a_e, b_e),
                                    xtol=1e-4))
            prev = j
        return float("nan")                # unbounded within the grid

    x0 = x_hat
    at_bound = _railed(dict(tau1=x0[0], tau2=x0[1], a=x0[2], c0=x0[3]),
                       (t1_lo, t1_hi), (t2_lo, t2_hi))
    if at_bound:
        warnings.warn(
            "at the best g2(0) the profiled parameters sit on a search "
            "bound (" + ", ".join(at_bound) + "); the interval may depend "
            "on where the range ends: widen t1_bounds / t2_bounds, pass "
            "'auto', or pin the flat level with c0_prior",
            RuntimeWarning, stacklevel=2)
    return dict(g2_hat=g_hat, lo=_edge(True), hi=_edge(False),
                level=float(level), grid=grid, deviance=dev,
                at_bound=at_bound, t1_bounds=(t1_lo, t1_hi),
                t2_bounds=(t2_lo, t2_hi))


# ----------------------------------------------------------------------
# Exact closed-form corrections (v0.4)
# ----------------------------------------------------------------------

def signal_fraction(signal_rate: float, background_rate: float) -> float:
    """rho = S / (S + B): the signal fraction entering the background
    correction. Rates in any common unit; both must be finite and
    non-negative and their sum positive. For rho with an interval from
    measured counts, see `signal_fraction_from_counts`."""
    S = float(signal_rate)
    B = float(background_rate)
    if not (np.isfinite(S) and np.isfinite(B)):
        raise ValueError("rates must be finite")
    if S < 0.0 or B < 0.0 or S + B <= 0.0:
        raise ValueError("rates must be non-negative with S + B > 0")
    return S / (S + B)


def signal_fraction_from_counts(n_on: float, t_on: float, n_off: float,
                                t_off: float, level: float = 0.95) -> dict:
    """The signal fraction rho = S/(S+B) from measured counts, with an
    exact confidence interval (new in 0.12.0).

    n_on clicks in t_on seconds on the emitter (signal plus
    background), n_off clicks in t_off seconds on a nearby spot with no
    emitter (background only), counted on the same detectors. Then
    rho = 1 - q with q = B/(S+B), estimated as (n_off/t_off) /
    (n_on/t_on). This assumes the background at the emitter equals the
    one beside it, and count rates low enough that dead time does not
    matter (or rates corrected with `deadtime_corrected_rate`).

    Interval: given n = n_on + n_off, n_off is binomial with success
    probability pi = q t_off / (q t_off + t_on), whatever the rates.
    The Clopper-Pearson interval for pi (Clopper and Pearson,
    Biometrika 26, 404 (1934)) covers it with probability at least
    `level`, and q, hence rho, is a monotone function of pi, so the
    rho interval keeps that guarantee. It is conservative (the tests
    measure a coverage at or above `level` by simulation).

    Refuses negative or non-integer counts, non-positive times, no
    counts on the emitter, and a background rate not below the rate on
    the emitter (rho would not be positive).

    Returns dict(rho, lo, hi, level).
    """
    from scipy.stats import beta as beta_dist
    counts = []
    for name, v in (("n_on", n_on), ("n_off", n_off)):
        v = float(v)
        if not (np.isfinite(v) and v >= 0 and v == np.floor(v)):
            raise ValueError(f"{name} must be a whole, non-negative count")
        counts.append(v)
    n1, n2 = counts
    for name, v in (("t_on", t_on), ("t_off", t_off)):
        if not (np.isfinite(float(v)) and float(v) > 0):
            raise ValueError(f"{name} must be a positive time (s)")
    t1, t2 = float(t_on), float(t_off)
    if not (0.0 < level < 1.0):
        raise ValueError("level must lie in (0, 1)")
    if n1 <= 0:
        raise ValueError("n_on must be positive: no counts on the emitter")
    q_hat = (n2 / t2) / (n1 / t1)
    if q_hat >= 1.0:
        raise ValueError(
            f"the background rate {n2 / t2:.4g} /s is not below the rate "
            f"on the emitter {n1 / t1:.4g} /s, so the signal fraction is "
            "not positive")
    n = n1 + n2
    a = 0.5 * (1.0 - level)
    p_lo = 0.0 if n2 == 0 else float(beta_dist.ppf(a, n2, n - n2 + 1))
    p_hi = 1.0 if n2 == n else float(beta_dist.ppf(1.0 - a, n2 + 1, n - n2))

    def q_of(p):
        return p * t1 / ((1.0 - p) * t2) if p < 1.0 else np.inf

    return dict(rho=1.0 - q_hat, lo=1.0 - q_of(p_hi),
                hi=1.0 - q_of(p_lo), level=float(level))


def background_corrected_g2(g2_meas: float, rho: float, ci=None,
                            rho_ci=None) -> dict:
    """Exact background correction of a measured g2(0).

    A Poissonian background at signal fraction rho = S/(S+B) maps the
    true correlation to the measured one as

        g2_meas = 1 + rho^2 (g2_true - 1)

    (Brouri, Beveratos, Poizat and Grangier, Opt. Lett. 25, 1294
    (2000)) -- exactly the forward map this package's own
    ``g2_zero(..., rho)`` applies. This function is its algebraic
    inverse,

        g2_true = 1 + (g2_meas - 1) / rho^2 ,

    so correcting the package's forward model recovers the rho = 1
    value to machine precision (asserted in the tests, not stated).
    The map is affine and increasing in g2_meas, so a confidence
    interval transforms endpoint-by-endpoint: pass ``ci=(lo, hi)`` to
    get the corrected interval.

    rho_ci (new in 0.12.0): an interval (lo, hi) for rho itself, for
    example from `signal_fraction_from_counts`. The corrected interval
    is then the smallest and largest corrected value over all pairs
    (g2 in ci, rho in rho_ci): the map rises with g2 and, for g2 below
    1, with rho, so these sit at corners of that box. If ci covers the
    measured g2 with probability 1 - a and rho_ci covers rho with
    probability 1 - b, the result covers the true g2 with probability
    at least 1 - a - b (Bonferroni): for a 95 % result use two 97.5 %
    intervals. Without ci, rho_ci alone gives the range from rho. The
    point value still uses `rho`, which must lie in rho_ci.

    rho must lie in (0, 1]; a corrected value below 0 is truncated to
    0 (a measured histogram can fluctuate below the physical floor)
    and the untruncated value is returned alongside.

    Returns dict(g2_corrected, g2_uncorrected_inverse, ci).
    """
    rho = float(rho)
    if not (0.0 < rho <= 1.0):
        raise ValueError("rho must lie in (0, 1]")
    g = float(g2_meas)
    if not np.isfinite(g):
        raise ValueError("g2_meas must be finite")

    def corr(gm, r):
        return 1.0 + (gm - 1.0) / r ** 2

    raw = corr(g, rho)
    out = dict(g2_corrected=max(raw, 0.0),
               g2_uncorrected_inverse=raw, ci=None)
    lo = hi = g
    if ci is not None:
        lo, hi = (float(ci[0]), float(ci[1]))
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError("ci must be finite")
        if lo > hi:
            raise ValueError("ci must be (lo, hi) with lo <= hi")
    if rho_ci is None:
        if ci is not None:
            out["ci"] = (max(corr(lo, rho), 0.0), max(corr(hi, rho), 0.0))
        return out
    r_lo, r_hi = (float(rho_ci[0]), float(rho_ci[1]))
    if np.isfinite(r_lo) and r_lo <= 0.0:
        raise ValueError(
            "rho_ci reaches down to 0: the background is not measured "
            "well enough to bound the correction (count longer on and "
            "beside the emitter)")
    if not (0.0 < r_lo <= rho <= r_hi <= 1.0):
        raise ValueError("rho_ci must be (lo, hi) with 0 < lo <= rho <= "
                         "hi <= 1")
    # corners: the smallest value at the lowest g2 with the rho that
    # pulls it down most, the largest at the highest g2 likewise
    low = corr(lo, r_lo) if lo < 1.0 else corr(lo, r_hi)
    high = corr(hi, r_hi) if hi < 1.0 else corr(hi, r_lo)
    out["ci"] = (max(low, 0.0), max(high, 0.0))
    return out


def deadtime_corrected_rate(r_meas_cps: float, dead_time_ns: float) -> float:
    """Exact non-paralyzable dead-time correction of a count rate.

    A detector that goes blind for tau_d after each accepted count maps
    the true rate to the measured one as r_meas = r / (1 + r tau_d)
    (non-paralyzable model; the package's Monte-Carlo detector chain
    implements exactly this greedy dead-time pass). The inverse is

        r = r_meas / (1 - r_meas tau_d),

    exact, and refused when r_meas tau_d >= 1 (a measured rate at or
    above the saturation rate 1/tau_d is inconsistent with the model
    rather than something to extrapolate). Anchors in the tests: the
    round trip is exact to machine precision, and the forward formula
    matches the Monte-Carlo detector chain's throughput on a Poisson
    stream within statistical error.
    """
    r = float(r_meas_cps)
    td = float(dead_time_ns) * 1e-9
    if not (np.isfinite(r) and np.isfinite(td)):
        raise ValueError("rate and dead time must be finite")
    if r < 0.0 or td < 0.0:
        raise ValueError("rate and dead time must be non-negative")
    x = r * td
    if x >= 1.0:
        raise ValueError(
            f"measured rate {r:.3g} cps is at or above the saturation "
            f"rate 1/tau_d = {1.0 / td:.3g} cps of the non-paralyzable "
            "model; the correction has no solution there")
    return r / (1.0 - x)
