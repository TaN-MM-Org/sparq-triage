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

import warnings

import numpy as np

from .datasets import locate_dip, rebin_real, robust_flat_rate
from .physics import HBTConfig
from .physics import _exp_conv_gauss
from .pulsed import T_REP_NS, calibrate_comb, g2_peak_area
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
    if counts.ndim != 1 or len(counts) != len(delay):
        raise ValueError("delay and counts must be 1-D arrays of equal length")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")
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


def _poisson_nll(hist, mu):
    mu = np.maximum(mu, 1e-12)
    return float(np.sum(mu - hist * np.log(mu)))


def profile_likelihood_ci(hist, T_s, r_hat, cfg: HBTConfig | None = None,
                          level: float = 0.95, n_grid: int = 41,
                          g2_max: float = 1.5,
                          t1_bounds=DEFAULT_T1_BOUNDS,
                          t2_bounds=DEFAULT_T2_BOUNDS,
                          c0_prior=None):
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
    """
    if cfg is None:
        cfg = HBTConfig()
    hist = np.asarray(hist, float)
    flat = (0.5 * r_hat) ** 2 * (cfg.bin_width * 1e-9) * T_s
    if flat <= 0 or hist.sum() < 5:
        raise ValueError("histogram carries too little signal to profile")
    nodes, wts = cfg.bin_nodes()          # bin average (cfg.bin_average)
    tau = np.abs(nodes)

    if c0_prior is not None:
        c0_m, c0_sd = float(c0_prior[0]), float(c0_prior[1])
        if not (np.isfinite(c0_m) and np.isfinite(c0_sd)
                and c0_m > 0.0 and c0_sd > 0.0):
            raise ValueError("c0_prior must be a (mean, sd) pair with "
                             "positive values")

    s_pair = np.sqrt(2.0) * float(cfg.sigma_irf)

    def nll_free(theta, g2):
        t1, t2, a, c0 = theta
        rho2 = 1.0 - g2                      # fixed by the profiled g2
        mu = flat * c0 * (1.0 - rho2 * (((1.0 + a) * _exp_conv_gauss(tau, t1, s_pair)
                                         - a * _exp_conv_gauss(tau, t2, s_pair)) @ wts))
        nll = _poisson_nll(hist, mu)
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
    for i, g2 in enumerate(grid):
        starts = [np.array([t1_st, t2_st, 0.3, c0g]),
                  np.array([t1_gm, t2_gm, 0.3, c0g])]
        if warm is not None:
            starts.insert(0, warm)
        best = None
        for x0 in starts:
            x0 = np.clip(x0, lo_b, hi_b)
            res = minimize(nll_free, x0, args=(g2,), method="L-BFGS-B",
                           bounds=list(zip(lo_b, hi_b)))
            if best is None or res.fun < best.fun:
                best = res
        prof[i] = best.fun
        xs[i] = best.x
        warm = best.x
    i0 = int(np.argmin(prof))
    dev = 2.0 * (prof - prof[i0])
    crit = float(chi2.ppf(level, 1))
    inside = dev <= crit

    def _edge(lo_side: bool):
        idx = range(i0, -1, -1) if lo_side else range(i0, len(grid))
        prev = i0
        for j in idx:
            if not inside[j]:
                # linear interpolation between prev (inside) and j (outside)
                x0, x1 = grid[prev], grid[j]
                y0, y1 = dev[prev], dev[j]
                return float(x0 + (crit - y0) * (x1 - x0) / max(y1 - y0, 1e-12))
            prev = j
        return float("nan")                # unbounded within the grid

    x0 = xs[i0]
    at_bound = _railed(dict(tau1=x0[0], tau2=x0[1], a=x0[2], c0=x0[3]),
                       (t1_lo, t1_hi), (t2_lo, t2_hi))
    if at_bound:
        warnings.warn(
            "at the best g2(0) the profiled parameters sit on a search "
            "bound (" + ", ".join(at_bound) + "); the interval may depend "
            "on where the range ends: widen t1_bounds / t2_bounds, pass "
            "'auto', or pin the flat level with c0_prior",
            RuntimeWarning, stacklevel=2)
    return dict(g2_hat=float(grid[i0]), lo=_edge(True), hi=_edge(False),
                level=float(level), grid=grid, deviance=dev,
                at_bound=at_bound, t1_bounds=(t1_lo, t1_hi),
                t2_bounds=(t2_lo, t2_hi))


# ----------------------------------------------------------------------
# Exact closed-form corrections (v0.4)
# ----------------------------------------------------------------------

def signal_fraction(signal_rate, background_rate):
    """rho = S / (S + B): the signal fraction entering the background
    correction. Rates in any common unit; both must be non-negative and
    their sum positive."""
    S = float(signal_rate)
    B = float(background_rate)
    if S < 0.0 or B < 0.0 or S + B <= 0.0:
        raise ValueError("rates must be non-negative with S + B > 0")
    return S / (S + B)


def background_corrected_g2(g2_meas, rho, ci=None):
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

    rho must lie in (0, 1]; a corrected value below 0 is truncated to
    0 (a measured histogram can fluctuate below the physical floor)
    and the untruncated value is returned alongside.

    Returns dict(g2_corrected, g2_uncorrected_inverse, ci).
    """
    rho = float(rho)
    if not (0.0 < rho <= 1.0):
        raise ValueError("rho must lie in (0, 1]")
    raw = 1.0 + (float(g2_meas) - 1.0) / rho ** 2
    out = dict(g2_corrected=max(raw, 0.0),
               g2_uncorrected_inverse=raw, ci=None)
    if ci is not None:
        lo, hi = (float(ci[0]), float(ci[1]))
        if lo > hi:
            raise ValueError("ci must be (lo, hi) with lo <= hi")
        out["ci"] = (max(1.0 + (lo - 1.0) / rho ** 2, 0.0),
                     max(1.0 + (hi - 1.0) / rho ** 2, 0.0))
    return out


def deadtime_corrected_rate(r_meas_cps, dead_time_ns):
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
    if r < 0.0 or td < 0.0:
        raise ValueError("rate and dead time must be non-negative")
    x = r * td
    if x >= 1.0:
        raise ValueError(
            f"measured rate {r:.3g} cps is at or above the saturation "
            f"rate 1/tau_d = {1.0 / td:.3g} cps of the non-paralyzable "
            "model; the correction has no solution there")
    return r / (1.0 - x)
