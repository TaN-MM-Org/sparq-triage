"""Saturation curve of an emitter: detected count rate against
excitation power (new in 0.13.0).

Every single-photon emitter is characterized by how its count rate
grows with the excitation power and levels off. Under continuous-wave
excitation the package's three-level rate model (`sparq.exact`) gives
this curve exactly. With the pump rate k_exc proportional to the power
P, its steady-state photon emission rate k_r p_e is

    R(P) = R_inf P / (P + P_sat),

    R_inf = k_r k_se / (k_se + k_es),
    k_sat = (k_r + k_es) k_se / (k_se + k_es)   (k_exc at P = P_sat),

(from p_e = 1 / (1 + (k_r + k_es)/k_exc + k_es/k_se), the steady state
of the rate equations; `saturation_from_rates` evaluates it and the
tests check it against the steady state of `sparq.exact.liouvillian`
to 1e-12). The detected rate is a fixed share of R. Background light
from the sample is taken to grow in proportion to the power, and the
detectors add dark counts, so the measured rate is

    rate(P) = R_inf P / (P + P_sat) + b P + d.

`fit_saturation` fits this to counts measured at several powers by
their Poisson likelihood and gives profile-likelihood intervals for
R_inf and P_sat. The curve is that of continuous-wave excitation in
this rate model; for data of another shape the fit's p-value says
that the curve does not describe them.
"""
from __future__ import annotations

import warnings

import numpy as np

from .pulsed import _deviance_p_value

__all__ = ["saturation_curve", "saturation_from_rates", "fit_saturation"]

_BACKGROUNDS = ("linear", "constant", "none")


def saturation_curve(power, R_inf: float, P_sat: float,
                     bg_slope: float = 0.0, dark: float = 0.0
                     ) -> "float | np.ndarray":
    """Count rate R_inf P / (P + P_sat) + bg_slope P + dark at the
    excitation powers `power` (any unit, the same as P_sat); the rate is
    in the unit of R_inf and dark (for example counts per second)."""
    P = np.asarray(power, dtype=float)
    out = R_inf * P / (P + P_sat) + bg_slope * P + dark
    return float(out) if out.ndim == 0 else out


def saturation_from_rates(k_r: float, k_es: float, k_se: float) -> dict:
    """The saturation curve of the three-level rate model of
    `sparq.exact` (rates in 1/ns): its steady-state emission rate is
    R_inf k_exc / (k_exc + k_sat) for a pump rate k_exc, with
    R_inf = k_r k_se / (k_se + k_es) (photons per ns) and
    k_sat = (k_r + k_es) k_se / (k_se + k_es) (1/ns). With k_exc
    proportional to the power, P_sat is the power at which
    k_exc = k_sat. Returns dict(R_inf, k_sat)."""
    k_r, k_es, k_se = float(k_r), float(k_es), float(k_se)
    if not (np.isfinite(k_r) and np.isfinite(k_es) and np.isfinite(k_se)
            and k_r > 0 and k_es >= 0 and k_se > 0):
        raise ValueError("need k_r > 0, k_es >= 0 and k_se > 0 (1/ns)")
    return dict(R_inf=k_r * k_se / (k_se + k_es),
                k_sat=(k_r + k_es) * k_se / (k_se + k_es))


def fit_saturation(power, counts, t_s=1.0, background: str = "linear",
                   level: float = 0.95, interval: str | None = "both"
                   ) -> dict:
    """Fit a saturation curve to counts measured at several excitation
    powers (new in 0.13.0).

    power : (n,) excitation powers (any unit, for example uW), >= 0.
    counts : (n,) detected counts (whole numbers, both detectors
        together or one, as you like) in `t_s` seconds at each power.
    t_s : counting time in seconds, one number or one per power.
    background : "linear" (default): rate(P) = R_inf P/(P + P_sat)
        + b P + d, with background light b P from the sample and dark
        counts d; "constant": b = 0; "none": b = d = 0. b and d are
        fitted (>= 0).
    level : confidence level of the intervals.
    interval : "both" (default), "P_sat" (only the P_sat interval, which
        is quick; the R_inf ends are then NaN) or None (no intervals).

    The counts are taken as Poisson; for a given P_sat the rate is
    linear in R_inf, b and d, which are then found exactly (a convex
    problem), so only P_sat is searched (on a log scale from 1e-3 times
    the smallest positive power to 1e3 times the largest). The interval
    for P_sat is where the profile deviance 2 [NLL(P_sat) - NLL_min]
    stays below the chi-square(1) quantile of `level` (Wilks), and the
    same for R_inf with P_sat, b and d profiled out. When the powers do
    not reach far enough into saturation, the curve cannot tell a high,
    late saturation from a linear rise, and the upper ends are not
    bounded: they are then NaN, `saturated` is False and a
    RuntimeWarning is issued (measure at higher powers). P_sat_lo is 0
    when the profile stays below the quantile down to the bottom of the
    search.

    Returns a dict: R_inf (counts/s), R_inf_lo, R_inf_hi, P_sat,
    P_sat_lo, P_sat_hi, bg_slope (counts/s per power unit), dark
    (counts/s), level, saturated (None with interval=None),
    signal_rate and rho (the fitted emitter rate and its share of the
    fitted total at each power), mu (the fitted mean counts),
    deviance, dof and p_value (goodness of
    fit from the exact moments of each point's Poisson deviance, as in
    `sparq.pulsed.fit_pulsed`; a small value says the curve does not
    describe the data, or the counts scatter more than Poisson counts,
    for example from a drifting focus, and the intervals are then too
    narrow).

    Refuses (ValueError): powers or counts that are not finite, negative
    powers, counts that are negative or not whole, non-positive times,
    arrays of different lengths, fewer distinct powers than the curve
    has parameters plus one, no positive power, no counts, counts at
    zero power with background="none" (from 0.14.0), an unknown
    `background` or `interval`, and a `level` outside (0, 1).
    """
    from scipy.optimize import brentq, minimize_scalar
    from scipy.stats import chi2

    P = np.asarray(power, dtype=float)
    n = np.asarray(counts, dtype=float)
    if P.ndim != 1 or n.shape != P.shape:
        raise ValueError("power and counts must be 1-D arrays of equal "
                         "length")
    if not (np.all(np.isfinite(P)) and np.all(np.isfinite(n))):
        raise ValueError("power and counts must be finite")
    if np.any(P < 0):
        raise ValueError("powers must be >= 0")
    if np.any(n < 0) or np.any(n != np.floor(n)):
        raise ValueError("counts must be whole, non-negative numbers")
    t = np.broadcast_to(np.asarray(t_s, dtype=float), P.shape).astype(float)
    if not (np.all(np.isfinite(t)) and np.all(t > 0)):
        raise ValueError("t_s must be positive (s)")
    if background not in _BACKGROUNDS:
        raise ValueError(f"background must be one of {_BACKGROUNDS}")
    if not (0.0 < float(level) < 1.0):
        raise ValueError("level must lie in (0, 1)")
    if interval not in ("both", "P_sat", None):
        raise ValueError('interval must be "both", "P_sat" or None')
    n_lin = {"linear": 3, "constant": 2, "none": 1}[background]
    if np.unique(P).size < n_lin + 2:
        raise ValueError(f"need at least {n_lin + 2} distinct powers for "
                         f"this curve ({n_lin + 1} parameters)")
    if not np.any(P > 0):
        raise ValueError("need at least one positive power")
    if n.sum() <= 0:
        raise ValueError("no counts")
    if background == "none" and np.any(n[P == 0] > 0):
        # new in 0.14.0: up to 0.13.0 this gave a fit with an infinite
        # deviance (R_inf and P_sat meaningless) without intervals, and
        # a NaN error from inside the interval search with them
        raise ValueError(
            "there are counts at zero power, which the curve without "
            "background (background='none') cannot give; use "
            "background='constant' or 'linear' (dark counts)")

    # columns of the linear part, scaled to unit size so that the Newton
    # steps of the Poisson solver see numbers of one order
    def columns(Ps):
        cols = [t * P / (P + Ps)]
        if background == "linear":
            cols.append(t * P)
        if background in ("linear", "constant"):
            cols.append(t.copy())
        return np.column_stack(cols)

    def scales(X):
        s = np.abs(X).max(axis=0)
        return np.where(s > 0, s, 1.0)

    rate_guess = n.sum() / t.sum()
    state = {}

    def solve(Ps, R_fixed=None):
        """Smallest objective (half the Poisson deviance) at this P_sat,
        and the linear coefficients (R_inf, [b], [d]) in counts/s."""
        X = columns(Ps)
        sc = scales(X)
        Xs = X / sc
        key = R_fixed is not None
        if R_fixed is None:
            b0 = state.get(key, np.full(n_lin, rate_guess)) * sc
            v, b = _poisson_linear(Xs, n, b0)
            beta = b / sc
        else:
            if n_lin == 1:
                mu = R_fixed * X[:, 0]
                if np.any(mu <= 0) and np.any(n[mu <= 0] > 0):
                    return np.inf, np.array([R_fixed])
                v = _half_deviance(n, mu)
                return v, np.array([R_fixed])
            off = R_fixed * X[:, 0]
            Xr, sr = Xs[:, 1:], sc[1:]
            b0 = state.get(key, np.full(n_lin - 1, rate_guess)) * sr
            v, b = _poisson_linear(Xr, n, b0, offset=off)
            beta = np.r_[R_fixed, b / sr]
        if np.isfinite(v):
            state[key] = beta if R_fixed is None else beta[1:]
        return v, beta

    pos = P[P > 0]
    lo_s, hi_s = np.log(pos.min() * 1e-3), np.log(pos.max() * 1e3)
    grid = np.linspace(lo_s, hi_s, 81)
    coarse = grid[::4]                  # 21 points for the R_inf profile

    def profile_s(R_fixed=None):
        g = grid if R_fixed is None else coarse
        vals = np.array([solve(np.exp(s), R_fixed)[0] for s in g])
        i = int(np.argmin(vals))
        a, b = g[max(i - 1, 0)], g[min(i + 1, g.size - 1)]
        res = minimize_scalar(lambda s: solve(np.exp(s), R_fixed)[0],
                              bounds=(a, b), method="bounded",
                              options=dict(xatol=1e-9))
        if res.fun <= vals[i]:
            return float(res.fun), float(res.x), vals
        return float(vals[i]), float(g[i]), vals

    f_min, s_hat, prof = profile_s()
    f_min, beta = solve(np.exp(s_hat))
    Ps_hat = float(np.exp(s_hat))
    R_hat = float(beta[0])
    crit = float(chi2.ppf(level, 1))

    # P_sat interval from the profile over the grid, refined by brentq
    def ex_s(s):
        return 2.0 * (solve(np.exp(s))[0] - f_min) - crit

    def s_end(upper):
        idx = np.searchsorted(grid, s_hat)
        rng = range(idx, grid.size) if upper else range(idx - 1, -1, -1)
        prev = s_hat
        for j in rng:
            if 2.0 * (prof[j] - f_min) - crit > 0:
                a, b = sorted((prev, grid[j]))
                return float(np.exp(brentq(ex_s, a, b, xtol=1e-9)))
            prev = grid[j]
        return np.nan

    Ps_lo = Ps_hi = R_lo = R_hi = np.nan
    if interval is not None:
        Ps_lo, Ps_hi = s_end(False), s_end(True)
        if np.isnan(Ps_lo):
            Ps_lo = 0.0

    # R_inf interval: P_sat, b and d profiled out
    def ex_R(R):
        return 2.0 * (profile_s(R)[0] - f_min) - crit

    if interval == "both":
        if R_hat <= 0 or ex_R(1e-9 * R_hat) <= 0:
            R_lo = 0.0
        else:
            R_lo = float(brentq(ex_R, 1e-9 * R_hat, R_hat,
                                xtol=1e-6 * R_hat))
        R_up = R_hat
        while R_up < 1e3 * R_hat:
            R_next = R_up * 1.5
            if ex_R(R_next) > 0:
                R_hi = float(brentq(ex_R, R_up, R_next, xtol=1e-6 * R_hat))
                break
            R_up = R_next
    _, beta = solve(Ps_hat)          # leave the state at the best fit
    X = columns(Ps_hat)
    mu = X @ beta
    rates = beta.tolist()
    R_inf = rates.pop(0)
    b = rates.pop(0) if background == "linear" else 0.0
    d = rates.pop(0) if background in ("linear", "constant") else 0.0
    sig = R_inf * P / (P + Ps_hat)
    tot = sig + b * P + d
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(tot > 0, sig / tot, np.nan)
    dev = 2.0 * f_min
    n_free = n_lin + 1
    dof = int(P.size - n_free)
    if interval is None:
        saturated = None
    else:
        saturated = bool(np.isfinite(Ps_hi) and (interval == "P_sat"
                                                 or np.isfinite(R_hi)))
    if saturated is False:
        warnings.warn(
            "the powers do not reach far enough into saturation: the "
            "upper end of the P_sat or R_inf interval is not bounded "
            "(NaN); measure at higher powers", RuntimeWarning, stacklevel=2)
    return dict(R_inf=float(R_inf), R_inf_lo=R_lo, R_inf_hi=R_hi,
                P_sat=Ps_hat, P_sat_lo=Ps_lo, P_sat_hi=Ps_hi,
                bg_slope=float(b), dark=float(d), level=float(level),
                saturated=saturated, signal_rate=sig, rho=rho, mu=mu,
                deviance=float(dev), dof=dof,
                p_value=_deviance_p_value(dev, mu, n_free) if dof > 0
                else np.nan)


def _half_deviance(h, mu):
    pos = h > 0
    return float(np.sum(mu) - h.sum()
                 - np.sum(h[pos] * np.log(mu[pos] / h[pos])))


def _poisson_linear(X, h, beta0, offset=None):
    """Minimize half the Poisson deviance of the counts h for the mean
    mu = X beta (+ offset) over beta >= 0 (a convex problem), by
    projected Newton steps with Levenberg-Marquardt damping, which stay
    reliable when the columns of X are nearly parallel (a P_sat far
    below or far above the powers). Returns (objective, beta); the
    objective is inf when no beta >= 0 gives a positive mean wherever
    there are counts."""
    X = np.asarray(X, dtype=float)
    off = 0.0 if offset is None else offset
    pos = h > 0
    hp = h[pos]
    hsum = float(h.sum())

    def f(b):
        mu = X @ b + off
        if np.any(mu[pos] <= 0.0) or np.any(mu < 0.0) \
                or not np.all(np.isfinite(mu)):
            return np.inf, mu
        return float(np.sum(mu) - hsum
                     - np.sum(hp * np.log(mu[pos] / hp))), mu

    b = np.maximum(np.asarray(beta0, dtype=float), 0.0)
    fb, mu = f(b)
    if not np.isfinite(fb):
        # a start with every coefficient positive, sized to the counts
        b = np.full(X.shape[1], max(hsum, 1.0) / max(X.sum(), 1e-300))
        fb, mu = f(b)
        if not np.isfinite(fb):
            return np.inf, b
    lam = 1e-9
    for _ in range(500):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(pos, h / mu, 0.0)
            w = np.where(pos, h / mu ** 2, 0.0)
        g = X.T @ (1.0 - r)
        H = (X * w[:, None]).T @ X
        free = ~((b <= 0.0) & (g > 0.0))
        if not free.any():
            break
        Hf = H[np.ix_(free, free)]
        gf = g[free]
        D = np.maximum(np.diag(Hf), 1e-12 * max(np.diag(Hf).max(), 1e-300))
        step = None
        while lam < 1e12:
            try:
                df = -np.linalg.solve(Hf + lam * np.diag(D), gf)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            if -gf @ df <= 1e-13 * (1.0 + abs(fb)):
                break                       # Newton decrement: converged
            bn = b.copy()
            bn[free] += df
            bn = np.maximum(bn, 0.0)
            fn, mun = f(bn)
            if fn <= fb:
                step = (bn, fn, mun)
                break
            lam *= 10.0
        if step is None:
            break
        done = fb - step[1] <= 1e-15 * (1.0 + abs(fb))
        b, fb, mu = step
        lam = max(lam / 10.0, 1e-12)
        if done:
            break
    return fb, b
