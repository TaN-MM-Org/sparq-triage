"""Excited-state lifetime of an emitter from a time-correlated
single-photon counting (TCSPC) decay histogram (new in 0.14.0).

With a pulsed laser, a TCSPC card in T3 mode records for every detected
photon its delay after the laser pulse (`read_ptu` returns it as
`dtime`). The histogram of these delays is the emitter's decay. This
module fits it.

Model. A photon emitted by a decay of lifetime tau that starts at t0
and is timed with Gaussian jitter of standard deviation sigma
(laser, detector and electronics together: the instrument response,
IRF) has the delay density of an exponential blurred by a Gaussian
(an exponentially modified Gaussian),

    p(t) = 1/(2 tau) exp(sigma^2/(2 tau^2) - (t - t0)/tau)
           erfc((sigma/tau - (t - t0)/sigma)/sqrt(2)),

whose distribution function is

    F(t) = Phi(x/sigma) - exp(sigma^2/(2 tau^2) - x/tau)
           Phi(x/sigma - sigma/tau),     x = t - t0,

with Phi the standard normal distribution function. Both follow from
the convolution integral; the tests check F against SciPy's
independent `scipy.stats.exponnorm` and against a numerical
convolution. Every bin's share of a decay is a difference of F at its
edges (no midpoint rule), evaluated with the scaled complementary
error function so that nothing overflows or cancels.

With a laser period t_rep, the decays of earlier pulses that have not
died out add to the histogram: a bin holds the sum of the shares of
the decays that start at t0 - m t_rep for all whole m. The terms in
which the Gaussian matters are summed one by one; the rest are pure
exponentials and form a geometric series, summed in closed form (the
tests compare with a direct sum over the pulses, back to where less
than 1e-18 of a decay is left).

The mean counts in a bin are

    mu = sum_j A_j P_j + B w,

with P_j the bin's share of component j (lifetime tau_j), A_j the
number of photons of component j in one period (with t_rep) or in the
whole decay (without), B a flat background in counts per ns of delay
(dark counts, uncorrelated light) and w the bin width. `fit_lifetime`
maximizes the Poisson likelihood of the counts: for given t0, sigma
and lifetimes the A_j and B enter linearly and are found exactly (a
convex problem), so only t0, sigma and the lifetimes are searched.

A measured instrument response (new in 0.15.0). A Gaussian does not
describe every response: many detectors add a slower tail. With `irf`,
the counts of a measured response on the same bins (for example of the
laser light scattered into the detector) replace the Gaussian. Each of
its bins is taken to hold its counts spread evenly over the bin; the
delay of a photon is then a time drawn from the response plus the
exponential, and a decay bin's share of the photons excited in a
response bin j is
    [Psi(x + w) - 2 Psi(x) + Psi(x - w)] / w,
    Psi(x) = int_0^x (1 - e^(-z/tau)) dz = x - tau (1 - e^(-x/tau))
    (x > 0; 0 otherwise),
with w the bin width and x the start of the decay bin minus the start
of the (shifted) response bin. Every bin of the decay is a sum of these
over the response bins (a convolution); with t_rep the pulses whose
response ends before the record begins add a geometric series, summed
in closed form. The tests check these shares against the convolution
integral computed by quadrature (1e-12), against a direct sum over the
pulses (1e-12), and against the Gaussian model of 0.14.0 for a finely
binned Gaussian response (the difference falls with the square of the
bin width).

What the model leaves out: the distortions of high count rates (more
than one photon per laser period, detector dead time) and, with a
measured response, its own counting noise and its shape within a bin.
The fit's p-value shows when the model does not describe the data.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.special import erfcx

from .pulsed import _deviance_p_value, _edges_from_centers

__all__ = ["expected_decay", "fit_lifetime"]

_SQ2 = np.sqrt(2.0)
_U_ZERO = 39.0       # beyond 39 sigma from t0 the Gaussian terms underflow


def _upper(x, tau, s):
    """1 - F(x) for x >= 0 (the decay's share above t0 + x)."""
    x = np.asarray(x, dtype=float)
    if s <= 0.0:
        return np.exp(-x / tau)
    u = x / s
    k = s / tau
    g = np.exp(-0.5 * u * u)
    e = 0.5 * g * erfcx(np.abs(k - u) / _SQ2)
    t1 = 0.5 * g * erfcx(u / _SQ2)
    tail = np.exp(np.minimum(k * (0.5 * k - u), 0.0))
    return t1 + np.where(k >= u, e, tail - e)


def _lower(x, tau, s):
    """F(x) for x <= 0 (the decay's share below t0 + x)."""
    x = np.asarray(x, dtype=float)
    if s <= 0.0:
        return np.zeros(x.shape)
    u = x / s
    k = s / tau
    g = np.exp(-0.5 * u * u)
    return 0.5 * g * (erfcx(-u / _SQ2) - erfcx((k - u) / _SQ2))


def _bin_shares(x, tau, s):
    """Share of one decay (start at 0) in the bins with edges x (along
    the last axis)."""
    up = _upper(np.maximum(x, 0.0), tau, s)
    lo = _lower(np.minimum(x, 0.0), tau, s)
    a_up, b_up = up[..., :-1], up[..., 1:]
    a_lo, b_lo = lo[..., :-1], lo[..., 1:]
    a, b = x[..., :-1], x[..., 1:]
    return np.where(a >= 0.0, a_up - b_up,
                    np.where(b <= 0.0, b_lo - a_lo, 1.0 - a_lo - b_up))


def _shares(edges, t0, tau, s, t_rep):
    """Share of the decay that starts at t0 in each bin, with the decays
    of the pulses at t0 - m t_rep for all whole m added (t_rep given),
    or of that one decay alone (t_rep None)."""
    x0 = edges - t0
    if t_rep is None:
        return _bin_shares(x0, tau, s)
    T = float(t_rep)
    k = s / tau
    # later pulses (m < 0) reach the record only through the Gaussian's
    # left side, which underflows beyond 39 sigma
    m_lo = int(np.floor((-_U_ZERO * s - x0[-1]) / T))
    # from m_hi on, every bin lies where 1 - F is a pure exponential to
    # within e^-50 (10 sigma beyond sigma^2/tau); for a lifetime below
    # sigma/30, where 1 - F is below e^-450 beyond 39 sigma, those
    # terms are left out
    if k <= 30.0:
        x_th, geometric = s * (k + 10.0), True
    else:
        x_th, geometric = _U_ZERO * s, False
    m_hi = max(m_lo, int(np.ceil((x_th - x0[0]) / T)))
    out = np.zeros(x0.size - 1)
    if m_hi > m_lo:
        m = np.arange(m_lo, m_hi, dtype=float)
        out += _bin_shares(x0[None, :] + m[:, None] * T, tau, s).sum(axis=0)
    if geometric:
        xa = x0[:-1] + m_hi * T
        w = np.diff(x0)
        with np.errstate(under="ignore"):
            out += (np.exp(0.5 * k * k - xa / tau) * -np.expm1(-w / tau)
                    / -np.expm1(-T / tau))
    return out


# ----------------------------------------------------------------------
# a measured instrument response (new in 0.15.0)
# ----------------------------------------------------------------------

def _psi(x, tau):
    """int_0^x F(z) dz for the exponential distribution function
    F(z) = 1 - e^(-z/tau) (0 for x <= 0): tau (z - 1 + e^-z), z = x/tau,
    with the series z^2/2 - z^3/6 + ... below z = 0.01, where the closed
    form would lose digits."""
    z = np.maximum(np.asarray(x, dtype=float), 0.0) / tau
    out = np.expm1(-z) + z
    small = z < 0.01
    if np.any(small):
        zs = z[small]
        s = np.zeros(zs.shape)
        for k in range(9, 1, -1):          # z^2/2! - z^3/3! + ... + z^9/9!
            s = zs * (s + (-1.0) ** k / _FACT[k])
        out[small] = s * zs
    return tau * out


_FACT = [1.0]
for _k in range(1, 12):
    _FACT.append(_FACT[-1] * _k)


def _irf_kernel(n, w, shift, tau):
    """beta(k), k = -(n-1) ... n-1: the share in decay bin i = j + k of
    the photons of a decay (lifetime tau) excited at a time spread evenly
    over response bin j shifted by `shift`; bins of width w. With
    x = k w - shift (the start of the decay bin minus the start of the
    shifted response bin),
        beta = [Psi(x + w) - 2 Psi(x) + Psi(x - w)] / w,
    the second difference of `_psi`; where x >= w (the response bin ends
    before the decay bin starts) it is written as the product
        tau e^(-(x - w)/tau) (1 - e^(-w/tau))^2 / w,
    which keeps its precision far from the response."""
    k = np.arange(-(n - 1), n, dtype=float)
    x = k * w - shift
    out = np.empty(k.size)
    sep = x >= w
    E = -np.expm1(-w / tau)
    with np.errstate(under="ignore"):
        out[sep] = tau * np.exp(-(x[sep] - w) / tau) * E * E / w
    xs = x[~sep]
    out[~sep] = (_psi(xs + w, tau) - 2.0 * _psi(xs, tau)
                 + _psi(xs - w, tau)) / w
    return np.maximum(out, 0.0)


def _irf_shares(w, q, shift, tau, t_rep):
    """Share of one pulse's decay in each of the n = q.size bins of width
    w, for the instrument response q (weights summing to
    1, each spread evenly over its bin, the bins the same as the decay's)
    shifted by `shift`; with t_rep the decays of all other pulses are
    added. The pulses whose response ends before the record begins form
    a geometric series, summed in closed form; the others are summed
    one by one (a convolution with `_irf_kernel` each)."""
    n = q.size

    def one(s):
        return np.convolve(q, _irf_kernel(n, w, s, tau))[n - 1:2 * n - 1]

    if t_rep is None:
        return one(shift)
    T = float(t_rep)
    span = n * w
    # pulse m (response at its bins shifted by shift - m T) reaches the
    # record when m > (shift - span) / T; from m_g on its response ends
    # before every decay bin starts
    m_lo = int(np.floor((shift - span) / T)) + 1
    m_g = max(m_lo, int(np.ceil((span + shift) / T)))
    out = np.zeros(n)
    for m in range(m_lo, m_g):
        out += one(shift - m * T)
    # m >= m_g: sum_j q_j tau E^2/w e^(-((i - j - 1) w - shift + m T)/tau)
    # summed over m; both exponents below are >= 0
    E = -np.expm1(-w / tau)
    j = np.arange(n)
    i = np.arange(n)
    with np.errstate(under="ignore"):
        inner = float(q @ np.exp(-((-j - 1) * w - shift + m_g * T) / tau))
        out += (tau * E * E / w) * np.exp(-i * w / tau) * inner \
            / -np.expm1(-T / tau)
    return out


def _check_irf(irf, n, flipped):
    """The measured response as weights summing to 1 (in the order of the
    increasing delays)."""
    q = np.asarray(irf, dtype=float)
    if q.ndim != 1 or q.size != n:
        raise ValueError(f"irf must hold one count per bin ({n}), on the "
                         "same delays as the decay")
    if not np.all(np.isfinite(q)) or np.any(q < 0):
        raise ValueError("irf must be finite and non-negative (subtract "
                         "its background and set negative bins to 0)")
    if q.sum() <= 0:
        raise ValueError("irf holds no counts")
    if flipped:
        q = q[::-1]
    return q / q.sum()


def _uniform_width(edges):
    """The width of equal bins; refuses unequal ones."""
    w = np.diff(edges)
    wm = float(np.mean(w))
    if np.max(np.abs(w - wm)) > 1e-6 * wm:
        raise ValueError("a measured instrument response needs bins of "
                         "equal width")
    return wm


def _poisson_nnls(X, h, beta0):
    """Minimize half the Poisson deviance of the counts h for the mean
    mu = X beta over beta >= 0 (a convex problem) by an active-set Newton
    method: Newton steps on the coefficients that are positive or whose
    gradient points into beta > 0; a coefficient at 0 that the step
    would push below 0 is held there and the step is solved again
    without it (so every step is a descent direction, which projecting
    the full Newton step onto beta >= 0 is not); the step is cut at the
    first coefficient that reaches 0 and halved until the objective
    falls. Converged when the Newton decrement g H^-1 g of the free
    coefficients is below 1e-11 (1 + objective) with no coefficient
    held back. mu may be 0 where there are no counts (where there are
    counts it must exceed 1e-100). Returns
    (objective, beta); the objective is inf when no beta >= 0 gives a
    positive mean wherever there are counts."""
    pos = h > 0
    hp = h[pos]
    hsum = float(h.sum())

    def f(b):
        mu = X @ b
        # a mean below 1e-100 where there are counts is treated as 0:
        # such points are never near the minimum (the background column
        # alone does better), and their Hessian would overflow
        if np.any(mu[pos] <= 1e-100) or not np.all(np.isfinite(mu)):
            return np.inf, mu
        return float(np.sum(mu) - hsum
                     - np.sum(hp * np.log(mu[pos] / hp))), mu

    colsum = X.sum(axis=0)
    # start every coefficient at least at a thousandth of the size that
    # would carry the counts on its own: from 0, the Newton steps of a
    # coefficient whose column is all but zero where there are counts
    # grow it only by a factor of about two each. At the minimum the
    # means add up to the counts (sum_j beta_j colsum_j = sum h), so no
    # coefficient exceeds sum h / colsum_j; a start above twice that (a
    # warm start from a far-off shape) is brought down to it.
    nat = max(hsum, 1.0) / np.maximum(colsum, 1e-300)
    b = np.clip(np.nan_to_num(np.asarray(beta0, dtype=float), nan=0.0,
                              posinf=np.inf),
                1e-3 * nat / X.shape[1], 2.0 * nat)
    fb, mu = f(b)
    if not np.isfinite(fb):
        return np.inf, b
    for _ in range(500):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            r = np.where(pos, h / np.where(mu > 0, mu, 1.0), 0.0)
            w = np.where(pos, r / np.where(mu > 0, mu, 1.0), 0.0)
            g = colsum - X.T @ r
            H = (X * w[:, None]).T @ X
        if not (np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
            break                   # a hopeless shape; keep this point
        free = (b > 0.0) | (g < 0.0)
        held = False
        d = np.zeros_like(b)
        while free.any():
            Hf = H[np.ix_(free, free)]
            Hf = Hf + np.eye(Hf.shape[0]) * 1e-13 * max(np.trace(Hf),
                                                        1e-300)
            try:
                d[free] = -np.linalg.solve(Hf, g[free])
            except np.linalg.LinAlgError:
                d[free] = -g[free] / np.maximum(np.diag(Hf), 1e-300)
            blocked = free & (b <= 0.0) & (d < 0.0)
            if not blocked.any():
                break
            free &= ~blocked
            d[~free] = 0.0
            held = True
        if not free.any():
            break
        dec = -float(g[free] @ d[free])
        if dec <= 1e-11 * (1.0 + abs(fb)) and not held:
            break
        neg = free & (d < 0.0)
        t_max = float(np.min(b[neg] / -d[neg])) if neg.any() else np.inf
        t = min(1.0, t_max)
        fn = np.inf
        while t > 1e-20:
            bn = np.maximum(b + t * d, 0.0)
            if t == t_max:
                bn[neg & (b + t * d <= b * 1e-12)] = 0.0
            fn, mun = f(bn)
            if fn < fb:
                break
            t *= 0.5
        if not fn < fb:
            if held:
                continue
            break
        b, fb, mu = bn, fn, mun
    return fb, b


def _check_axis(delay):
    d = np.asarray(delay, dtype=float)
    if d.ndim != 1 or d.size < 5:
        raise ValueError("delay must be a 1-D array of at least 5 bin "
                         "centers (ns)")
    if not np.all(np.isfinite(d)):
        raise ValueError("delays must be finite")
    step = np.diff(d)
    if np.all(step < 0):
        return d[::-1], True
    if np.any(step <= 0):
        raise ValueError("delays must be strictly increasing (or "
                         "decreasing)")
    return d, False


def _check_t_rep(t_rep):
    if t_rep is None:
        return None
    try:
        T = float(t_rep)
    except (TypeError, ValueError):
        raise ValueError("t_rep must be a positive number (ns) or None") \
            from None
    if not (np.isfinite(T) and T > 0.0):
        raise ValueError("t_rep must be positive and finite (ns) or None")
    return T


def expected_decay(delay, area, tau, t0: float, sigma_irf: float = 0.0,
                   t_rep: float | None = None,
                   background: float = 0.0, irf=None) -> np.ndarray:
    """Mean counts per bin of a TCSPC decay histogram (new in 0.14.0).

    delay : bin centers in ns (strictly increasing or decreasing; the bin
        edges are the midpoints, as in `fit_pulsed`).
    area : photons of the decay (one number, or one per component): per
        laser period with `t_rep`, in the whole decay without.
    tau : lifetime(s) in ns, as many as `area`.
    t0 : delay at which the decay starts (ns); with `irf`, the shift of
        the measured response instead (0: as measured).
    sigma_irf : standard deviation of the Gaussian instrument response
        (ns; 0 for none).
    t_rep : laser period (ns); with it the decays of earlier (and later)
        pulses are added, without it the histogram holds one decay.
    background : flat background in counts per ns of delay.
    irf : a measured instrument response (new in 0.15.0): its counts on
        the same bins as the decay, for example a histogram of the laser
        light scattered into the detector, with its own background
        subtracted. The decay is then the exponential convolved with this
        response, shifted by t0, instead of with a Gaussian (sigma_irf
        must be 0). The response's counts in each bin are taken to be
        spread evenly over the bin, and each bin's share of the decay is
        then a closed form (see `fit_lifetime`). The bins must have equal
        widths.

    Returns the mean counts of each bin (in the order of `delay`).
    """
    d, flipped = _check_axis(delay)
    A = np.atleast_1d(np.asarray(area, dtype=float))
    taus = np.atleast_1d(np.asarray(tau, dtype=float))
    if A.ndim != 1 or A.shape != taus.shape:
        raise ValueError("area and tau must have the same length")
    if not (np.all(np.isfinite(taus)) and np.all(taus > 0)):
        raise ValueError("lifetimes must be positive and finite (ns)")
    if not (np.all(np.isfinite(A)) and np.all(A >= 0)):
        raise ValueError("areas must be finite and >= 0")
    s = float(sigma_irf)
    if not (np.isfinite(s) and s >= 0.0):
        raise ValueError("sigma_irf must be finite and >= 0 (ns)")
    t0 = float(t0)
    if not np.isfinite(t0):
        raise ValueError("t0 must be finite (ns)")
    bg = float(background)
    if not (np.isfinite(bg) and bg >= 0.0):
        raise ValueError("background must be finite and >= 0")
    T = _check_t_rep(t_rep)
    edges = _edges_from_centers(d)
    mu = bg * np.diff(edges)
    if irf is not None:
        if s != 0.0:
            raise ValueError("give sigma_irf (a Gaussian response) or irf "
                             "(a measured one), not both")
        q = _check_irf(irf, d.size, flipped)
        w = _uniform_width(edges)
        for a, ta in zip(A, taus):
            mu = mu + a * _irf_shares(w, q, t0, float(ta), T)
        return mu[::-1] if flipped else mu
    for a, ta in zip(A, taus):
        mu = mu + a * _shares(edges, t0, float(ta), s, T)
    return mu[::-1] if flipped else mu


def fit_lifetime(delay, counts, t_rep: float | None = None,
                 n_components: int = 1, sigma_irf: float | None = None,
                 t0: float | None = None, level: float = 0.95,
                 interval: bool = True, irf=None) -> dict:
    """Fit the decay of a TCSPC histogram: lifetime(s) with
    profile-likelihood intervals (new in 0.14.0).

    delay, counts : bin centers (ns; strictly increasing or decreasing)
        and the photon counts of each bin (for example the histogram of
        the T3 `dtime` values of `read_ptu`, times the resolution). The
        bins must cover the rise of the decay, so that t0 and the
        instrument response are seen.
    t_rep : the laser period (ns). Give it for pulsed data: the decays
        of earlier pulses, of which a share e^(-t_rep/tau) is left at
        each next pulse, are then added (see the module docstring); when
        that share is negligible the fit agrees with the one without it.
        Without it the histogram is taken to hold one decay, and a
        lifetime that is not short compared with the period comes out
        wrong (too short).
    n_components : 1 (default) or 2 decays with their own lifetimes and
        the same t0 and instrument response.
    sigma_irf : standard deviation (ns) of the Gaussian instrument
        response, if known (for example from a measurement of the
        laser light); fitted otherwise.
    t0 : the delay at which the decay starts, if known; fitted
        otherwise.
    level : confidence level of the intervals.
    interval : False skips the intervals.
    irf : a measured instrument response (new in 0.15.0): its counts on
        the same (equal) bins as the decay, for example a histogram of the
        laser light scattered into the detector, recorded with the same
        settings, with its own background subtracted. The decay is then
        fitted as the exponential(s) convolved with this response (the
        usual "reconvolution" fit), so a response of any shape, such as
        one with a slow tail, is described. Its counts in each bin are
        taken to be spread evenly over the bin, and each bin's share of
        the decay is then a closed form (an error of second order in the
        bin width against the true shape of the response); with `t_rep`
        the pulses whose response ends before the record are summed as a
        geometric series. `t0` is then the shift of the response (0: as
        measured), fitted unless given; `sigma_irf` must not be given and
        is returned as NaN. The response's own counting noise is not
        included in the intervals: record it with many more counts than
        the decay.

    The search starts from a scan of lifetimes between the bin width
    and the record length (or the period, if longer) and is local from
    there; with two components the likelihood can have more than one
    minimum, and a component far outside that range is not looked for.

    The counts are taken as Poisson. The interval of a lifetime is
    where the profile deviance 2 [NLL(tau) - NLL_min], with every other
    parameter fitted again, stays below the chi-square(1) quantile of
    `level` (Wilks). When the data do not bound a lifetime (a lifetime
    much longer than the record, or much shorter than the instrument
    response), the open end is NaN above or 0 below, `resolved` is
    False and a RuntimeWarning is issued.

    Returns a dict: tau, tau_lo, tau_hi (arrays, one entry per
    component, shortest lifetime first; ns), area (photons of each
    component, per period with t_rep), fraction (each component's share
    of the emitter's photons), t0, sigma_irf (ns), background (counts
    per ns of delay), mu (the fitted mean counts, in the order of
    `delay`), deviance, dof, p_value (goodness of fit from the exact
    moments of each bin's Poisson deviance, as in `fit_pulsed`; a small
    value says the model does not describe the decay, for example a
    second lifetime or an instrument response with a slower tail than
    a Gaussian; give `irf` for such a response),
    at_bound (searched parameters that ended on a search limit), level,
    resolved (None without intervals) and ok.

    Refuses (ValueError): delays that are not finite or not strictly
    monotonic, fewer than 5 bins, counts that are negative or not
    finite or all zero, arrays of different lengths, too few bins for
    the parameters, a t_rep, sigma_irf or t0 that is not a finite
    number (t_rep must be positive, sigma_irf at least 0), an
    n_components other than 1 or 2 and a level outside (0, 1); with
    `irf` (from 0.15.0) also a response of another length, with negative
    or non-finite counts or none, `sigma_irf` given as well, and bins of
    unequal width.
    """
    from scipy.optimize import brentq, minimize
    from scipy.stats import chi2

    d, flipped = _check_axis(delay)
    h = np.asarray(counts, dtype=float)
    if h.shape != np.asarray(delay).shape:
        raise ValueError("delay and counts must have the same length")
    if flipped:
        h = h[::-1]
    if not np.all(np.isfinite(h)) or np.any(h < 0):
        raise ValueError("counts must be finite and non-negative")
    if h.sum() <= 0:
        raise ValueError("the histogram has no counts")
    if n_components not in (1, 2) or isinstance(n_components, bool):
        raise ValueError("n_components must be 1 or 2")
    T = _check_t_rep(t_rep)
    if not (0.0 < float(level) < 1.0):
        raise ValueError("level must lie in (0, 1)")
    if sigma_irf is not None:
        s_fix = float(sigma_irf)
        if not (np.isfinite(s_fix) and s_fix >= 0.0):
            raise ValueError("sigma_irf must be finite and >= 0 (ns)")
    if t0 is not None:
        t0_fix = float(t0)
        if not np.isfinite(t0_fix):
            raise ValueError("t0 must be finite (ns)")
    nc = int(n_components)
    edges = _edges_from_centers(d)
    width = np.diff(edges)
    span = float(edges[-1] - edges[0])
    use_irf = irf is not None
    if use_irf:
        if sigma_irf is not None:
            raise ValueError("give sigma_irf (a Gaussian response) or irf "
                             "(a measured one), not both")
        q_irf = _check_irf(irf, h.size, flipped)
        w_irf = _uniform_width(edges)
        s_fix = 0.0
    names = []
    if t0 is None:
        names.append("t0")
    names += [f"tau{j + 1}" for j in range(nc)]
    if sigma_irf is None and not use_irf:
        names.append("sigma_irf")
    n_lin = nc + 1
    if h.size <= n_lin + len(names) + 1:
        raise ValueError(f"need more than {n_lin + len(names) + 1} bins "
                         "for this model")

    # search limits (t0 in ns; lifetimes and sigma on a log scale)
    smooth = np.convolve(h, np.ones(3) / 3.0, mode="same")
    c_peak = float(d[int(np.argmax(smooth))])
    tau_lo_b = 1e-3
    tau_hi_b = 1e3 * max(span, T or 0.0)
    lo_b, hi_b = [], []
    for n in names:
        if n == "t0" and use_irf:          # the shift of the response
            half = 0.5 * T if T is not None else 0.25 * span
            lo_b.append(-half)
            hi_b.append(half)
        elif n == "t0":
            half = 0.5 * T if T is not None else span
            lo_b.append(c_peak - half)
            hi_b.append(c_peak + (0.5 * T if T is not None else
                                  0.25 * span))
        elif n.startswith("tau"):
            lo_b.append(np.log(tau_lo_b))
            hi_b.append(np.log(tau_hi_b))
        else:
            lo_b.append(np.log(1e-4))
            hi_b.append(np.log(0.25 * span))
    lo_b, hi_b = np.array(lo_b), np.array(hi_b)
    i_t0 = names.index("t0") if "t0" in names else None
    i_s = names.index("sigma_irf") if "sigma_irf" in names else None
    i_tau = [names.index(f"tau{j + 1}") for j in range(nc)]

    def unpack(th):
        c = th[i_t0] if i_t0 is not None else t0_fix
        s = float(np.exp(th[i_s])) if i_s is not None else s_fix
        taus = [float(np.exp(th[i])) for i in i_tau]
        return float(c), taus, s

    def columns(th):
        c, taus, s = unpack(th)
        if use_irf:
            cols = [_irf_shares(w_irf, q_irf, c, ta, T) for ta in taus]
        else:
            cols = [_shares(edges, c, ta, s, T) for ta in taus]
        cols.append(width.copy())
        return np.column_stack(cols)

    pos = h > 0
    state = {"beta": None}

    def solve(th):
        X = columns(th)
        sc = np.abs(X).max(axis=0)
        sc = np.where(sc > 0, sc, 1.0)
        b0 = state["beta"]
        if b0 is None:
            b0 = np.full(n_lin, h.sum() / X.sum())
        v, b = _poisson_nnls(X / sc, h, b0 * sc)
        beta = b / sc
        if np.isfinite(v):
            state["beta"] = beta
        return v, beta, X

    def objective(th, which=None):
        """Half the Poisson deviance with the linear parameters solved,
        and its gradient with respect to the parameters `which` (all by
        default): by the envelope theorem, the derivative at the solved
        linear parameters, with the derivatives of the bin shares taken
        by central differences of their closed form."""
        which = range(th.size) if which is None else which
        v, beta, X = solve(th)
        if not np.isfinite(v):
            return 1e300, np.zeros(len(which))
        mu = X @ beta
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            r = np.where(pos, h / np.where(mu > 0, mu, 1.0), 0.0)
        wgt = 1.0 - r
        if not np.all(np.isfinite(wgt)):
            return 1e300, np.zeros(len(which))
        g = np.zeros(len(which))
        for n, i in enumerate(which):
            step = 1e-6 * (max(unpack(th)[2], float(np.min(width)))
                           if i == i_t0 else 1.0)
            tp, tm = th.copy(), th.copy()
            tp[i] += step
            tm[i] -= step
            dX = (columns(tp)[:, :-1] - columns(tm)[:, :-1]) / (2.0 * step)
            g[n] = float(wgt @ (dX @ beta[:-1]))
        return float(v), g

    # starting points: a coarse scan of the shape, then a bounded
    # quasi-Newton search from the best points of the scan
    w_min = float(np.min(width))
    tau_scan = np.geomspace(max(w_min, 2e-3), max(span, T or 0.0), 8)
    s_scan = (np.geomspace(max(0.5 * w_min, 2e-4), 0.05 * span, 3)
              if sigma_irf is None and not use_irf else [None])
    if use_irf:
        # the response as measured, and shifted so that its peak falls
        # on the decay's
        i_pk = int(np.argmax(np.convolve(q_irf, np.ones(3) / 3.0,
                                         mode="same")))
        t0_starts = sorted({0.0, float(c_peak - d[i_pk])})
    else:
        t0_starts = [None]
    if nc == 1:
        tau_sets = [(t,) for t in tau_scan]
    else:
        tau_sets = [(a, b) for i, a in enumerate(tau_scan)
                    for b in tau_scan[i + 1:]]
    scan = []
    for s0, t0s in [(a, b) for a in s_scan for b in t0_starts]:
        for ts in tau_sets:
            x0 = []
            for n in names:
                if n == "t0" and use_irf:
                    x0.append(t0s)
                elif n == "t0":
                    x0.append(c_peak - (s0 if s0 is not None else s_fix))
                elif n == "sigma_irf":
                    x0.append(np.log(s0))
                else:
                    x0.append(np.log(ts[int(n[3:]) - 1]))
            x0 = np.clip(np.array(x0), lo_b, hi_b)
            state["beta"] = None
            scan.append((solve(x0)[0], x0))
    scan.sort(key=lambda t: t[0])
    bnds = list(zip(lo_b, hi_b))
    best = None
    for _, x0 in scan[:2]:
        state["beta"] = None
        res = minimize(objective, x0, jac=True, method="L-BFGS-B",
                       bounds=bnds)
        if np.isfinite(res.fun) and res.fun < 1e299 and (
                best is None or res.fun < best.fun):
            best = res
    if best is None:
        return dict(ok=False, tau=np.full(nc, np.nan),
                    tau_lo=np.full(nc, np.nan), tau_hi=np.full(nc, np.nan))
    th_hat = best.x.copy()
    state["beta"] = None
    f_min, beta, X = solve(th_hat)
    mu = X @ beta
    c, taus, s = unpack(th_hat)
    order = np.argsort(taus)
    dev = 2.0 * f_min
    n_free = n_lin + len(names)
    dof = int(h.size - n_free)
    # lifetimes named in the returned order (shortest first)
    rank = {f"tau{j + 1}": f"tau{int(r) + 1}"
            for j, r in enumerate(np.argsort(order))}
    at_bound = [rank.get(n, n) + ("_low" if abs(x - l) < 1e-3 else "_high")
                for n, x, l, u in zip(names, th_hat, lo_b, hi_b)
                if abs(x - l) < 1e-3 or abs(x - u) < 1e-3]
    areas = beta[:nc]
    tot = float(areas.sum())
    out = dict(ok=True, tau=np.array(taus)[order],
               tau_lo=np.full(nc, np.nan), tau_hi=np.full(nc, np.nan),
               area=areas[order],
               fraction=(areas[order] / tot if tot > 0
                         else np.full(nc, np.nan)),
               t0=float(c), sigma_irf=np.nan if use_irf else float(s),
               background=float(beta[-1]),
               mu=mu[::-1] if flipped else mu, deviance=float(dev),
               dof=dof,
               p_value=(_deviance_p_value(dev, mu, n_free) if dof > 0
                        else np.nan),
               at_bound=at_bound, level=float(level), resolved=None)
    if not interval:
        return out

    crit = float(chi2.ppf(level, 1))
    beta_hat = beta.copy()

    def profile(i, v, known):
        """Smallest objective with the log lifetime i fixed at v, started
        from the solution at the nearest value already profiled (`known`:
        v -> (objective, parameters)); the other lifetime stays on its
        side of this one."""
        if v in known:
            return known[v][0]
        keep = [j for j in range(th_hat.size) if j != i]
        lo_p, hi_p = lo_b.copy(), hi_b.copy()
        for j in i_tau:
            if j == i:
                continue
            if th_hat[j] > th_hat[i]:
                lo_p[j] = max(lo_p[j], v)
            else:
                hi_p[j] = min(hi_p[j], v)
        near = min(known, key=lambda u: abs(u - v)) if known else None
        start = known[near][1] if near is not None else th_hat
        x0 = np.clip(start[keep], lo_p[keep], hi_p[keep])

        def f(y):
            th = th_hat.copy()
            th[keep] = y
            th[i] = v
            return objective(th, keep)

        state["beta"] = beta_hat.copy()
        th = th_hat.copy()
        th[i] = v
        if not keep:                    # nothing else is searched
            val = float(f(x0)[0])
        else:
            r = minimize(f, x0, jac=True, method="L-BFGS-B",
                         bounds=list(zip(lo_p[keep], hi_p[keep])))
            val = float(r.fun)
            th[keep] = r.x
        known[v] = (val, th)
        return val

    los, his = np.full(nc, np.nan), np.full(nc, np.nan)
    for slot, i in enumerate(i_tau):
        v_hat = float(th_hat[i])
        ends = []
        for sgn, bound in ((-1.0, float(lo_b[i])), (1.0, float(hi_b[i]))):
            # A first step of 0.01 in log tau; the curvature of the
            # profile there predicts the crossing. It is then bracketed,
            # stepping out by half again each time (by at most 1 in log
            # tau, so that each profile starts near the last), and
            # refined by brentq. Every profiled value is kept, so the
            # bracket's ends are not recomputed from other starts.
            known = {v_hat: (f_min, th_hat.copy())}

            def excess(v, i=i, known=known):
                return 2.0 * (profile(i, float(v), known) - f_min) - crit

            room = abs(bound - v_hat)
            end = np.nan
            v_in = v_hat
            dist = min(0.01, room)
            ex = excess(v_hat + sgn * dist)
            if ex <= 0.0 and dist < room:
                v_in = v_hat + sgn * dist
                dev1 = ex + crit
                grow = (np.sqrt(crit / dev1) if dev1 > 1e-12 else 10.0)
                new_dist = min(max(grow * dist, 1.5 * dist), dist + 1.0)
                dist = min(new_dist, room)
                while True:
                    ex = excess(v_hat + sgn * dist)
                    if ex > 0.0 or dist >= room:
                        break
                    v_in = v_hat + sgn * dist
                    dist = min(1.5 * dist, dist + 1.0, room)
            if ex > 0.0:
                lo_v, hi_v = sorted((v_in, v_hat + sgn * dist))
                end = float(np.exp(brentq(excess, lo_v, hi_v, xtol=1e-6)))
            ends.append(end)
        los[slot] = 0.0 if np.isnan(ends[0]) else ends[0]
        his[slot] = ends[1]
    out["tau_lo"], out["tau_hi"] = los[order], his[order]
    resolved = bool(np.all(np.isfinite(his)) and np.all(los > 0.0))
    out["resolved"] = resolved
    if not resolved:
        warnings.warn(
            "the data do not bound every lifetime: an interval end is open "
            "(NaN above, 0 below); a lifetime much longer than the record "
            "or much shorter than the instrument response cannot be "
            "measured here", RuntimeWarning, stacklevel=2)
    return out
