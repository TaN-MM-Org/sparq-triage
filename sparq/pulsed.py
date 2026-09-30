"""Pulsed-excitation extension of the twin, matched to the sps-quality
quantum-dot measurements (80 MHz repetition, InGaAs/GaAs QD at 1.3 um).

Under pulsed excitation the HBT histogram is a comb of peaks at integer
multiples of the repetition period T_rep. The standard analysis estimates
g2(0) as the area of the tau = 0 peak divided by the mean area of the
side peaks; blinking/memory effects modulate the side-peak envelope as
1 + a exp(-|k| T_rep / tau2), and uncorrelated background adds a flat
floor. Expected counts per bin (the pulsed histogram twin):

    mu(tau) = A [ g2_0 f(tau) + sum_{k!=0} (1 + a e^{-|k|T_rep/tau2})
                 f(tau - k T_rep) ] + B

with f a two-sided exponential of lifetime tau_e convolved with the
pair IRF (Gaussian, sigma_pair), A the side-peak area, and B the flat
background coincidence floor.

Two analyses: the peak-area method (`calibrate_comb`, `g2_peak_area`),
which needs peaks that are narrow compared with the period, and, new in
0.11.0, `fit_pulsed`, a Poisson maximum-likelihood fit of this model
with each peak's share of each bin computed exactly (`peak_cdf`), which
handles overlapping peaks.
"""
from __future__ import annotations
import numpy as np
from scipy.special import erfcx

from .physics import HBTConfig, _exp_conv_gauss

T_REP_NS = 12.5           # 80 MHz


def peak_shape(tau, tau_e, sigma_pair):
    """Normalized (unit-area) two-sided exponential x Gaussian."""
    f = _exp_conv_gauss(tau, tau_e, sigma_pair)   # peak value 1 at center
    return f / (2.0 * tau_e)                       # unit area


def expected_hist_pulsed(cfg: HBTConfig, T_s, rate_cps, g2_0, tau_e,
                         a=0.0, tau2=200.0, bg_frac=0.0, sigma_pair=0.5,
                         center_off=0.0, t_rep=T_REP_NS):
    """Mean counts per bin on the cfg grid for acquisition T_s seconds."""
    r_a = r_b = 0.5 * rate_cps
    A = r_a * r_b * (t_rep * 1e-9) * T_s          # counts per side peak
    tau = cfg.bin_centers - center_off
    kmax = int(np.ceil((cfg.tau_max + abs(center_off)) / t_rep)) + 1
    mu = np.zeros(cfg.n_bins)
    for k in range(-kmax, kmax + 1):
        amp = g2_0 if k == 0 else 1.0 + a * np.exp(-abs(k) * t_rep / tau2)
        mu += amp * peak_shape(tau - k * t_rep, tau_e, sigma_pair)
    mu *= A * cfg.bin_width
    # flat background floor (signal-background and bg-bg coincidences)
    mu += bg_frac * A * cfg.bin_width / t_rep
    return mu


# ----------------------------------------------------------------------
# Comb calibration and conventional peak-area analysis (real + sim)
# ----------------------------------------------------------------------

def calibrate_comb(delay, hist, t_rep=T_REP_NS):
    """Find the comb phase and the suppressed (center) peak position from a
    full accumulation. Returns (center, phase, period)."""
    # comb phase: fold onto the period, take circular max of the folded sum
    nphase = 250
    phases = np.linspace(0, t_rep, nphase, endpoint=False)
    folded = np.zeros(nphase)
    for i, ph in enumerate(phases):
        m = np.abs(((delay - ph + t_rep / 2) % t_rep) - t_rep / 2) < 1.5
        folded[i] = hist[m].mean()
    phase = phases[int(np.argmax(folded))]
    # peak positions across the record
    kmin = int(np.ceil((delay[0] - phase) / t_rep))
    kmax = int(np.floor((delay[-1] - phase) / t_rep))
    pos = phase + np.arange(kmin, kmax + 1) * t_rep
    # peak areas (+- 3 ns windows)
    areas = np.array([hist[(delay >= p - 3) & (delay <= p + 3)].sum()
                      for p in pos])
    # suppressed peak = global minimum of smoothed areas (avoid edges)
    v = areas.copy().astype(float)
    if v.size > 4:
        v[:2] = v[-2:] = np.inf
    else:
        # a short record (new in 0.11.0): up to 0.10.0 every peak was
        # excluded here and the first one was returned; now only peaks
        # whose window is cut by the record's ends are left out
        inside = (pos - 3 >= delay[0]) & (pos + 3 <= delay[-1])
        if inside.any():
            v[~inside] = np.inf
    center = float(pos[int(np.argmin(v))])
    return center, float(phase), t_rep


def g2_peak_area(delay, hist, center, t_rep=T_REP_NS, n_side=6, half=3.0):
    """Conventional pulsed analysis: center-peak area over mean side-peak
    area (background floor subtracted from both).

    The windows are +-`half` ns around each peak, and the floor is the
    median of the bins more than 4.5 ns from every peak. This assumes
    the peaks are narrow compared with the period: when their tails
    fill the space between them (lifetimes of more than about a tenth
    of the period), the "floor" holds peak tails and the result is too
    low, down to 0 (see `fit_pulsed`, which models the overlap).
    Refuses (ValueError, new in 0.11.0) a period that leaves no bins
    4.5 ns from every peak (up to 0.10.0: NaN with NumPy warnings).
    Returns NaN when no side peak lies inside the record or the side
    peaks' mean area is not positive."""
    # background floor from inter-peak regions
    off = np.abs(((delay - center + t_rep / 2) % t_rep) - t_rep / 2)
    if not np.any(off > 4.5):
        raise ValueError(
            f"no delays lie more than 4.5 ns from every peak at t_rep = "
            f"{t_rep:g} ns, so the peak-area method has no floor to "
            "subtract; use fit_pulsed (analyze_pulsed(method='fit'))")
    floor = np.median(hist[off > 4.5])
    def area(p):
        m = (delay >= p - half) & (delay <= p + half)
        return hist[m].sum() - floor * m.sum()
    a0 = area(center)
    sides = []
    for k in range(1, n_side + 1):
        for s in (-1, 1):
            p = center + s * k * t_rep
            if p - half >= delay[0] and p + half <= delay[-1]:
                sides.append(area(p))
    a_side = np.mean(sides) if sides else np.nan
    if not np.isfinite(a_side) or a_side <= 0:
        return np.nan
    return float(max(a0, 0.0) / a_side)


# ----------------------------------------------------------------------
# Exact comb fit (new in 0.11.0)
# ----------------------------------------------------------------------

_SQ2 = np.sqrt(2.0)
G2_MAX_PULSED = 3.0        # upper end of the g2(0) search of `fit_pulsed`


def _peak_lower_tail(y, tau_e, sigma_pair):
    """P[X <= -y] for y >= 0, X distributed as the unit-area peak
    `peak_shape`: a two-sided exponential (Laplace law, scale tau_e)
    plus independent Gaussian jitter (std sigma_pair).

    With b = tau_e, s = sigma_pair, x = -y and Phi the standard normal
    distribution function,
        F(x) = Phi(x/s) - 1/2 e^{s^2/2b^2 - x/b} Phi(x/s - s/b)
                        + 1/2 e^{s^2/2b^2 + x/b} Phi(-x/s - s/b),
    each term written with erfcx(u) = e^{u^2} erfc(u) so that nothing
    overflows or cancels (the same device as the jitter blur in
    `sparq.physics`)."""
    y = np.asarray(y, dtype=float)
    b, s = float(tau_e), float(sigma_pair)
    if s <= 1e-12:
        return 0.5 * np.exp(-y / b)
    g = np.exp(-0.5 * (y / s) ** 2)
    phi = 0.5 * g * erfcx(y / (s * _SQ2))
    t2 = 0.25 * g * erfcx((s / b + y / s) / _SQ2)
    u3 = (s / b - y / s) / _SQ2
    e3 = np.exp(np.minimum(0.5 * (s / b) ** 2 - y / b, 0.0))
    t3 = np.where(u3 >= 0.0, 0.25 * g * erfcx(np.abs(u3)),
                  0.5 * e3 - 0.25 * g * erfcx(np.abs(u3)))
    return phi - t2 + t3


def peak_cdf(x, tau_e, sigma_pair):
    """Distribution function of the unit-area peak `peak_shape`: the
    share of one peak's coincidences at delays below x (ns) from the
    peak center (new in 0.11.0). Closed form; the tests compare it with
    a numerical integral of `peak_shape`."""
    x = np.asarray(x, dtype=float)
    lt = _peak_lower_tail(np.abs(x), tau_e, sigma_pair)
    out = np.where(x < 0.0, lt, 1.0 - lt)
    return float(out) if out.ndim == 0 else out


def _edges_from_centers(delay):
    mid = 0.5 * (delay[1:] + delay[:-1])
    return np.concatenate([[2.0 * delay[0] - mid[0]], mid,
                           [2.0 * delay[-1] - mid[-1]]])


class _CombBasis:
    """Exact bin integrals of the comb model's parts on fixed bin edges:
    the center peak, the side peaks, the side peaks weighted by the
    envelope e^{-|k| t_rep / tau2}, and the flat floor (per t_rep)."""

    def __init__(self, edges, t_rep):
        self.edges = edges
        self.t_rep = float(t_rep)
        self.width = np.diff(edges)
        self._key = None

    def _shares(self, center, tau_e, sigma_pair):
        key = (center, tau_e, sigma_pair)
        if key != self._key:
            # peaks whose tails reach the record (e^-30 of a peak beyond)
            reach = 30.0 * tau_e + 8.0 * sigma_pair
            e = self.edges
            k = np.arange(int(np.floor((e[0] - reach - center) / self.t_rep)),
                          int(np.ceil((e[-1] + reach - center) / self.t_rep))
                          + 1)
            x = e[:, None] - center - k[None, :] * self.t_rep
            G = _peak_lower_tail(np.abs(x), tau_e, sigma_pair)
            lo, hi, g_lo, g_hi = x[:-1], x[1:], G[:-1], G[1:]
            # the share of each peak in each bin, from the smaller tail
            # on each side (bins far from a peak keep their precision)
            self._P = np.where(lo >= 0.0, g_lo - g_hi,
                               np.where(hi <= 0.0, g_hi - g_lo,
                                        1.0 - g_lo - g_hi))
            self._k = k
            self._key = key
        return self._P, self._k

    def columns(self, center, tau_e, sigma_pair, tau2=None):
        P, k = self._shares(center, tau_e, sigma_pair)
        side = (k != 0).astype(float)
        cols = [P @ (k == 0).astype(float), P @ side]
        if tau2 is not None:
            cols.append(P @ (side * np.exp(-np.abs(k) * self.t_rep / tau2)))
        cols.append(self.width / self.t_rep)
        return np.column_stack(cols)


def _poisson_nonneg(X, h, beta0, lower, prior=None):
    """Minimize sum(mu - h log mu) over beta >= lower, mu = X beta (a
    convex problem), by projected Newton steps with backtracking.

    prior=(i, m, sd): adds 0.5 ((beta_i - m) / sd)^2. Returns
    (objective, beta) with the objective written as half the Poisson
    deviance (zero for a perfect fit) plus the prior term."""
    lower = np.asarray(lower, dtype=float)
    pos = h > 0
    hp = h[pos]
    hsum = float(h.sum())

    def f(b):
        mu = X @ b
        if np.any(mu <= 0.0) or not np.all(np.isfinite(mu)):
            return np.inf, mu
        v = float(np.sum(mu) - hsum - np.sum(hp * np.log(mu[pos] / hp)))
        if prior is not None:
            v += 0.5 * ((b[prior[0]] - prior[1]) / prior[2]) ** 2
        return v, mu

    b = np.maximum(np.asarray(beta0, dtype=float), lower)
    fb, mu = f(b)
    if not np.isfinite(fb):
        b = lower + np.maximum(np.abs(b), 1.0)
        fb, mu = f(b)
    colsum = X.sum(0)
    for _ in range(200):
        with np.errstate(over="ignore", invalid="ignore"):
            r = h / mu
            g = colsum - X.T @ r
            H = (X * (r / mu)[:, None]).T @ X
        if not (np.all(np.isfinite(H)) and np.all(np.isfinite(g))):
            break                   # a hopeless shape; keep this point
        if prior is not None:
            i, m, sd = prior
            g[i] += (b[i] - m) / sd ** 2
            H[i, i] += 1.0 / sd ** 2
        free = ~((b <= lower) & (g > 0.0))
        if not free.any():
            break
        Hf = H[np.ix_(free, free)]
        Hf = Hf + np.eye(Hf.shape[0]) * 1e-12 * max(np.trace(Hf), 1e-300)
        d = np.zeros_like(b)
        try:
            d[free] = -np.linalg.solve(Hf, g[free])
        except np.linalg.LinAlgError:
            d[free] = -g[free] / np.maximum(np.diag(Hf), 1e-300)
        t = 1.0
        while True:
            bn = np.maximum(b + t * d, lower)
            fn, mun = f(bn)
            if fn <= fb or t < 1e-12:
                break
            t *= 0.5
        if not (np.isfinite(fn) and fn <= fb):
            break
        done = fb - fn <= 1e-13 * (1.0 + abs(fb))
        b, fb, mu = bn, fn, mun
        if done:
            break
    return fb, b


_MU_ASYMPTOTIC = 1000.0      # above this, the large-mean expansion
_MOMENT_BLOCK = 200_000     # summation cells per block (memory bound)


def _deviance_moments(mu):
    """Mean and variance of each bin's Poisson deviance term
    2 [m - k + k log(k/m)], k ~ Poisson(m).

    For m <= 1000 they are summed exactly over k in a window of
    +-(12 sqrt(m) + 20) around m (all but a negligible share of the
    probability). The bins are sorted by m and summed in blocks of at
    most about 200 000 cells, each block with the window of its own
    largest m, so memory stays bounded for any number of bins (0.11.0
    as first written used the window of the largest bin for all bins,
    over 1 GB for 4096 bins with one bin at 1e5 counts). For m > 1000
    the expansions
        mean = 1 + 1/(6m) + 1/(6m^2),  var = 2 + 2/(3m) + 4/(3m^2)
    are used; their coefficients were read off the exact sums, and the
    tests check them against the exact sums to 1e-8 for m from 1000 to
    1e5."""
    from scipy.stats import poisson
    m = np.maximum(np.asarray(mu, dtype=float).ravel(), 1e-300)
    mean = np.empty(m.size)
    var = np.empty(m.size)
    big = m > _MU_ASYMPTOTIC
    mb = m[big]
    mean[big] = 1.0 + 1.0 / (6.0 * mb) + 1.0 / (6.0 * mb * mb)
    var[big] = 2.0 + 2.0 / (3.0 * mb) + 4.0 / (3.0 * mb * mb)
    idx = np.nonzero(~big)[0]
    idx = idx[np.argsort(m[idx], kind="stable")]
    Js = np.ceil(12.0 * np.sqrt(m[idx]) + 20.0).astype(np.int64)
    i = 0
    while i < idx.size:
        # the block i..j-1 uses the window of its last (largest) bin;
        # cells (j - i) (2 J + 1) grow with j, so the largest j that
        # keeps them within the bound is found by bisection
        cells = np.arange(1, idx.size - i + 1) * (2 * Js[i:] + 1)
        j = i + max(1, int(np.searchsorted(cells, _MOMENT_BLOCK,
                                           side="right")))
        J = int(Js[j - 1])
        sel = idx[i:j]
        mm = m[sel][:, None]
        k = np.floor(mm) + np.arange(-J, J + 1)[None, :]
        valid = k >= 0
        kk = np.maximum(k, 0.0)
        pk = np.where(valid, poisson.pmf(kk, mm), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl = np.where(kk > 0, kk * np.log(np.maximum(kk, 1.0) / mm),
                          0.0)
        d = 2.0 * (mm - kk + kl)
        e1 = np.sum(pk * d, axis=1)
        mean[sel] = e1
        var[sel] = np.sum(pk * d * d, axis=1) - e1 ** 2
        i = j
    shape = np.shape(mu)
    return mean.reshape(shape), np.maximum(var, 0.0).reshape(shape)


def _deviance_p_value(dev, mu, n_free):
    """Goodness-of-fit p-value of a Poisson deviance whose bins may hold
    few counts: the deviance is compared with a scaled chi-square law
    s chi2(nu) whose mean and variance match the exact ones of the
    fitted means (the mean lowered by the number of fitted parameters,
    as for the usual chi-square with bins - parameters degrees of
    freedom). For bins with many counts this is the usual chi-square
    test; for bins with few counts the chi-square law with bins -
    parameters degrees of freedom does not hold."""
    from scipy.stats import chi2
    mean, var = _deviance_moments(mu)
    M = float(mean.sum()) - n_free
    V = float(var.sum())
    if not (M > 0 and V > 0):
        return np.nan
    scale = V / (2.0 * M)
    nu = 2.0 * M * M / V
    return float(chi2.sf(dev / scale, nu))


def fit_pulsed(delay, counts, t_rep=T_REP_NS, center=None, sigma_pair=None,
               envelope=False, T_s=None, singles_cps=None,
               singles_rel_sd=0.01, level=0.95, interval=True):
    """g2(0) of a pulsed HBT histogram by an exact fit of the whole comb
    of peaks, with a profile-likelihood interval (new in 0.11.0).

    The conventional peak-area method (`g2_peak_area`) adds up counts in
    fixed windows and subtracts a floor measured between the peaks.
    When the peaks are not narrow compared with the period, their tails
    fill the space between them, so that floor is not a floor, and the
    tails also spill across the windows; g2(0) then comes out too low,
    down to 0 (noise-free data, 80 MHz, 2 ns lifetime, true g2(0)
    0.12). This fit models the overlap instead.

    Model (the one `expected_hist_pulsed` simulates): the expected
    counts in a bin are
        A [g2_0 P_0 + sum_{k != 0} (1 + a e^{-|k| t_rep/tau2}) P_k]
          + bg_frac A w / t_rep,
    with P_k the share of peak k (at center + k t_rep) that falls in the
    bin, computed exactly from the peak's distribution function
    (`peak_cdf`: a two-sided exponential of lifetime tau_e blurred by
    Gaussian jitter sigma_pair), A the area of an uncorrelated side
    peak, w the bin width and bg_frac a flat floor from uncorrelated
    background (in units of A per period). The counts are Poisson; the
    fit maximizes their likelihood. A, A g2_0, A a and the floor enter
    linearly and are solved exactly for each peak shape; the center,
    tau_e, sigma_pair (and tau2) are searched.

    delay, counts : bin centers (ns; increasing or decreasing, spanning
        at least three periods) and raw counts.
    t_rep : repetition period (ns).
    center : the zero-delay peak position (ns), if known; by default it
        is located with `calibrate_comb` and then fitted within
        +-t_rep/4.
    sigma_pair : the jitter of the two detectors together (ns), if
        known; fitted otherwise.
    envelope : False (default): all side peaks have the same area A,
        and g2_0 is the center area over it. True: the side peaks carry
        the blinking envelope 1 + a e^{-|k| t_rep/tau2}, and g2_0 is
        relative to the far (uncorrelated) level A. A histogram alone
        pins that level only if the envelope decays within the record;
        give T_s and singles_cps to pin it (below). Otherwise the upper
        end of the interval is often unbounded (NaN).
    T_s, singles_cps, singles_rel_sd : acquisition time (s) and the two
        detectors' count rates (counts/s; one number is taken as their
        sum, split evenly). With both, A is constrained to
        r_a r_b t_rep T_s (the side-peak area of uncorrelated clicks),
        to a relative standard deviation singles_rel_sd.
    level : confidence level of the interval.
    interval : False skips the interval (it needs about ten more fits).

    Returns a dict: g2_0; lo and hi, the interval where the profile
    deviance 2 [NLL(g2_0) - NLL_min] stays below the chi-square(1)
    quantile of `level` (Wilks), lo = 0 when the interval reaches 0 and
    hi = NaN when it does not close below G2_MAX_PULSED = 3; level;
    params (center, tau_e, sigma_pair, area, bg_frac, a, tau2); mu (the
    fitted mean counts); deviance, dof and p_value (chi-square tail of
    the Poisson deviance: a small value says the model does not
    describe the peaks, for example a center peak of another shape);
    at_bound (searched parameters that ended on a search limit); ok.
    """
    from scipy.optimize import brentq, minimize
    from scipy.stats import chi2

    delay = np.asarray(delay, dtype=float)
    h = np.asarray(counts, dtype=float)
    if delay.ndim != 1 or h.shape != delay.shape or delay.size < 5:
        raise ValueError("delay and counts must be 1-D arrays of equal "
                         "length with at least 5 bins")
    if not (np.all(np.isfinite(delay)) and np.all(np.isfinite(h))):
        raise ValueError("delay and counts must be finite")
    if np.any(h < 0):
        raise ValueError("counts must be non-negative")
    d = np.diff(delay)
    if np.all(d < 0):
        delay, h = delay[::-1], h[::-1]
    elif np.any(d <= 0):
        raise ValueError("delays must be strictly increasing (or "
                         "decreasing)")
    T = float(t_rep)
    if not (np.isfinite(T) and T > 0):
        raise ValueError("t_rep must be positive (ns)")
    edges = _edges_from_centers(delay)
    if edges[-1] - edges[0] < 3.0 * T:
        raise ValueError("the record must span at least three periods "
                         "t_rep (the center peak and side peaks)")
    if not (0.0 < level < 1.0):
        raise ValueError("level must lie in (0, 1)")
    if h.sum() <= 0:
        raise ValueError("the histogram has no counts")
    if sigma_pair is not None and not (np.isfinite(sigma_pair)
                                       and sigma_pair > 0):
        raise ValueError("sigma_pair must be positive (ns)")
    prior_A = None
    if (T_s is None) != (singles_cps is None):
        raise ValueError("give both T_s and singles_cps, or neither")
    if T_s is not None:
        sc = np.atleast_1d(np.asarray(singles_cps, dtype=float))
        if sc.size == 1:
            sc = np.array([0.5 * sc[0], 0.5 * sc[0]])
        if sc.size != 2 or not np.all(np.isfinite(sc)) or np.any(sc <= 0):
            raise ValueError("singles_cps must be (r_a, r_b) or their sum, "
                             "positive counts per second")
        if not (np.isfinite(T_s) and T_s > 0):
            raise ValueError("T_s must be positive (s)")
        if not (np.isfinite(singles_rel_sd) and singles_rel_sd > 0):
            raise ValueError("singles_rel_sd must be positive")
        A_s = float(sc[0] * sc[1] * T * 1e-9 * T_s)
        prior_A = (A_s, float(singles_rel_sd) * A_s)

    if center is None:
        c0, _, _ = calibrate_comb(delay, h, t_rep=T)
    else:
        c0 = float(center)
    basis = _CombBasis(edges, T)
    env = bool(envelope)

    # searched parameters: center (unless given), log tau_e,
    # log sigma_pair (unless given), log tau2 (with the envelope)
    names, lo_b, hi_b = [], [], []
    if center is None:
        names.append("center")
        lo_b.append(c0 - T / 4.0)
        hi_b.append(c0 + T / 4.0)
    names.append("tau_e")
    lo_b.append(np.log(1e-3))
    hi_b.append(np.log(T))
    if sigma_pair is None:
        names.append("sigma_pair")
        lo_b.append(np.log(1e-3))
        hi_b.append(np.log(T / 2.0))
    if env:
        names.append("tau2")
        lo_b.append(np.log(0.1 * T))
        hi_b.append(np.log(1e3 * T))
    lo_b, hi_b = np.array(lo_b), np.array(hi_b)
    bnds = list(zip(lo_b, hi_b))

    def shape_of(th):
        p = dict(zip(names, th))
        c = p.get("center", c0)
        te = float(np.exp(p["tau_e"]))
        sp = float(np.exp(p["sigma_pair"])) if "sigma_pair" in p \
            else float(sigma_pair)
        t2 = float(np.exp(p["tau2"])) if env else None
        return c, te, sp, t2

    # linear coefficients: (A g2_0, A, [A a], A bg_frac) for the free
    # fit; (A, [A a], A bg_frac) at a fixed g2_0 (profile)
    n_lin = 4 if env else 3
    A_guess = max(h.sum() / ((edges[-1] - edges[0]) / T), 1.0)
    state = {"beta": np.r_[0.3 * A_guess, A_guess,
                           [0.1 * A_guess] if env else [], 0.05 * A_guess]}

    def objective(th, g=None, keep=None):
        X = basis.columns(*shape_of(th))
        if g is None:
            lower = np.r_[0.0, 1e-12, np.zeros(n_lin - 2)]
            pr = None if prior_A is None else (1, prior_A[0], prior_A[1])
            v, b = _poisson_nonneg(X, h, state["beta"], lower, pr)
            if np.isfinite(v):
                state["beta"] = b           # warm start for the next call
        else:
            Xg = np.column_stack([g * X[:, 0] + X[:, 1], X[:, 2:]])
            lower = np.r_[1e-12, np.zeros(n_lin - 2)]
            pr = None if prior_A is None else (0, prior_A[0], prior_A[1])
            v, b = _poisson_nonneg(Xg, h, state["beta"][1:], lower, pr)
            b = np.r_[g * b[0], b]
            if np.isfinite(v):
                state["beta"] = b
        if keep is not None:
            keep["beta"] = b
        return v

    # starts: a coarse scan of the peak shape (lifetime, and jitter when
    # it is searched), then a bounded quasi-Newton search from the two
    # best points of the scan
    scan = []
    for te0 in np.geomspace(0.005 * T, 0.5 * T, 9):
        for sp0 in ((0.01 * T, 0.03 * T, 0.1 * T) if sigma_pair is None
                    else (None,)):
            x0 = []
            for n in names:
                x0.append({"center": c0, "tau_e": np.log(te0),
                           "sigma_pair": np.log(sp0) if sp0 else 0.0,
                           "tau2": np.log(3.0 * T)}[n])
            x0 = np.clip(x0, lo_b, hi_b)
            scan.append((objective(x0), x0))
    scan.sort(key=lambda t: t[0])
    best = None
    for _, x0 in scan[:2]:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bnds)
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return dict(ok=False, g2_0=np.nan, lo=np.nan, hi=np.nan)
    keep = {}
    f_min = objective(best.x, keep=keep)
    beta = keep["beta"]
    state["beta"] = beta.copy()
    A = float(beta[1])
    g_hat = float(beta[0] / A)
    c, te, sp, t2 = shape_of(best.x)
    X = basis.columns(c, te, sp, t2)
    mu = X @ beta
    pos = h > 0
    dev = float(2.0 * (np.sum(mu - h)
                       - np.sum(h[pos] * np.log(mu[pos] / h[pos]))))
    n_free = n_lin + len(names)
    dof = int(h.size - n_free)
    at_bound = [n + ("_low" if abs(x - l) < 1e-3 else "_high")
                for n, x, l, u in zip(names, best.x, lo_b, hi_b)
                if abs(x - l) < 1e-3 or abs(x - u) < 1e-3]
    out = dict(ok=True, g2_0=g_hat, lo=np.nan, hi=np.nan, level=float(level),
               params=dict(center=float(c), tau_e=te, sigma_pair=sp, area=A,
                           bg_frac=float(beta[-1] / A),
                           a=float(beta[2] / A) if env else 0.0,
                           tau2=t2),
               mu=mu, deviance=dev, dof=dof,
               p_value=_deviance_p_value(dev, mu, n_free) if dof > 0
               else np.nan,
               at_bound=at_bound)
    if not interval:
        return out

    crit = float(chi2.ppf(level, 1))
    warm = {"x": best.x.copy()}

    def excess(g):
        state["beta"] = beta.copy()
        r = minimize(lambda th: objective(th, g), warm["x"],
                     method="L-BFGS-B", bounds=bnds)
        return 2.0 * (min(r.fun, np.inf) - f_min) - crit

    if g_hat <= 0.0 or excess(0.0) <= 0.0:
        lo = 0.0
    else:
        lo = float(brentq(excess, 0.0, g_hat, xtol=1e-4))
    step = max(0.02, 0.5 * g_hat)
    g_up, hi = g_hat, np.nan
    while g_up < G2_MAX_PULSED:
        g_next = min(g_up + step, G2_MAX_PULSED)
        if excess(g_next) > 0.0:
            hi = float(brentq(excess, g_up, g_next, xtol=1e-4))
            break
        g_up = g_next
        step *= 2.0
    out.update(lo=lo, hi=hi)
    return out
