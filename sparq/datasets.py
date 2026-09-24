"""Synthetic acquisition generators (the twin as a training environment)
and the loader for the real experimental HBT dataset (UTS-CASLab
sps-quality, FI-SEQUR InGaAs/GaAs quantum dot; Kedziora et al., MLST 2023).
"""
from __future__ import annotations
import glob
import os
import numpy as np

from .physics import (HBTConfig, EmitterSite, PLATFORMS, sample_site,
                      mean_detected_rate_cps,
                      expected_histogram, g2_zero)

CFG = HBTConfig(tau_max=60.5, n_bins=121, sigma_irf=0.35)
N_SLICES = 32
N_AUX = 5


def make_batch(rng, batch, T_dist=("logu", 0.03, 30.0), platform="NV",
               cfg=CFG, n_slices=N_SLICES, sites=None, T_fixed=None,
               n_probs=(0.42, 0.30, 0.18, 0.10)):
    """Sample a batch of synthetic acquisitions.

    Returns dict with:
      stream  [B, S, K]  sliced coincidence counts (SNN input)
      hist    [B, K]     integrated histogram (CNN/fit input)
      aux     [B, 5]     log10 T, log10 singles-rate estimate, log10(1 + counts),
                         log10(1 + expected flat total), log10(1 + central counts)
      y_cls   [B]        1 if physical g2(0) < 0.5
      y_g2    [B]        physical g2(0)
      T       [B]        acquisition times (s)
    """
    S, K = n_slices, cfg.n_bins
    stream = np.zeros((batch, S, K), np.float32)
    aux = np.zeros((batch, N_AUX), np.float32)
    y_cls = np.zeros(batch, np.int64)
    y_g2 = np.zeros(batch, np.float32)
    Ts = np.zeros(batch, np.float32)
    site_list = []
    for i in range(batch):
        site = sites[i] if sites is not None else sample_site(
            rng, platform, n_probs)
        if T_fixed is not None:
            T = float(T_fixed)
        else:
            kind, lo, hi = T_dist
            T = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        mu = np.maximum(expected_histogram(site, T, cfg), 0.0) / S
        stream[i] = rng.poisson(np.broadcast_to(mu, (S, K)))
        # observable singles rate (Poisson-sampled counter reading)
        n_singles = rng.poisson(mean_detected_rate_cps(site) * T)
        r_hat = max(n_singles / T, 1.0)
        tot = stream[i].sum()
        # expected flat-level total (from the measured singles rate) and
        # central-window counts: the near-sufficient statistics of the
        # antibunching decision at low counts
        exp_flat = (0.5 * r_hat) ** 2 * (cfg.bin_width * 1e-9) * T \
            * cfg.n_bins
        central = stream[i][:, np.abs(cfg.bin_centers) < 12.0].sum()
        aux[i] = (np.log10(T), np.log10(r_hat), np.log10(1.0 + tot),
                  np.log10(1.0 + exp_flat), np.log10(1.0 + central))
        y_g2[i] = site.g2_0
        y_cls[i] = 1 if site.g2_0 < 0.5 else 0
        Ts[i] = T
        site_list.append(site)
    # classification margin: boundary sites (0.4 < g2(0) < 0.6) are
    # excluded from the classification metric/loss (regression covers
    # them) — triage decides clear cases; boundary cases need precision
    # metrology, not classification.
    y_valid = (np.abs(y_g2 - 0.5) > 0.1)
    y_good = np.array([1 if s.is_good else 0 for s in site_list], np.int64)
    return dict(stream=stream, hist=stream.sum(1), aux=aux, y_cls=y_cls,
                y_g2=y_g2, T=Ts, sites=site_list, y_valid=y_valid,
                y_good=y_good)


def make_eval_set(rng, n, T_s, platform="NV", cfg=CFG, n_slices=N_SLICES,
                  sites=None):
    return make_batch(rng, n, platform=platform, cfg=cfg, n_slices=n_slices,
                      sites=sites, T_fixed=T_s)


# ----------------------------------------------------------------------
# Real experimental data (sps-quality, FI-SEQUR demonstrator sample)
# ----------------------------------------------------------------------

def load_fisequr(path_dir):
    """Load the eight FI-SEQUR HBT measurement series.

    path_dir: the directory holding the dataset's .txt files (the
    sps-quality release of Kedziora et al., MLST 2023) -- there is no
    default, because the data lives wherever YOU downloaded it.
    Each file: rows = delay bins, first column = delay (ns), remaining
    columns = coincidence counts of successive 10-s snapshots.
    Returns list of dicts with delay axis, snapshot matrix, and metadata.
    """
    if not path_dir or not os.path.isdir(path_dir):
        raise ValueError(
            "load_fisequr needs the directory of the downloaded "
            "sps-quality dataset; got %r" % (path_dir,))
    series = {}
    for f in sorted(glob.glob(os.path.join(path_dir, "*.txt"))):
        raw = np.loadtxt(f)
        delay, counts = raw[:, 0], raw[:, 1:]
        name = os.path.basename(f).replace(".txt", "")
        # merge multi-part measurements of the same physical series
        key = name.split("_part")[0]
        if key in series:
            series[key]["counts"] = np.concatenate(
                [series[key]["counts"], counts], axis=1)
        else:
            series[key] = dict(delay=delay, counts=counts, name=key)
    out = list(series.values())
    for s in out:
        s["snapshot_s"] = 10.0
        s["total"] = s["counts"].sum(1)
        s["T_total"] = s["counts"].shape[1] * 10.0
    return out


def save_hbt_csv(path, delay_ns, counts):
    """Write a measured HBT histogram in the documented two-column
    contract: header exactly ``delay_ns,counts``, one row per bin.
    The round trip through `load_hbt_csv` is exact (asserted in the
    tests)."""
    d = np.asarray(delay_ns, dtype=float).ravel()
    c = np.asarray(counts, dtype=float).ravel()
    if d.size != c.size or d.size < 5:
        raise ValueError("delay_ns and counts must be equal-length "
                         "arrays with at least 5 bins")
    with open(path, "w") as fh:
        fh.write("delay_ns,counts\n")
        for a, b in zip(d, c):
            fh.write(f"{float(a)!r},{float(b)!r}\n")


def load_hbt_csv(path):
    """Read a measured HBT histogram in the documented contract
    (header exactly ``delay_ns,counts``); returns (delay_ns, counts)
    ready for `analyze_histogram` / `rebin_real`.

    Refusals instead of guesses: a wrong header, rows without exactly
    two fields, non-finite values, negative counts, a non-increasing
    delay axis, or fewer than 5 bins all raise with an explanation.
    """
    with open(path) as fh:
        header = fh.readline().strip()
        if header != "delay_ns,counts":
            raise ValueError(
                "histogram file header must be exactly 'delay_ns,counts'; "
                f"got {header!r}")
        d, c = [], []
        for lineno, line in enumerate(fh, start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                raise ValueError(f"line {lineno}: expected 2 fields")
            d.append(float(parts[0]))
            c.append(float(parts[1]))
    d = np.asarray(d, dtype=float)
    c = np.asarray(c, dtype=float)
    if d.size < 5:
        raise ValueError("histogram has fewer than 5 bins")
    if not (np.all(np.isfinite(d)) and np.all(np.isfinite(c))):
        raise ValueError("histogram contains non-finite values")
    if np.any(c < 0):
        raise ValueError("counts must be non-negative")
    if np.any(np.diff(d) <= 0):
        raise ValueError("delay axis must be strictly increasing")
    return d, c


def robust_flat_rate(hist, cfg, T_s, lo_frac=0.65):
    """Effective singles rate from a trimmed median of far-delay bins
    (robust to real detector artifacts such as echo peaks)."""
    m = np.abs(cfg.bin_centers) >= lo_frac * cfg.tau_max
    flat = float(np.median(hist[m]))
    return 2.0 * np.sqrt(max(flat, 1e-9) / (cfg.bin_width * 1e-9 * T_s))


def rebin_real(delay, hist, cfg=CFG, center=None, method="split",
               return_variance=False):
    """Re-bin a real histogram onto the analysis grid `cfg`, centered on
    the antibunching dip (or on `center`, in ns, when given).

    With center=None the dip is located: for method="split" as the most
    significant local minimum over a range of widths (`locate_dip`),
    then refined
    to a fraction of a bin by symmetry (`_refine_center`); for "whole"
    by the heavily smoothed minimum used up to 0.9.1.

    The analysis window must lie inside the data (new in 0.10.0): a grid
    bin beyond the recorded delays would be left empty or partly
    empty, silently. When the data do not reach tau_max on both sides
    of the dip, this is refused with ValueError naming the dip position
    and the largest tau_max that fits; pass a cfg with a smaller
    tau_max (keeping n_bins odd). A located center within a quarter of
    an input bin of the range that fits is moved onto it.

    method="split" (the default from 0.10.0): each input bin is taken to
    hold its counts spread evenly over its width, and is shared between
    the grid bins it overlaps in proportion to the overlap. Total counts
    are kept exactly; a flat input gives a flat output for any ratio of
    bin widths, and the zero-delay grid bin is centered on `center`.
    The input bin edges are the midpoints between neighbouring delays
    (the delays are taken to be bin centers), so the grid need not be
    uniform. A split histogram is no longer made of independent
    Poisson counts: neighbouring grid bins share input bins, and a
    grid bin's counts vary less than Poisson counts of the same mean.
    The bootstrap of `analyze_histogram` redraws the INPUT counts, so
    its interval includes this.

    method="whole" (the only method up to 0.9.1): each input bin goes
    whole into the grid bin that contains its delay. If the grid bin
    width is not a whole multiple of the input width, grid bins get
    unequal numbers of input bins (a ripple), and even for a whole
    multiple the zero-delay bin can be off-center by half an input bin.

    return_variance=True (new in 0.10.0) adds a third output: the
    Poisson variance of each grid bin, sum_j f_kj^2 h_j for the shares
    f_kj of input bins (equal to the counts for "whole"). The fits of
    `analyze_histogram` use it as their error bars.

    Either way the input bins must not be wider than the grid bins: a
    coarser input would leave some grid bins empty (whole) or invent
    resolution the data do not have (split), and is refused with
    ValueError (pass a coarser `cfg` instead).
    """
    delay = np.asarray(delay, dtype=float)
    hist = np.asarray(hist, dtype=float)
    if method not in ("split", "whole"):
        raise ValueError("method must be 'split' or 'whole'")
    step = np.abs(np.diff(delay))
    if step.size and step.max() > cfg.bin_width * (1.0 + 1e-6):
        raise ValueError(
            f"input delay bins ({step.max():.4g} ns) are coarser than the "
            f"analysis grid's {cfg.bin_width:.4g} ns bins, so some grid "
            "bins would stay empty; pass a cfg whose bin width is at "
            "least the input bin width")
    if method == "split":
        if delay.size < 2:
            raise ValueError("need at least two delay bins")
        d = np.diff(delay)
        if np.all(d < 0):                    # a descending axis is fine
            delay, hist = delay[::-1], hist[::-1]
        elif np.any(d <= 0):
            raise ValueError("method='split' needs the delays in strictly "
                             "increasing (or decreasing) order")
        in_edges = _input_edges(delay)
    else:
        order = np.sort(delay)
        half = 0.5 * float(np.median(np.diff(order))) if delay.size > 1 \
            else 0.0
        in_edges = np.array([order[0] - half, order[-1] + half])
    tol = 1e-6 * cfg.bin_width
    auto_located = False
    lo_c = in_edges[0] + cfg.tau_max
    hi_c = in_edges[-1] - cfg.tau_max
    if center is None:
        if method == "whole":
            # robust dip locate: heavily smoothed minimum (0.9.1)
            k = 41
            sm = np.convolve(hist, np.ones(k) / k, mode="same")
            m = len(hist)
            center = delay[np.argmin(sm[m // 10: 9 * m // 10]) + m // 10]
        else:
            c0, width = _locate_dip(delay, hist)
            center = _refine_center(delay, hist, cfg, c0, width)
            auto_located = True
            # a center within a quarter input bin of the range whose
            # window fits (e.g. a record exactly as wide as the window)
            # is moved onto that range; farther out it is refused below
            q = 0.25 * float(np.median(np.diff(delay)))
            if lo_c - q <= center < lo_c:
                center = lo_c
            elif hi_c < center <= hi_c + q:
                center = hi_c
    try:
        _check_window(float(center), lo_c, hi_c, tol, cfg, in_edges)
    except ValueError as err:
        if method == "split" and auto_located and \
                locate_dip(delay, hist)["ambiguous"]:
            raise ValueError(
                str(err) + ". The dip is not clearly above the noise "
                "here (see sparq.locate_dip), so this position may be "
                "wrong: pass `center` if you know where the dip is"
            ) from None
        raise
    edges = np.linspace(-cfg.tau_max, cfg.tau_max, cfg.n_bins + 1) + center
    if method == "whole":
        idx = np.searchsorted(edges, delay)
        out = np.zeros(cfg.n_bins, np.float32)
        for b in range(cfg.n_bins):
            out[b] = hist[(idx == b + 1)].sum()
        if return_variance:
            return out, center, out.astype(float)
        return out, center
    out = _split(delay, hist, edges)
    if return_variance:
        _, var = _split_moments(delay, hist, edges)
        return out, center, var
    return out, center


def _input_edges(delay):
    """Bin edges of increasing bin centers: midpoints, and half a step
    beyond the first and last."""
    mid = 0.5 * (delay[1:] + delay[:-1])
    return np.concatenate([[delay[0] - (mid[0] - delay[0])], mid,
                           [delay[-1] + (delay[-1] - mid[-1])]])


def _check_window(center, lo_c, hi_c, tol, cfg, in_edges):
    if not (lo_c - tol <= center <= hi_c + tol):
        room = min(center - in_edges[0], in_edges[-1] - center)
        raise ValueError(
            f"the dip is at {center:.4g} ns, and the analysis window "
            f"+-{cfg.tau_max:.4g} ns around it reaches beyond the data "
            f"({in_edges[0]:.4g} to {in_edges[-1]:.4g} ns), which would "
            "leave grid bins empty; pass a cfg with tau_max at most "
            f"{max(room, 0.0):.4g} ns (n_bins odd, bin width unchanged)")


def _split(delay, hist, edges):
    """Overlap-weighted re-binning of (delay centers, counts) onto edges."""
    if delay.size < 2 or np.any(np.diff(delay) <= 0):
        raise ValueError("method='split' needs at least two delays in "
                         "strictly increasing order")
    in_edges = _input_edges(delay)
    cum = np.concatenate([[0.0], np.cumsum(hist)])
    # counts below x, with each input bin's counts spread evenly
    F = np.interp(edges, in_edges, cum)
    return np.diff(F).astype(np.float32)


def _split_moments(delay, hist, edges):
    """Split re-binning and its Poisson variance: (o, v) with
    o_k = sum_j f_kj h_j and v_k = sum_j f_kj^2 h_j, f_kj the share of
    input bin j that falls in grid bin k (grid edges uniform, input bins
    no wider than grid bins, so each input bin meets at most two)."""
    mid = 0.5 * (delay[1:] + delay[:-1])
    L = np.concatenate([[delay[0] - (mid[0] - delay[0])], mid])
    R = np.concatenate([mid, [delay[-1] + (delay[-1] - mid[-1])]])
    W = R - L
    g0, w = edges[0], edges[1] - edges[0]
    K = edges.size - 1
    iL = np.floor((L - g0) / w).astype(np.int64)
    iR = np.floor((R - g0) / w).astype(np.int64)
    cut = g0 + iR * w
    fl = np.where(iL == iR, 1.0, (cut - L) / W)
    fr = 1.0 - fl
    o = np.zeros(K)
    v = np.zeros(K)
    for idx, f in ((iL, fl), (iR, fr)):
        ok = (idx >= 0) & (idx < K) & (f > 0)
        o += np.bincount(idx[ok], weights=f[ok] * hist[ok], minlength=K)
        v += np.bincount(idx[ok], weights=f[ok] ** 2 * hist[ok],
                         minlength=K)
    return o, v


def locate_dip(delay, counts):
    """Where is the antibunching dip, and how clearly? (new in 0.10.0)

    For each half-width W (one input bin, doubling, up to an eighth of
    the recorded span) and each input bin, the count rate within +-W is
    compared separately with the rate between W and 3W on the left and
    on the right, each difference in units of its Poisson standard
    error. A dip must lie below BOTH sides, so its score z is the less
    negative of the two; the flank of a bunching peak (below one side,
    above the other) scores nothing, and a noise dip in a flat stretch
    scores only a few standard errors. Only windows whose two sides lie
    inside the record are scored.

    Returns dict(center, width, z, z_other, ambiguous): the position and
    half-width with the most negative z; z_other, the most negative z
    at least 3 half-widths away from it; and ambiguous = True when that
    other place is at least 0.6 times as significant, or the best
    score is weaker than 5 standard errors -- the automatic dip
    position is then not to be trusted (pass `center`). center is None
    when no window fits the record.
    """
    delay = np.asarray(delay, dtype=float)
    h = np.asarray(counts, dtype=float)
    if delay.size > 1 and np.all(np.diff(delay) < 0):
        delay, h = delay[::-1], h[::-1]
    e = _input_edges(delay)
    cum = np.concatenate([[0.0], np.cumsum(h)])

    def cnt(lo, hi):                     # counts between delays lo and hi
        return np.interp(hi, e, cum) - np.interp(lo, e, cum), hi - lo

    step = float(np.median(np.diff(delay)))
    span = e[-1] - e[0]
    scores = []
    W = step
    while W <= span / 8.0:
        c = delay
        inside = (c - 3 * W >= e[0]) & (c + 3 * W <= e[-1])
        if inside.any():
            c = c[inside]
            n_in, w_in = cnt(c - W, c + W)
            z_sides = []
            for lo, hi in ((c - 3 * W, c - W), (c + W, c + 3 * W)):
                n_s, w_s = cnt(lo, hi)
                se = np.sqrt(np.maximum(n_in, 1.0) / w_in ** 2
                             + np.maximum(n_s, 1.0) / w_s ** 2)
                z_sides.append((n_in / w_in - n_s / w_s) / se)
            scores.append((W, c, np.maximum(z_sides[0], z_sides[1])))
        W *= 2.0
    if not scores:
        return dict(center=None, width=None, z=None, z_other=None,
                    ambiguous=True)
    z1, c1, W1 = np.inf, None, None
    for W, c, z in scores:
        i = int(np.argmin(z))
        if z[i] < z1:
            z1, c1, W1 = float(z[i]), float(c[i]), W
    z2 = np.inf
    for W, c, z in scores:
        far = np.abs(c - c1) > 3.0 * max(W, W1)
        if far.any():
            z2 = min(z2, float(z[far].min()))
    ambiguous = bool(z1 > -5.0 or (np.isfinite(z2) and z2 < 0
                                   and z2 <= 0.6 * z1))
    return dict(center=c1, width=W1, z=z1,
                z_other=z2 if np.isfinite(z2) else None,
                ambiguous=ambiguous)


def _locate_dip(delay, hist):
    """(center, W) from `locate_dip`, falling back to the heavily
    smoothed minimum of 0.9.1 when no window fits the record."""
    d = locate_dip(delay, hist)
    if d["center"] is not None:
        return d["center"], d["width"]
    step = float(np.median(np.diff(delay)))
    k = min(41, len(hist))
    sm = np.convolve(hist, np.ones(k) / k, mode="same")
    m = len(hist)
    j = int(np.argmin(sm[m // 10: 9 * m // 10])) + m // 10 \
        if m >= 10 else int(np.argmin(sm))
    return float(delay[j]), 20.0 * step


def _refine_center(delay, hist, cfg, c_rough, width):
    """Refine the dip position to a fraction of a bin (new in 0.10.0).

    g2 is symmetric in the delay, so the histogram split onto a grid
    centered on the true zero is symmetric up to noise. Around the first
    estimate the center is moved within the larger of +-2 grid bins and
    +-width (the dip's half-width from `_locate_dip`) to minimize the
    mean symmetry chi-square per bin pair,
    (o_k - o_-k)^2 / (v_k + v_-k), over the pairs whose bins the data
    cover fully, with v the Poisson variance of each split bin (a scan
    in steps of a quarter bin, then a bounded 1-D search). Dividing by
    the variance, not by the counts, matters: splitting input bins
    between grid bins averages their noise, so the counts-normalized
    sum is smallest half an input bin off center even for perfectly
    symmetric data. The caller refuses a result whose window does not
    fit the data. Detectors with asymmetric timing jitter would bias
    this; give `center` explicitly for them.
    """
    from scipy.optimize import minimize_scalar
    w = cfg.bin_width
    base = np.linspace(-cfg.tau_max, cfg.tau_max, cfg.n_bins + 1)
    K = cfg.n_bins
    e = _input_edges(delay)

    def asym(c):
        edges = base + c
        o, v = _split_moments(delay, hist, edges)
        full = (edges[:-1] >= e[0] - 1e-9) & (edges[1:] <= e[-1] + 1e-9)
        k = np.arange(K // 2)                # pairs k <-> K-1-k
        m = full[k] & full[K - 1 - k]
        if not m.any():
            return np.inf
        k = k[m]
        d = o[k] - o[K - 1 - k]
        return float(np.mean(d * d / np.maximum(v[k] + v[K - 1 - k], 1.0)))

    step = float(np.median(np.diff(delay)))
    reach = max(2 * w, width, step)
    a, b = c_rough - reach, c_rough + reach
    n = int(np.clip(np.ceil((b - a) / (0.25 * min(w, step))), 41, 801))
    grid = np.linspace(a, b, n)
    v = [asym(c) for c in grid]
    i = int(np.argmin(v))
    res = minimize_scalar(asym, bounds=(grid[max(i - 1, 0)],
                                        grid[min(i + 1, n - 1)]),
                          method="bounded", options=dict(xatol=1e-4 * w))
    return float(res.x) if res.fun <= v[i] else float(grid[i])
