"""Raw photon time tags: ingestion, hardware-style correlation and the
standard accidental normalization (new in v0.7).

Real HBT data arrives as time tags, not histograms. This module covers
the path from a two-channel tag list to a normalized g2 with error
bars, with the two correlator conventions actually found in labs:

* `correlate` (in :mod:`sparq.physics`): the ALL-pairs correlator a
  software correlator or a multistop TDC computes -- every (start,
  stop) pair within the window counts. This is the estimator all
  analysis in this package assumes.
* `correlate_start_stop` (here): the classic start-stop convention of
  a TAC/single-stop TCSPC card -- after each start only the FIRST
  stop within the window is recorded. At low count rates per window
  the two agree; at high rates start-stop undercounts long delays
  (pile-up), which is why the all-pairs histogram is the one to fit.
  Both facts are asserted in the tests, not stated.

The normalization is the standard accidental-coincidence form: two
uncorrelated streams with N1 and N2 detected counts in acquisition
time T produce on average N1 N2 w / T coincidences per bin of width
w, so

    g2(tau) = C(tau) * T / (N1 N2 w),

which makes two independent Poisson streams read g2 = 1 -- asserted
by simulation against the Poisson error bars. (See e.g. R. Brouri,
A. Beveratos, J.-P. Poizat and P. Grangier, Opt. Lett. 25, 1294
(2000), and the standard treatment in Fox, Quantum Optics (2006),
ch. 6.) The per-bin one-sigma error bars are Poisson,
sqrt(C)/(N1 N2 w / T), stated as such.

No proprietary binary formats are parsed here (a PTU reader written
without the vendor spec would be a guess); the CSV contract is exact
and round-trips bit-for-bit through `save_timetags_csv` /
`load_timetags_csv`, and any two-column (channel, time) export from
vendor software fits it.
"""
from __future__ import annotations

import numpy as np

from .physics import HBTConfig, correlate

__all__ = ["load_timetags_csv", "save_timetags_csv",
           "correlate_start_stop", "normalize_g2"]


def save_timetags_csv(path, channels, times_ns):
    """Write a two-column time-tag CSV: integer channel, time in ns.

    The exact-contract partner of `load_timetags_csv`; the round trip
    is asserted in the tests.
    """
    ch = np.asarray(channels)
    t = np.asarray(times_ns, dtype=float)
    if ch.shape != t.shape or ch.ndim != 1:
        raise ValueError("channels and times_ns must be equal-length 1D")
    with open(path, "w") as f:
        f.write("channel,time_ns\n")
        for c, tt in zip(ch, t):
            f.write(f"{int(c)},{float(tt)!r}\n")


def load_timetags_csv(path, channel_a=0, channel_b=1):
    """Read a two-column time-tag CSV into two sorted arrays (ns).

    Expects a header line and rows `channel,time_ns`. Returns
    (t_a, t_b) for the two requested channels, each sorted ascending
    (tags from interleaved channels are not necessarily sorted per
    channel, so this sorts explicitly rather than assuming). Refuses
    non-finite times, unknown channels, and an empty file, instead of
    correlating garbage silently.
    """
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.size == 0:
        raise ValueError(f"{path}: no time tags")
    data = np.atleast_2d(data)
    if data.shape[1] != 2:
        raise ValueError(f"{path}: expected two columns (channel, time_ns)")
    ch = data[:, 0]
    t = data[:, 1]
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(ch)):
        raise ValueError(f"{path}: non-finite entries")
    known = np.isin(ch, [channel_a, channel_b])
    if not np.all(known):
        bad = sorted(set(ch[~known].astype(int).tolist()))
        raise ValueError(f"{path}: unknown channels {bad}; pass "
                         "channel_a/channel_b to map your numbering")
    t_a = np.sort(t[ch == channel_a])
    t_b = np.sort(t[ch == channel_b])
    return t_a, t_b


def correlate_start_stop(t_a, t_b, cfg: HBTConfig):
    """Classic TAC start-stop histogram: first stop per start only.

    For each start tag on channel A, the single earliest stop tag on
    channel B with delay in (-tau_max, +tau_max] is histogrammed;
    later stops in the same window are discarded, exactly as a
    single-stop TCSPC card does. Provided so hardware histograms can
    be emulated and compared; FIT THE ALL-PAIRS `correlate` HISTOGRAM,
    which has no pile-up distortion. The tests assert exact agreement
    with `correlate` on streams sparse enough that no window holds
    two stops, and the elementwise undercount otherwise.
    """
    t_a = np.asarray(t_a, dtype=float)
    t_b = np.asarray(t_b, dtype=float)
    if np.any(np.diff(t_a) < 0) or np.any(np.diff(t_b) < 0):
        raise ValueError("timestamp arrays must be sorted ascending")
    hist = np.zeros(cfg.n_bins, dtype=np.float32)
    if len(t_a) == 0 or len(t_b) == 0:
        return hist
    lo = np.searchsorted(t_b, t_a - cfg.tau_max)
    hi = np.searchsorted(t_b, t_a + cfg.tau_max)
    first = []
    for i in range(len(t_a)):
        if hi[i] > lo[i]:
            first.append(t_b[lo[i]] - t_a[i])   # earliest stop in window
    if not first:
        return hist
    h, _ = np.histogram(np.asarray(first), bins=cfg.n_bins,
                        range=(-cfg.tau_max, cfg.tau_max))
    return h.astype(np.float32)


def normalize_g2(hist, n_a, n_b, T_s, cfg: HBTConfig):
    """Normalize a coincidence histogram by the accidental rate.

    hist : per-bin coincidence counts (from `correlate`).
    n_a, n_b : total detected counts on the two channels during the
    acquisition (len(t_a), len(t_b) -- the counts that MADE the
    histogram, not a rate from elsewhere).
    T_s : acquisition time in SECONDS (bin width is in ns; the unit
    conversion is inside, one less way to be wrong by 1e9).

    Returns dict(g2, sigma, accidentals_per_bin): the normalized
    correlation g2 = hist * T / (N1 N2 w), its per-bin Poisson
    one-sigma sqrt(hist)/(N1 N2 w / T) (zero-count bins get the
    one-count upper scale, stated plainly), and the accidental level
    N1 N2 w / T the normalization divides by. Uncorrelated streams
    read 1 within errors -- asserted by simulation in the tests.
    """
    h = np.asarray(hist, dtype=float)
    if h.ndim != 1 or h.shape[0] != cfg.n_bins:
        raise ValueError(f"hist must have cfg.n_bins = {cfg.n_bins} bins")
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        raise ValueError("hist must be finite and non-negative")
    na, nb, T = float(n_a), float(n_b), float(T_s)
    if not (na > 0 and nb > 0):
        raise ValueError("n_a and n_b must be positive counts")
    if not (T > 0 and np.isfinite(T)):
        raise ValueError("T_s must be a positive acquisition time in s")
    acc = na * nb * (cfg.bin_width * 1e-9) / T     # accidentals per bin
    g2 = h / acc
    sigma = np.sqrt(np.maximum(h, 1.0)) / acc
    return dict(g2=g2, sigma=sigma, accidentals_per_bin=acc)
