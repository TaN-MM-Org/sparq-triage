"""Synthetic acquisition generators (the twin as a training environment)
and the loader for the real experimental HBT dataset (UTS-CASLab
sps-quality, FI-SEQUR InGaAs/GaAs quantum dot; Kedziora et al., MLST 2023).
"""
from __future__ import annotations
import glob
import os
import numpy as np

from .physics import (HBTConfig, EmitterSite, PLATFORMS, sample_site,
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
      aux     [B, 3]     log10 T, log10 singles-rate estimate, log10 counts
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
        p = site.params
        duty = 1.0
        if p["blinking"]:
            duty = p["t_on_ms"] / (p["t_on_ms"] + p["t_off_ms"])
        n_singles = rng.poisson(p["rate_kcps"] * 1e3 * duty * T)
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

FISEQUR_DIR = "/home/claude/sps-quality/data/InGaAs-GaAs QDs/FI-SEQUR project demonstrator sample"


def load_fisequr(path_dir=FISEQUR_DIR):
    """Load the eight FI-SEQUR HBT measurement series.

    Each file: rows = delay bins, first column = delay (ns), remaining
    columns = coincidence counts of successive 10-s snapshots.
    Returns list of dicts with delay axis, snapshot matrix, and metadata.
    """
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


def robust_flat_rate(hist, cfg, T_s, lo_frac=0.65):
    """Effective singles rate from a trimmed median of far-delay bins
    (robust to real detector artifacts such as echo peaks)."""
    m = np.abs(cfg.bin_centers) >= lo_frac * cfg.tau_max
    flat = float(np.median(hist[m]))
    return 2.0 * np.sqrt(max(flat, 1e-9) / (cfg.bin_width * 1e-9 * T_s))


def rebin_real(delay, hist, cfg=CFG, center=None):
    """Re-bin a real histogram onto the twin's 121-bin +-60.5 ns grid,
    centered on the antibunching dip."""
    if center is None:
        # robust dip locate: heavily smoothed minimum
        k = 41
        sm = np.convolve(hist, np.ones(k) / k, mode="same")
        # ignore edges
        m = len(hist)
        center = delay[np.argmin(sm[m // 10: 9 * m // 10]) + m // 10]
    edges = np.linspace(-cfg.tau_max, cfg.tau_max, cfg.n_bins + 1) + center
    idx = np.searchsorted(edges, delay)
    out = np.zeros(cfg.n_bins, np.float32)
    for b in range(cfg.n_bins):
        out[b] = hist[(idx == b + 1)].sum()
    return out, center
