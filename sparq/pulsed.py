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
"""
from __future__ import annotations
import numpy as np

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
    v[:2] = v[-2:] = np.inf
    center = float(pos[int(np.argmin(v))])
    return center, float(phase), t_rep


def g2_peak_area(delay, hist, center, t_rep=T_REP_NS, n_side=6, half=3.0):
    """Conventional pulsed analysis: center-peak area over mean side-peak
    area (background floor subtracted from both)."""
    # background floor from inter-peak regions
    off = np.abs(((delay - center + t_rep / 2) % t_rep) - t_rep / 2)
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
