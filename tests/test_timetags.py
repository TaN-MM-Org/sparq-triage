"""v0.7 time-tag anchors: the searchsorted correlator against a
brute-force O(N^2) double loop (exact equality), start-stop equal to
all-pairs on sparse streams and elementwise below it otherwise,
Poisson flatness of the accidental normalization on independent
streams, and an exact CSV round trip."""
import numpy as np
import pytest

from sparq import HBTConfig, correlate
from sparq.timetags import (correlate_start_stop, load_timetags_csv,
                            normalize_g2, save_timetags_csv)

CFG = HBTConfig()


def _brute_force(t_a, t_b, cfg):
    """Independent path: every pair, plain double loop, no searchsorted."""
    d = []
    for ta in t_a:
        for tb in t_b:
            if -cfg.tau_max <= tb - ta < cfg.tau_max:
                d.append(tb - ta)
    h, _ = np.histogram(d, bins=cfg.n_bins,
                        range=(-cfg.tau_max, cfg.tau_max))
    return h.astype(np.float32)


def test_correlate_matches_brute_force_exactly():
    rng = np.random.default_rng(3)
    t_a = np.sort(rng.uniform(0, 5e4, 400))
    t_b = np.sort(rng.uniform(0, 5e4, 350))
    assert np.array_equal(correlate(t_a, t_b, CFG), _brute_force(t_a, t_b, CFG))


def test_start_stop_sparse_equality_and_pileup_undercount():
    # sparse: consecutive B tags farther apart than the full window,
    # so no start can see two stops -> conventions identical
    rng = np.random.default_rng(5)
    t_b = np.cumsum(rng.uniform(3 * CFG.tau_max, 6 * CFG.tau_max, 300))
    t_a = np.sort(t_b[::3] + rng.uniform(-20.0, 20.0, 100))
    h_all = correlate(t_a, t_b, CFG)
    h_ss = correlate_start_stop(t_a, t_b, CFG)
    assert np.array_equal(h_all, h_ss)
    # dense: start-stop can only lose pairs, never invent them
    t_a = np.sort(rng.uniform(0, 2e4, 2000))
    t_b = np.sort(rng.uniform(0, 2e4, 2000))
    h_all = correlate(t_a, t_b, CFG)
    h_ss = correlate_start_stop(t_a, t_b, CFG)
    assert h_ss.sum() < h_all.sum()
    assert h_ss.sum() <= len(t_a)              # at most one stop per start
    with pytest.raises(ValueError):
        correlate_start_stop(t_a[::-1], t_b, CFG)   # unsorted refused


def test_poisson_streams_normalize_to_flat_unity():
    """Two independent Poisson streams must read g2 = 1: mean within
    a few standard errors, per-bin scatter consistent with the
    reported Poisson sigmas (reduced chi-square near 1)."""
    rng = np.random.default_rng(11)
    T_s = 0.5                                    # 0.5 s
    rate = 2e5                                   # 200 kcps per channel
    n = rng.poisson(rate * T_s)                  # ~20 accidentals per bin
    m = rng.poisson(rate * T_s)
    t_a = np.sort(rng.uniform(0, T_s * 1e9, n))
    t_b = np.sort(rng.uniform(0, T_s * 1e9, m))
    out = normalize_g2(correlate(t_a, t_b, CFG), n, m, T_s, CFG)
    mean_g2 = out["g2"].mean()
    sem = out["sigma"].mean() / np.sqrt(CFG.n_bins)
    assert abs(mean_g2 - 1.0) < 4 * sem
    z = (out["g2"] - 1.0) / out["sigma"]
    red_chi2 = float(np.mean(z ** 2))
    assert 0.6 < red_chi2 < 1.5
    # the accidental level itself: N1 N2 w / T, checked independently
    acc = n * m * (CFG.bin_width * 1e-9) / T_s
    assert abs(out["accidentals_per_bin"] - acc) < 1e-9 * acc


def test_antibunched_stream_dips_below_accidental_level():
    """End-to-end: tags simulated from a single quantum emitter must
    normalize to g2 < 1 at zero delay and ~1 far away."""
    from sparq import sample_site, simulate_photon_stream
    rng = np.random.default_rng(2)
    site = sample_site(rng, n_probs=(1.0, 0.0, 0.0, 0.0))   # one emitter
    site.params["blinking"] = False
    T_s = 2.0
    t_a, t_b = simulate_photon_stream(site, T_s, rng, imp=None)
    n_a, n_b = len(t_a), len(t_b)
    out = normalize_g2(correlate(t_a, t_b, CFG), n_a, n_b, T_s, CFG)
    c = CFG.n_bins // 2                          # tau = 0 bin
    wings = np.r_[out["g2"][:15], out["g2"][-15:]].mean()
    assert out["g2"][c] < 0.7 * wings            # clear antibunching
    assert abs(wings - 1.0) < 0.2                # wings near accidental level


def test_csv_round_trip_and_refusals(tmp_path):
    rng = np.random.default_rng(9)
    t_a = np.sort(rng.uniform(0, 1e6, 50))
    t_b = np.sort(rng.uniform(0, 1e6, 60))
    ch = np.r_[np.zeros(50, int), np.ones(60, int)]
    t = np.r_[t_a, t_b]
    order = rng.permutation(110)                 # interleaved on disk
    p = tmp_path / "tags.csv"
    save_timetags_csv(p, ch[order], t[order])
    ra, rb = load_timetags_csv(p)
    assert np.array_equal(ra, t_a) and np.array_equal(rb, t_b)  # bit-exact
    save_timetags_csv(tmp_path / "bad.csv", [0, 7, 7], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        load_timetags_csv(tmp_path / "bad.csv")  # unknown channel 7
    ra2, rb2 = load_timetags_csv(tmp_path / "bad.csv", channel_a=0,
                                 channel_b=7)
    assert ra2.size == 1 and rb2.size == 2       # remap accepted
    with pytest.raises(ValueError):
        normalize_g2(np.zeros(CFG.n_bins), 0, 10, 1.0, CFG)
    with pytest.raises(ValueError):
        normalize_g2(np.zeros(CFG.n_bins), 10, 10, -1.0, CFG)
    with pytest.raises(ValueError):
        normalize_g2(np.zeros(5), 10, 10, 1.0, CFG)
