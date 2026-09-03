"""Closed-loop emitter-triage environment.

A confocal field of M candidate sites must be triaged: certify the good
single-photon emitters (g2(0) < 0.5, bright, stable) and reject the rest,
spending as little measurement time as possible.  The agent works through
a shuffled queue of sites; at each step it either exposes the current site
a bit longer (three dwell choices), rejects it, or certifies it.  Its
perception is the posterior of the physics-in-the-loop estimator applied
to the site's accumulated coincidence record.  Rewards are computed from
the twin's ground truth (training happens *inside* the validated twin).
"""
from __future__ import annotations
import numpy as np
import torch

from .physics import sample_site, expected_histogram
from .datasets import CFG

DWELLS = (0.25, 1.0, 4.0)
A_MEAS0, A_MEAS1, A_MEAS2, A_REJECT, A_CERTIFY = range(5)
N_ACTIONS = 5
OBS_DIM = 8
MOVE_OVERHEAD_S = 0.5      # stage move + settling per site
MAX_DWELL = 15.0


class TriageEnv:
    def __init__(self, estimator, n_sites=48, platform="NV", seed=0,
                 lam_time=0.06, r_certify=1.0, r_false=3.0, r_miss=1.0,
                 cfg=CFG):
        self.est = estimator
        self.n_sites = n_sites
        self.platform = platform
        self.cfg = cfg
        self.lam = lam_time
        self.r_certify, self.r_false, self.r_miss = r_certify, r_false, r_miss
        self.rng = np.random.default_rng(seed)

    def new_field(self, rng=None):
        rng = rng or self.rng
        return [sample_site(rng, self.platform) for _ in range(self.n_sites)]

    def reset(self, field=None, noise_rng=None):
        self.field = field if field is not None else self.new_field()
        self.noise_rng = noise_rng or self.rng
        self.idx = 0
        self.hist = np.zeros(self.cfg.n_bins, np.float32)
        self.dwell = 0.0
        self.singles = 0.0
        self.t_total = MOVE_OVERHEAD_S
        self.certified = []          # (site_index, is_good)
        self.done = False
        self._update_posterior()
        return self._obs()

    # ------------------------------------------------------------------
    def _measure(self, T):
        site = self.field[self.idx]
        mu = expected_histogram(site, T, self.cfg)
        self.hist += self.noise_rng.poisson(mu).astype(np.float32)
        p = site.params
        duty = 1.0
        if p["blinking"]:
            duty = p["t_on_ms"] / (p["t_on_ms"] + p["t_off_ms"])
        self.singles += self.noise_rng.poisson(p["rate_kcps"] * 1e3 * duty * T)
        self.dwell += T
        self.t_total += T
        self._update_posterior()

    def _update_posterior(self):
        if self.dwell <= 0:
            self.p_good = 0.5
            return
        r_hat = max(self.singles / self.dwell, 1.0)
        exp_flat = (0.5 * r_hat) ** 2 * (self.cfg.bin_width * 1e-9) \
            * self.dwell * self.cfg.n_bins
        central = self.hist[np.abs(self.cfg.bin_centers) < 12.0].sum()
        aux = np.array([[np.log10(self.dwell), np.log10(r_hat),
                         np.log10(1.0 + self.hist.sum()),
                         np.log10(1.0 + exp_flat),
                         np.log10(1.0 + central)]], np.float32)
        with torch.no_grad():
            logits, _ = self.est(torch.from_numpy(self.hist[None]),
                                 torch.from_numpy(aux))
            self.p_good = float(torch.softmax(logits, 1)[0, 1])

    def _obs(self):
        return np.array([
            self.p_good,
            abs(2 * self.p_good - 1.0),
            np.log10(1.0 + self.dwell) / 1.5,
            np.log10(1.0 + self.hist.sum()) / 5.0,
            np.log10(max(self.singles / max(self.dwell, 0.25), 1.0)) / 6.0,
            (self.n_sites - self.idx) / self.n_sites,
            min(self.t_total / (4.0 * self.n_sites), 1.5),
            len(self.certified) / max(1, self.n_sites // 4),
        ], np.float32)

    def _advance(self):
        self.idx += 1
        if self.idx >= self.n_sites:
            self.done = True
        else:
            self.hist = np.zeros(self.cfg.n_bins, np.float32)
            self.dwell = 0.0
            self.singles = 0.0
            self.t_total += MOVE_OVERHEAD_S
            self._update_posterior()

    def step(self, a):
        assert not self.done
        r = 0.0
        site = self.field[self.idx]
        if a in (A_MEAS0, A_MEAS1, A_MEAS2):
            T = DWELLS[a]
            self._measure(T)
            r -= self.lam * T
            if self.dwell > MAX_DWELL:      # force a decision beyond cap
                a = A_CERTIFY if self.p_good > 0.5 else A_REJECT
        if a == A_REJECT:
            if site.is_good:
                r -= self.r_miss
            self._advance()
        elif a == A_CERTIFY:
            if site.is_good:
                r += self.r_certify
                self.certified.append((self.idx, True))
            else:
                r -= self.r_false
                self.certified.append((self.idx, False))
            self._advance()
        r -= self.lam * 0.0
        return self._obs(), r, self.done, {}

    # ------------------------------------------------------------------
    def summary(self):
        good = [i for i, s in enumerate(self.field) if s.is_good]
        cert_good = [i for i, g in self.certified if g]
        n_false = sum(1 for _, g in self.certified if not g)
        prec = (len(cert_good) / len(self.certified)) if self.certified else 1.0
        rec = (len(cert_good) / len(good)) if good else 1.0
        return dict(time_s=self.t_total, precision=prec, recall=rec,
                    n_good=len(good), n_cert=len(self.certified),
                    n_false=n_false,
                    good_per_min=60.0 * len(cert_good) / self.t_total)


# ----------------------------------------------------------------------
# Baseline policies
# ----------------------------------------------------------------------

def run_raster(env, field, T_fix, noise_rng, p_thresh=0.5):
    """Fixed-dwell raster: measure every site T_fix, threshold posterior."""
    env.reset(field=field, noise_rng=noise_rng)
    while not env.done:
        # measure in chunks matching available dwell actions
        remaining = T_fix
        for d, a in ((4.0, A_MEAS2), (1.0, A_MEAS1), (0.25, A_MEAS0)):
            while remaining >= d - 1e-9:
                env.step(a)
                remaining -= d
        env.step(A_CERTIFY if env.p_good > p_thresh else A_REJECT)
    return env.summary()


def run_adaptive_heuristic(env, field, noise_rng, margin=0.9, T_cap=5.0):
    """Uncertainty heuristic: expose in 0.25 s steps until confident."""
    env.reset(field=field, noise_rng=noise_rng)
    while not env.done:
        if abs(2 * env.p_good - 1) >= margin or env.dwell >= T_cap:
            env.step(A_CERTIFY if env.p_good > 0.5 else A_REJECT)
        else:
            env.step(A_MEAS0)
    return env.summary()


def run_policy(env, field, agent, noise_rng, greedy=True):
    obs = env.reset(field=field, noise_rng=noise_rng)
    while not env.done:
        a = agent.act(obs, greedy=greedy)
        obs, _, _, _ = env.step(a)
    return env.summary()
