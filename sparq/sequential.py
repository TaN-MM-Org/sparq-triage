"""Sequential certification of single-photon emitters (Wald SPRT).

Fixed-time acquisition wastes photons on easy sites and starves hard
ones.  This module implements the sequential probability ratio test on
accumulating HBT histograms: after every increment of data the exact
Poisson log-likelihood ratio between two fully specified emitter
hypotheses is updated, and acquisition stops the moment the evidence
crosses a Wald threshold.  For simple hypotheses the SPRT is optimal
(Wald and Wolfowitz 1948): no other test with the same error rates has a
smaller expected sample size.

For independent Poisson bins with means mu1_i (accept hypothesis, e.g.
the single-emitter site) and mu0_i (reject hypothesis, e.g. the
two-emitter site), an increment of counts k_i contributes

    Lambda += sum_i [ k_i ln(mu1_i / mu0_i) - (mu1_i - mu0_i) ],

and the decision thresholds for error rates alpha (falsely accepting on
a reject-hypothesis site) and beta (falsely rejecting an
accept-hypothesis site) are Wald's

    A = ln((1 - beta) / alpha),      B = ln(beta / (1 - alpha)).

In practice the nuisance parameters (lifetimes, rates) are plugged into
the two hypothesis sites from a calibration fit rather than known
exactly, so the guarantees are approximate; the test suite checks the
realized error rates and stopping times empirically against the twin.
The expected decision times follow Wald's approximations,

    E1[Lambda per second] = D1 = sum_i [mu1'_i ln(mu1_i/mu0_i) - mu1'_i + mu0'_i],

with mu' the per-second means, and E1[tau] ~ ((1-beta) A + beta B) / D1
(and the mirrored expression under the reject hypothesis).
"""
from __future__ import annotations

import numpy as np

from .physics import EmitterSite, HBTConfig, expected_histogram

CONTINUE, ACCEPT, REJECT = "continue", "accept", "reject"


class SPRTCertifier:
    """Sequential certifier between two fully specified emitter hypotheses.

    Parameters: site_accept and site_reject are :class:`EmitterSite`
    hypotheses (typically the same photophysics with n_emitters 1 and 2);
    alpha bounds the probability of accepting a reject-hypothesis site,
    beta the probability of rejecting an accept-hypothesis site; cfg is
    the histogram grid the data arrives on.

    Feed data with :meth:`update`; the accumulated evidence is additive,
    so it does not matter how the acquisition is chopped into increments.
    """

    def __init__(self, site_accept: EmitterSite, site_reject: EmitterSite,
                 cfg: HBTConfig | None = None,
                 alpha: float = 0.05, beta: float = 0.05):
        if not (0 < alpha < 1 and 0 < beta < 1):
            raise ValueError("alpha and beta must lie in (0, 1)")
        self.site_accept = site_accept
        self.site_reject = site_reject
        self.cfg = HBTConfig() if cfg is None else cfg
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.upper = float(np.log((1.0 - beta) / alpha))
        self.lower = float(np.log(beta / (1.0 - alpha)))
        # per-second means (expected_histogram is linear in T)
        self._mu1s = expected_histogram(site_accept, 1.0, self.cfg)
        self._mu0s = expected_histogram(site_reject, 1.0, self.cfg)
        if np.any(self._mu1s <= 0) or np.any(self._mu0s <= 0):
            raise ValueError("both hypotheses must give positive bin means")
        self._log_ratio = np.log(self._mu1s / self._mu0s)
        self.llr = 0.0
        self.T_total = 0.0
        self.decision = CONTINUE

    def update(self, counts, T_s: float) -> str:
        """Add an acquisition increment (histogram ``counts`` accumulated
        over ``T_s`` seconds) and return 'accept', 'reject' or 'continue'."""
        if self.decision != CONTINUE:
            return self.decision
        counts = np.asarray(counts, float)
        if counts.shape != self._mu1s.shape:
            raise ValueError("counts must be on the certifier's histogram grid")
        if T_s <= 0:
            raise ValueError("T_s must be positive")
        self.llr += float(counts @ self._log_ratio
                          - T_s * np.sum(self._mu1s - self._mu0s))
        self.T_total += float(T_s)
        if self.llr >= self.upper:
            self.decision = ACCEPT
        elif self.llr <= self.lower:
            self.decision = REJECT
        return self.decision

    def kl_rates(self):
        """(D1, D0): expected log-likelihood-ratio drift per second under
        the accept and reject hypotheses (D1 > 0 > -D0)."""
        d1 = float(np.sum(self._mu1s * self._log_ratio
                          - self._mu1s + self._mu0s))
        d0 = float(np.sum(self._mu0s * np.log(self._mu0s / self._mu1s)
                          - self._mu0s + self._mu1s))
        return d1, d0

    def expected_times(self):
        """Wald's approximate expected decision times (seconds) under the
        accept and reject hypotheses."""
        d1, d0 = self.kl_rates()
        A, B = self.upper, self.lower
        t_accept = ((1 - self.beta) * A + self.beta * B) / d1
        t_reject = -(self.alpha * A + (1 - self.alpha) * B) / d0
        return float(t_accept), float(t_reject)


class WindowSPRT:
    """Sequential test that needs no emitter model and no count rate
    (new in 0.10.0).

    It uses the same two windows as `sparq.bayes.bayesian_g2`: n0
    central bins and nr far reference bins. If the flat (uncorrelated)
    level is lam per bin and the window-averaged g2 is g, the central
    counts have mean g lam per bin and the reference counts lam. Given
    the total number of counts that fall in the two windows, the number
    k0 in the central window is binomial with success probability

        p(g) = n0 g / (n0 g + nr),

    which does not involve lam. The test is Wald's sequential
    probability ratio test between p(g_accept) and p(g_reject) on these
    counts, conditionally on the totals, so:

    * nothing about the emitter (lifetimes, shoulder, rate) and nothing
      about the count rate is assumed; the rate may even change from one
      increment to the next;
    * because the binomial has a monotone likelihood ratio in g, the
      error bounds hold for the whole ranges, not just the two values:
      for every g >= g_reject the probability of wrongly accepting is at
      most alpha / (1 - beta), and for every g <= g_accept the
      probability of wrongly rejecting is at most beta / (1 - alpha)
      (Wald's inequalities; they include the overshoot of the last
      increment). The tests check both on simulated runs.

    What it tests: the window-averaged g2 of `bayesian_g2` (an upper
    bound on g2(0) when the dip is wider than the window), with
    "accept" meaning g <= g_accept and "reject" meaning g >= g_reject.
    Between the two values either decision may come out.

    Feed each new increment's histogram (on the grid `cfg`, centered on
    the dip) with `update`; its time is not needed.
    """

    def __init__(self, g_accept=0.25, g_reject=0.5,
                 cfg: HBTConfig | None = None, alpha: float = 0.05,
                 beta: float = 0.05, n_center_bins: int = 1,
                 lo_frac: float = 0.65):
        if not (0 < alpha < 1 and 0 < beta < 1):
            raise ValueError("alpha and beta must lie in (0, 1)")
        if not (0.0 < g_accept < g_reject):
            raise ValueError("need 0 < g_accept < g_reject")
        self.cfg = HBTConfig() if cfg is None else cfg
        if self.cfg.n_bins % 2 == 0:
            raise ValueError("cfg.n_bins must be odd (a bin centered at 0)")
        if n_center_bins < 1 or n_center_bins % 2 == 0:
            raise ValueError("n_center_bins must be odd and >= 1")
        tau = self.cfg.bin_centers
        half = (n_center_bins // 2 + 0.5) * self.cfg.bin_width
        self._m0 = np.abs(tau) < half
        self._mr = np.abs(tau) >= lo_frac * self.cfg.tau_max
        if int(self._m0.sum()) != n_center_bins or not self._mr.any() \
                or np.any(self._m0 & self._mr):
            raise ValueError("central and reference windows must be "
                             "non-empty and must not overlap")
        n0, nr = float(self._m0.sum()), float(self._mr.sum())
        self.p_accept = n0 * g_accept / (n0 * g_accept + nr)
        self.p_reject = n0 * g_reject / (n0 * g_reject + nr)
        self._w0 = float(np.log(self.p_accept / self.p_reject))
        self._wr = float(np.log((1 - self.p_accept) / (1 - self.p_reject)))
        self.g_accept, self.g_reject = float(g_accept), float(g_reject)
        self.alpha, self.beta = float(alpha), float(beta)
        self.upper = float(np.log((1.0 - beta) / alpha))
        self.lower = float(np.log(beta / (1.0 - alpha)))
        self.llr = 0.0
        self.k0 = 0.0
        self.kr = 0.0
        self.decision = CONTINUE

    @property
    def error_bounds(self):
        """(bound on wrongly accepting, bound on wrongly rejecting)."""
        return (self.alpha / (1.0 - self.beta),
                self.beta / (1.0 - self.alpha))

    def update(self, counts) -> str:
        """Add an increment's histogram; returns 'accept', 'reject' or
        'continue'."""
        if self.decision != CONTINUE:
            return self.decision
        c = np.asarray(counts, dtype=float)
        if c.shape != (self.cfg.n_bins,):
            raise ValueError("counts must be on the test's histogram grid")
        if np.any(c < 0) or not np.all(np.isfinite(c)):
            raise ValueError("counts must be finite and non-negative")
        k0 = float(c[self._m0].sum())
        kr = float(c[self._mr].sum())
        self.k0 += k0
        self.kr += kr
        self.llr += k0 * self._w0 + kr * self._wr
        if self.llr >= self.upper:
            self.decision = ACCEPT
        elif self.llr <= self.lower:
            self.decision = REJECT
        return self.decision

    def expected_window_counts(self):
        """Wald's approximate expected number of window counts (central
        plus reference) to a decision when g = g_accept and when
        g = g_reject (it ignores the overshoot)."""
        out = []
        for p, pa in ((self.p_accept, 1 - self.beta),
                      (self.p_reject, self.alpha)):
            drift = p * self._w0 + (1 - p) * self._wr
            out.append((pa * self.upper + (1 - pa) * self.lower) / drift)
        return float(out[0]), float(out[1])
