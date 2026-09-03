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
