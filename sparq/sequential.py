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

Both tests, and every fixed-time analysis, assume that the source does
not change while it is measured. `check_stationarity` (new in 0.12.0)
tests that assumption on a series of snapshots before they are summed.
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
        if not np.all(np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError("counts must be finite and non-negative")
        if not (np.isfinite(float(T_s)) and float(T_s) > 0):
            raise ValueError(f"T_s must be a positive, finite time (s); "
                             f"got {T_s!r}")
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


# ----------------------------------------------------------------------
# are the increments of one acquisition consistent? (new in 0.12.0)
# ----------------------------------------------------------------------

def _pearson(observed, expected):
    return float(np.sum((observed - expected) ** 2 / expected))


def _homogeneity(table, min_expected, what):
    """Pearson chi-square test that every row of a (J, K) table of
    counts has the same shape (the rows may have any totals): expected
    count R_j C_k / N, dof (J - 1)(K - 1)."""
    from scipy.stats import chi2
    R = table.sum(axis=1)
    C = table.sum(axis=0)
    N = float(table.sum())
    E = np.outer(R, C) / N
    if E.min() < min_expected:
        raise ValueError(
            f"the {what} test needs at least {min_expected:g} expected "
            f"counts in every cell; the smallest is {E.min():.3g}: merge "
            "neighboring snapshots or measure longer")
    stat = _pearson(table, E)
    dof = (table.shape[0] - 1) * (table.shape[1] - 1)
    return stat, dof, float(chi2.sf(stat, dof))


def check_stationarity(snapshots, durations=None,
                       cfg: HBTConfig | None = None,
                       n_center_bins: int = 1, lo_frac: float = 0.65,
                       min_expected: float = 5.0,
                       alpha: float = 0.01) -> dict:
    """Are successive histograms of one site consistent with a source
    that did not change while it was measured? (new in 0.12.0)

    Every analysis in this package treats a histogram as the sum of
    Poisson counts from a steady source. Bleaching, blinking slower than
    a snapshot, a drifting focus or a second emitter that switches on
    break that, and summing such snapshots gives a g2 that belongs to no
    single state of the source. This check tests a series of snapshots
    before they are summed.

    snapshots : (J, n_bins) coincidence counts, J >= 2, each on the grid
        `cfg` and centered on the dip (native bins, or bins merged
        whole: the tests assume Poisson counts, as `bayesian_g2` does).
    durations : (J,) acquisition time of each snapshot in seconds
        (default: all equal).

    Three Pearson chi-square tests, each conditional on totals, so no
    model of the emitter and no count rate is needed:

    * rate: are the snapshot totals n_j proportional to the durations
      t_j? Given their sum N they are multinomial with shares
      t_j / sum(t) if the coincidence rate is steady; dof J - 1.
    * window: is the window g2 the same in every snapshot? The central
      window and the far reference window are those of `bayesian_g2`;
      given a snapshot's count in both windows together, its central
      count is binomial with a share fixed by the window g2 (see
      `WindowSPRT`), whatever the rate: a 2 x J homogeneity test,
      dof J - 1.
    * shape: is the whole histogram the same up to a scale? A J x K
      homogeneity test of the bins, with neighboring bins merged
      (from the left) until every expected count reaches
      `min_expected`; dof (J - 1)(K - 1).

    The chi-square law of the statistics holds when the expected counts
    are not small; a rate or window test with an expected count below
    `min_expected` (default 5) is refused with ValueError (merge
    snapshots or measure longer). `consistent` is True when all three
    p-values are at least alpha / 3, so that a steady source is called
    inconsistent with probability at most about alpha (Bonferroni).

    Returns a dict: rate_p, rate_chi2, rate_dof, window_p, window_chi2,
    window_dof, shape_p, shape_chi2, shape_dof, shape_bins (the number
    of merged bins), rate_cps (coincidences per second in each
    snapshot), g2_window (each snapshot's window g2, NaN without
    reference counts), alpha and consistent.
    """
    from scipy.stats import chi2
    cfg = HBTConfig() if cfg is None else cfg
    S = np.asarray(snapshots, dtype=float)
    if S.ndim != 2 or S.shape[0] < 2 or S.shape[1] != cfg.n_bins:
        raise ValueError(f"snapshots must be a (J, {cfg.n_bins}) array "
                         f"with J >= 2; got shape {S.shape}")
    if not np.all(np.isfinite(S)) or np.any(S < 0):
        raise ValueError("snapshot counts must be finite and non-negative")
    J = S.shape[0]
    if durations is None:
        t = np.ones(J)
    else:
        t = np.asarray(durations, dtype=float)
        if t.shape != (J,) or not np.all(np.isfinite(t)) or np.any(t <= 0):
            raise ValueError(f"durations must be {J} positive times (s)")
    if not (np.isfinite(min_expected) and min_expected > 0):
        raise ValueError("min_expected must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    if cfg.n_bins % 2 == 0:
        raise ValueError("cfg.n_bins must be odd (a bin centered at 0)")
    if n_center_bins < 1 or n_center_bins % 2 == 0:
        raise ValueError("n_center_bins must be odd and >= 1")
    tau = cfg.bin_centers
    m0 = np.abs(tau) < (n_center_bins // 2 + 0.5) * cfg.bin_width
    mr = np.abs(tau) >= lo_frac * cfg.tau_max
    if int(m0.sum()) != n_center_bins or not mr.any() or np.any(m0 & mr):
        raise ValueError("central and reference windows must be "
                         "non-empty and must not overlap")
    n = S.sum(axis=1)
    N = float(n.sum())
    if N <= 0:
        raise ValueError("the snapshots hold no counts")
    # rate
    e = N * t / t.sum()
    if e.min() < min_expected:
        raise ValueError(
            f"the rate test needs at least {min_expected:g} expected "
            f"counts per snapshot; the smallest is {e.min():.3g}")
    rate_stat = _pearson(n, e)
    rate_p = float(chi2.sf(rate_stat, J - 1))
    # window g2
    k0 = S[:, m0].sum(axis=1)
    kr = S[:, mr].sum(axis=1)
    win_stat, win_dof, win_p = _homogeneity(np.column_stack([k0, kr]),
                                            min_expected, "window")
    with np.errstate(divide="ignore", invalid="ignore"):
        g_win = np.where(kr > 0, (k0 / m0.sum()) / (kr / mr.sum()), np.nan)
    # shape: merge neighboring bins until every expected count is large
    C = S.sum(axis=0)
    need = min_expected * N / n.min() if n.min() > 0 else np.inf
    groups, cur, acc = [], [], 0.0
    for k in range(cfg.n_bins):
        cur.append(k)
        acc += C[k]
        if acc >= need:
            groups.append(cur)
            cur, acc = [], 0.0
    if cur:
        if groups:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    if len(groups) < 2:
        raise ValueError("too few counts for the shape test: even merged, "
                         "the bins do not reach min_expected; merge "
                         "snapshots or measure longer")
    table = np.column_stack([S[:, g].sum(axis=1) for g in groups])
    shp_stat, shp_dof, shp_p = _homogeneity(table, min_expected, "shape")
    consistent = bool(min(rate_p, win_p, shp_p) >= alpha / 3.0)
    return dict(rate_p=rate_p, rate_chi2=rate_stat, rate_dof=J - 1,
                window_p=win_p, window_chi2=win_stat, window_dof=win_dof,
                shape_p=shp_p, shape_chi2=shp_stat, shape_dof=shp_dof,
                shape_bins=len(groups), rate_cps=n / t, g2_window=g_win,
                alpha=float(alpha), consistent=consistent)
