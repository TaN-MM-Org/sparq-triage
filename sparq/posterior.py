"""Bayesian posterior of g2(0) itself under the three-level model (new
in 0.10.0).

`sparq.bayes.bayesian_g2` gives the exact posterior of the raw
window-averaged ratio, which needs no model but only bounds g2(0) from
above. This module gives the posterior of the model's g2(0) = 1 - rho2
-- the value without detector jitter, as `fit_g2_histogram` and
`profile_likelihood_ci` estimate it -- by sampling, since it has no
closed form.

Model and priors, stated in full because the answer depends on them.
The counts in bin k are Poisson with mean

    mu_k = F c0 [1 - rho2 ((1 + a) E(tau1) - a E(tau2))]_k,

F = (r_hat/2)^2 w T the flat level from the singles rate, E the
jitter-blurred exponential averaged over the bin (as in the fits).
Priors, independent: g2(0) = 1 - rho2 uniform on [0, 1]; tau1 and tau2
uniform in log over `t1_bounds` and `t2_bounds`; a uniform on [0, a_max];
c0 either scale-free, p(c0) ~ 1/c0, or, with `c0_prior=(m, sd)`, a
Gamma law with that mean and standard deviation (close to a Gaussian
when sd << m). For both, c0 is integrated out exactly (the Poisson
likelihood and a Gamma law are conjugate), so the sampler works in the
other four parameters and c0 is drawn afterwards from its exact
conditional law. Without the c0 prior the posterior of g2(0) is wide
when the shoulder is slow, for the reason given in
`profile_likelihood_ci`, and its width then depends on the prior
limits of tau2 and a; that is a property of the data, and the result
reports the share of samples near those limits.

Sampler: the affine-invariant ensemble sampler of Goodman and Weare
(stretch move; `stretch_sampler`) on (rho2, log tau1, a, log tau2),
started around the best fit. It copes with the strong correlations of
this posterior (the shoulder amplitude, its time and the dip depth
trade against each other) far better than a random walk. The walkers
are treated as chains for the Gelman-Rubin statistic R-hat and the
effective sample size that are reported. The tests check the
sampler against the closed-form beta-prime posterior of `bayesian_g2`
and check that the credible intervals cover the truth at their nominal
rate on simulated data.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .analysis import (DEFAULT_T1_BOUNDS, DEFAULT_T2_BOUNDS,
                       _check_c0_prior, _check_time_bounds,
                       fit_g2_histogram)
from .physics import HBTConfig, _exp_conv_gauss

__all__ = ["ModelPosterior", "bayesian_g2_model", "stretch_sampler"]


def stretch_sampler(logp, x0, lo, hi, n_keep, n_burn, rng, n_walkers=24,
                    a=2.0, spread=1e-3):
    """Affine-invariant ensemble sampler with the stretch move (Goodman
    and Weare, Commun. Appl. Math. Comput. Sci. 5, 65 (2010)).

    logp : log target density (up to a constant) of an (n, d) array of
        points, returning (n,); it is only called inside the box
        [lo, hi], outside which the target is taken to be zero.
    x0 : (d,) center of the starting ensemble (walkers spread by
        `spread` times the box width around it).
    Each iteration updates the two halves of the ensemble in turn, each
    walker against a random walker of the other half, which keeps the
    target invariant. Returns (samples (n_keep, n_walkers, d),
    acceptance rate over the kept iterations).
    """
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    x0 = np.asarray(x0, float)
    d = x0.size
    W = int(n_walkers)
    if W < 2 * d or W % 2:
        raise ValueError("n_walkers must be even and at least 2 d")

    def lp_of(X):
        out = np.full(X.shape[0], -np.inf)
        inb = np.all((X >= lo) & (X <= hi), axis=1)
        if inb.any():
            out[inb] = logp(X[inb])
        return out

    X = np.clip(x0 + spread * (hi - lo) * rng.standard_normal((W, d)),
                lo, hi)
    lp = lp_of(X)
    if not np.all(np.isfinite(lp)):
        raise ValueError("the starting ensemble has zero density; start "
                         "closer to the posterior")
    keep = np.empty((int(n_keep), W, d))
    acc = 0
    half = W // 2
    for it in range(int(n_burn) + int(n_keep)):
        for s0, s1 in ((slice(0, half), slice(half, W)),
                       (slice(half, W), slice(0, half))):
            xs, xo = X[s0], X[s1]
            n = xs.shape[0]
            j = rng.integers(0, xo.shape[0], n)
            z = ((a - 1.0) * rng.random(n) + 1.0) ** 2 / a
            Y = xo[j] + z[:, None] * (xs - xo[j])
            ly = lp_of(Y)
            ok = np.log(rng.random(n)) < (d - 1) * np.log(z) + ly - lp[s0]
            xs = xs.copy()
            xs[ok] = Y[ok]
            X[s0] = xs
            lps = lp[s0].copy()
            lps[ok] = ly[ok]
            lp[s0] = lps
            if it >= n_burn:
                acc += int(ok.sum())
        if it >= n_burn:
            keep[it - int(n_burn)] = X
    return keep, acc / (max(int(n_keep), 1) * W)


def _rhat_ess(chains):
    """Gelman-Rubin R-hat and a simple effective sample size (initial
    positive sequence of the autocorrelation, pooled over chains) for
    a (n_chains, n) array (the walkers of `stretch_sampler` as chains)."""
    m, n = chains.shape
    means = chains.mean(axis=1)
    W = chains.var(axis=1, ddof=1).mean()
    B = n * means.var(ddof=1) if m > 1 else 0.0
    var = (n - 1) / n * W + B / n
    rhat = float(np.sqrt(var / W)) if W > 0 and m > 1 else np.nan
    x = chains - means[:, None]
    ac = np.zeros(n)
    for c in range(m):
        f = np.fft.rfft(x[c], 2 * n)
        ac += np.fft.irfft(f * np.conj(f))[:n]
    ac /= ac[0] if ac[0] > 0 else 1.0
    s = 0.0
    for k in range(1, n - 1, 2):
        pair = ac[k] + ac[k + 1]
        if pair <= 0:
            break
        s += pair
    tau = 1.0 + 2.0 * s
    return rhat, float(m * n / max(tau, 1.0))


@dataclasses.dataclass
class ModelPosterior:
    """Samples of the three-level model's posterior.

    g2_0 : samples of g2(0) = 1 - rho2 (all walkers, all kept steps).
    params : dict of samples of rho2, tau1, a, tau2, c0.
    rhat : Gelman-Rubin statistic of g2_0 with the walkers as chains
        (near 1 when they agree; above about 1.05 means run longer).
    ess : effective sample size of g2_0.
    acceptance : share of accepted stretch moves.
    near_prior_limit : share of samples within 1 % (in log) of a limit
        of tau1 or tau2, or within 1 % of the upper limit of a -- a
        sizeable share means the prior limits shape the answer.
    """

    g2_0: np.ndarray
    params: dict
    rhat: float
    ess: float
    acceptance: float
    near_prior_limit: float

    def credible_interval(self, level=0.95):
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie in (0, 1)")
        h = 50.0 * (1.0 - level)
        lo, hi = np.percentile(self.g2_0, [h, 100.0 - h])
        return float(lo), float(hi)

    def prob_below(self, threshold=0.5):
        return float(np.mean(self.g2_0 < threshold))

    def median(self):
        return float(np.median(self.g2_0))


def bayesian_g2_model(hist, T_s, r_hat, cfg: HBTConfig | None = None,
                      c0_prior=None, t1_bounds=DEFAULT_T1_BOUNDS,
                      t2_bounds=DEFAULT_T2_BOUNDS, a_max=3.0,
                      n_samples=2000, n_burn=1000, n_walkers=24, seed=0):
    """Posterior of g2(0) under the three-level model (see the module
    docstring for the model and priors).

    hist : Poisson counts on the grid `cfg`, centered on the dip (native
        bins, or `rebin_real(..., method="whole")` at a whole multiple).
    T_s, r_hat : acquisition time (s) and singles rate (counts/s, both
        detectors) setting the flat level.
    c0_prior : optional (mean, sd) for the flat-level scale; with r_hat
        from the measured singles rates, (1.0, relative sd).
    Returns a `ModelPosterior`.
    """
    if cfg is None:
        cfg = HBTConfig()
    h = np.asarray(hist, dtype=float)
    if h.ndim != 1 or h.size != cfg.n_bins:
        raise ValueError(f"hist must have cfg.n_bins = {cfg.n_bins} bins")
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        raise ValueError("hist must be finite and non-negative")
    F = (0.5 * float(r_hat)) ** 2 * (cfg.bin_width * 1e-9) * float(T_s)
    if not F > 0:
        raise ValueError("r_hat and T_s must be positive")
    t1_lo, t1_hi = _check_time_bounds("t1_bounds", t1_bounds)
    t2_lo, t2_hi = _check_time_bounds("t2_bounds", t2_bounds)
    if not a_max > 0:
        raise ValueError("a_max must be positive")
    if c0_prior is not None:
        c0_m, c0_sd = _check_c0_prior(c0_prior)
    nodes, wts = cfg.bin_nodes()
    s = np.sqrt(2.0) * float(cfg.sigma_irf)
    # g2 is even in the delay and the grid is symmetric, so bins k and
    # n_bins-1-k have the same value: evaluate one half and mirror
    K_all = cfg.n_bins
    n_half = (K_all + 1) // 2
    an = np.abs(nodes[:n_half])
    mirror = np.concatenate([np.arange(n_half),
                             np.arange(K_all - n_half - 1, -1, -1)])
    # c0 is integrated out exactly: with a Gamma(k, k/m) prior (mean m,
    # sd m/sqrt(k)), or p(c0) ~ 1/c0 without a prior (k = 0), the
    # Poisson likelihood gives
    #   int prod_k Pois(h_k | c0 s_k) p(c0) dc0
    #     ~ prod_k s_k^h_k * Gamma(H + k) / (S + k/m)^(H + k),
    # s_k = F * shape_k, H = sum h, S = sum s_k.
    if c0_prior is not None:
        kk = (c0_m / c0_sd) ** 2
        rate0 = c0_m / c0_sd ** 2
    else:
        kk, rate0 = 0.0, 0.0
    H = float(h.sum())
    if H <= 0:
        raise ValueError("the histogram has no counts")
    lo = np.array([0.0, np.log(t1_lo), 0.0, np.log(t2_lo)])
    hi = np.array([1.0, np.log(t1_hi), a_max, np.log(t2_hi)])

    def shape(X):
        """Mean counts without c0 for an (n, 4) array of parameters."""
        T1 = np.exp(X[:, 1])[:, None, None]
        T2 = np.exp(X[:, 3])[:, None, None]
        a = X[:, 2][:, None, None]
        dip = ((1.0 + a) * _exp_conv_gauss(an[None], T1, s)
               - a * _exp_conv_gauss(an[None], T2, s)) @ wts
        return F * (1.0 - X[:, 0][:, None] * dip[:, mirror])

    def logp(X):
        m = shape(X)
        good = np.all(m > 0, axis=1)
        m = np.where(m > 0, m, 1.0)
        v = np.sum(h * np.log(m), axis=1) - (H + kk) * np.log(
            m.sum(axis=1) + rate0)
        return np.where(good, v, -np.inf)

    # start at the best fit (least squares, as fit_g2_histogram)
    _, ok, det = fit_g2_histogram(h, T_s, r_hat, cfg, t1_bounds=t1_bounds,
                                  t2_bounds=t2_bounds, return_details=True,
                                  c0_prior=c0_prior, check_bounds=False)
    if ok:
        p = det["params"]
        x0 = np.array([p["rho2"], np.log(p["tau1"]), min(p["a"], a_max),
                       np.log(p["tau2"])])
    else:
        x0 = np.array([0.5, 0.5 * (lo[1] + hi[1]), 0.3 * a_max,
                       0.5 * (lo[3] + hi[3])])
    x0 = np.clip(x0, lo + 0.01 * (hi - lo), hi - 0.01 * (hi - lo))
    rng = np.random.default_rng(seed)
    K, acc = stretch_sampler(logp, x0, lo, hi, n_samples, n_burn, rng,
                             n_walkers=n_walkers)
    g = 1.0 - K[:, :, 0].T                    # walkers as chains
    rhat, ess = _rhat_ess(g)
    allx = K.reshape(-1, 4)
    # c0 from its exact conditional, Gamma(H + k, S + k/m)
    S = np.concatenate([shape(allx[i:i + 4096]).sum(axis=1)
                        for i in range(0, allx.shape[0], 4096)])
    c0 = rng.gamma(H + kk, 1.0 / (S + rate0))
    near = ((allx[:, 1] - lo[1] < 0.01) | (hi[1] - allx[:, 1] < 0.01)
            | (allx[:, 3] - lo[3] < 0.01) | (hi[3] - allx[:, 3] < 0.01)
            | (allx[:, 2] > a_max * 0.99))
    params = dict(rho2=allx[:, 0], tau1=np.exp(allx[:, 1]), a=allx[:, 2],
                  tau2=np.exp(allx[:, 3]), c0=c0)
    return ModelPosterior(g2_0=1.0 - allx[:, 0], params=params, rhat=rhat,
                          ess=ess, acceptance=float(acc),
                          near_prior_limit=float(np.mean(near)))
