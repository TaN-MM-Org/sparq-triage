"""Exact-Poisson Bayesian posterior for the raw g2(0) of an HBT histogram.

The package's three interval tools now cover the three inferential
schools honestly: the parametric bootstrap (:func:`analyze_histogram`)
propagates shot noise through the full fitting pipeline, the profile
likelihood (:func:`profile_likelihood_ci`) gives Wilks-calibrated
frequentist intervals under the three-level model, and this module
gives the EXACT finite-count Bayesian posterior of the model-free
quantity every experimental paper quotes first: the raw central-window
g2, the ratio of the coincidence rate at zero delay to the flat
(uncorrelated) level.

The model is stated exactly, because exactness is the point.  Let the
n0 central-window bins carry counts that are Poisson with a common
mean mu0 per bin, and the nr far-delay reference bins carry counts
Poisson with mean lam per bin; the estimand is g = mu0 / lam.  With
independent Gamma(a, b) priors on the two rates (default a = 1/2,
b = 0: Jeffreys' prior for a Poisson mean) the posteriors are the
conjugate closed forms

    mu0 | data ~ Gamma(a + k0, b + n0),
    lam | data ~ Gamma(a + kr, b + nr),

with k0, kr the summed window counts -- and the posterior of the RATIO
of two independent Gamma variables is a scaled beta-prime law:

    (beta0 / betar) * g ~ BetaPrime(alpha0, alphar),

so the density, distribution function, quantiles, moments and any
posterior probability (for instance P[g < 1/2 | data], the
single-emitter triage verdict) are closed-form expressions in the four
counts, evaluated through `scipy.special` -- no sampling, no
optimizer, no asymptotics.  Acquisition time and detection rates
cancel in the ratio, so none need be supplied.

What this estimates, stated plainly: g is the histogram's coincidence
rate in the chosen central window relative to the reference level --
the "raw g2(0)" convention.  Averaging over a window of nonzero width
sits ON TOP of the antibunching dip, so for a dip wider than the
window g upper-bounds the true g2(0); with the default single central
bin (1 ns here) against nanosecond-scale lifetimes the bias is small
but not zero, and it is a property of the estimand, not of the
inference.  For dip-shape-aware estimates of g2(0) itself, use
`profile_likelihood_ci` (same exact Poisson likelihood, three-level
model) -- the two tools answer different questions and the docstrings
say which.

Anchors, asserted in the tests rather than stated: the hand-built
density and distribution function agree with `scipy.stats.betaprime`
(an independent implementation) to 1e-12; the density integrates to
one and reproduces the closed-form mean; the closed form agrees with
a direct numerical marginalization of the exact Poisson likelihood
over the nuisance rate (an integral this module never uses); seeded
Monte-Carlo ratios of Gamma draws match the distribution function;
posterior credible intervals achieve their nominal coverage on
twin-generated histograms; and the verdict probability orders single-
against two-emitter sites correctly.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.special import betainc, betaincinv, betaln

from .physics import HBTConfig

__all__ = ["G2Posterior", "bayesian_g2"]


@dataclasses.dataclass
class G2Posterior:
    """Closed-form posterior of the raw central-window g2 ratio.

    alpha0, beta0 : Gamma posterior (shape, rate) of the central-window
        per-bin mean mu0.
    alphar, betar : Gamma posterior of the reference per-bin mean lam.
    k0, n0, kr, nr : the summed counts and bin counts behind them.

    The ratio g = mu0/lam follows the scaled beta-prime law
    (beta0/betar) g ~ BetaPrime(alpha0, alphar); all methods evaluate
    that law exactly.
    """

    alpha0: float
    beta0: float
    alphar: float
    betar: float
    k0: float
    n0: int
    kr: float
    nr: int

    @property
    def _s(self):
        """Scale: g = _s * BetaPrime(alpha0, alphar)."""
        return self.betar / self.beta0

    def pdf(self, g):
        """Posterior density p(g | data), exactly."""
        g = np.asarray(g, dtype=float)
        r = self.beta0 * g / self.betar
        a, c = self.alpha0, self.alphar
        with np.errstate(divide="ignore"):
            logp = ((a - 1.0) * np.log(r) - (a + c) * np.log1p(r)
                    - betaln(a, c) + np.log(self.beta0 / self.betar))
        out = np.where(g > 0.0, np.exp(logp), 0.0)
        return float(out) if out.ndim == 0 else out

    def cdf(self, g):
        """P[g' <= g | data] via the regularized incomplete beta."""
        g = np.asarray(g, dtype=float)
        r = self.beta0 * np.maximum(g, 0.0) / self.betar
        x = r / (1.0 + r)
        out = betainc(self.alpha0, self.alphar, x)
        out = np.where(g > 0.0, out, 0.0)
        return float(out) if out.ndim == 0 else out

    def ppf(self, q):
        """Posterior quantile function (inverse of `cdf`), exactly."""
        q = np.asarray(q, dtype=float)
        if np.any((q <= 0.0) | (q >= 1.0)):
            raise ValueError("quantile levels must lie in (0, 1)")
        x = betaincinv(self.alpha0, self.alphar, q)
        out = (self.betar / self.beta0) * x / (1.0 - x)
        return float(out) if out.ndim == 0 else out

    def mean(self):
        """E[g | data] = (alpha0/beta0) * betar/(alphar - 1); requires
        alphar > 1 (otherwise the posterior mean does not exist and
        this raises rather than returning a number)."""
        if self.alphar <= 1.0:
            raise ValueError(
                "posterior mean does not exist for alphar <= 1 "
                "(essentially no reference counts); quote the median")
        return (self.alpha0 / self.beta0) * self.betar / (self.alphar - 1.0)

    def median(self):
        return self.ppf(0.5)

    def mode(self):
        """Posterior mode (betar/beta0)(alpha0 - 1)/(alphar + 1) for
        alpha0 >= 1, else 0."""
        if self.alpha0 < 1.0:
            return 0.0
        return (self.betar / self.beta0) * (self.alpha0 - 1.0) \
            / (self.alphar + 1.0)

    def credible_interval(self, level=0.95):
        """Equal-tailed credible interval at the given level."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie in (0, 1)")
        h = 0.5 * (1.0 - level)
        return self.ppf(h), self.ppf(1.0 - h)

    def prob_below(self, threshold=0.5):
        """P[g < threshold | data] -- the single-emitter verdict
        probability at the conventional threshold 1/2."""
        return float(self.cdf(threshold))


def bayesian_g2(hist, cfg: HBTConfig | None = None, prior=(0.5, 0.0),
                n_center_bins: int = 1, lo_frac: float = 0.65
                ) -> G2Posterior:
    """Exact Bayesian posterior of the raw central-window g2 ratio.

    Parameters
    ----------
    hist : (n_bins,) coincidence counts on the grid of `cfg` (as
        produced by `expected_histogram`-shaped acquisitions or
        `rebin_real`), centered on the dip.
    cfg : the histogram grid; the package default when omitted.
    prior : (a, b) of the independent Gamma(a, b) priors on the two
        per-bin Poisson means.  The default (1/2, 0) is Jeffreys'
        prior; (1, 0) is the flat-on-rate prior; any a > 0, b >= 0 is
        accepted.  The induced prior on the ratio is the corresponding
        beta-prime law -- a prior placed directly on g would break the
        closed form, so this module does not offer one.
    n_center_bins : number of bins in the central window (odd grid:
        1 keeps exactly the zero-delay bin; 3 adds one neighbor each
        side).  Wider windows average further up the dip walls; the
        module docstring discusses the estimand honestly.
    lo_frac : reference window is |tau| >= lo_frac * tau_max, the same
        convention as `robust_flat_rate`.

    Raises ValueError on malformed input: counts negative or
    non-finite, windows that overlap or are empty, or a grid without a
    zero-centered bin (even n_bins).
    """
    if cfg is None:
        cfg = HBTConfig()
    y = np.asarray(hist, dtype=float)
    if y.ndim != 1 or y.size != cfg.n_bins:
        raise ValueError(f"hist must have cfg.n_bins = {cfg.n_bins} bins")
    if not np.all(np.isfinite(y)) or np.any(y < 0.0):
        raise ValueError("counts must be finite and nonnegative")
    if cfg.n_bins % 2 == 0:
        raise ValueError("cfg.n_bins must be odd (a bin centered at 0)")
    if n_center_bins < 1 or n_center_bins % 2 == 0:
        raise ValueError("n_center_bins must be odd and >= 1")
    tau = cfg.bin_centers
    half = (n_center_bins // 2 + 0.5) * cfg.bin_width
    m0 = np.abs(tau) < half
    mr = np.abs(tau) >= lo_frac * cfg.tau_max
    if int(m0.sum()) != n_center_bins:
        raise ValueError("central window exceeds the histogram")
    if not mr.any():
        raise ValueError("reference window is empty; lower lo_frac")
    if np.any(m0 & mr):
        raise ValueError("central and reference windows overlap")
    a, b = float(prior[0]), float(prior[1])
    if a <= 0.0 or b < 0.0:
        raise ValueError("prior must have shape a > 0 and rate b >= 0")
    k0 = float(y[m0].sum())
    kr = float(y[mr].sum())
    n0 = int(m0.sum())
    nr = int(mr.sum())
    return G2Posterior(alpha0=a + k0, beta0=b + n0, alphar=a + kr,
                       betar=b + nr, k0=k0, n0=n0, kr=kr, nr=nr)
