"""Posterior of g2(0) itself under the three-level model (new in 0.10.0):
the sampler against a closed-form posterior, and the credible intervals
against the truth of simulated data."""
import warnings

import numpy as np
import pytest

from sparq import (EmitterSite, HBTConfig, bayesian_g2, expected_histogram,
                   profile_likelihood_ci)
from sparq.posterior import bayesian_g2_model, stretch_sampler


def test_sampler_reproduces_the_closed_form_ratio_posterior():
    """Two Poisson rates with Jeffreys priors: the ratio has the exact
    beta-prime posterior of `bayesian_g2`. Sampling the two rates with
    the ensemble sampler must give the same distribution."""
    cfg = HBTConfig()
    hist = np.full(cfg.n_bins, 100.0)
    hist[cfg.n_bins // 2] = 30.0
    exact = bayesian_g2(hist, cfg)
    a0, b0, ar, br = exact.alpha0, exact.beta0, exact.alphar, exact.betar

    def logp(X):
        m, lam = X[:, 0], X[:, 1]
        return ((a0 - 1) * np.log(m) - b0 * m
                + (ar - 1) * np.log(lam) - br * lam)

    x0 = np.array([a0 / b0, ar / br])
    rng = np.random.default_rng(0)
    K, acc = stretch_sampler(logp, x0, [1e-9, 1e-9], [1e3, 1e3], 3000,
                             500, rng, n_walkers=16, spread=1e-4)
    g = (K[:, :, 0] / K[:, :, 1]).ravel()
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        x = exact.ppf(q)
        assert abs(np.mean(g < x) - q) < 0.02
    assert 0.3 < acc < 0.9
    with pytest.raises(ValueError):
        stretch_sampler(logp, x0, [1e-9, 1e-9], [1e3, 1e3], 10, 10, rng,
                        n_walkers=3)


def test_credible_intervals_cover_the_truth():
    cfg = HBTConfig()
    site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                            rho=0.95, blinking=False), 1)
    mu = expected_histogram(site, 30.0, cfg)
    cover = 0
    for s in range(15):
        h = np.random.default_rng(s).poisson(mu).astype(float)
        post = bayesian_g2_model(h, 30.0, 150e3, cfg, c0_prior=(1.0, 0.01),
                                 n_samples=600, n_burn=400, seed=s)
        lo, hi = post.credible_interval(0.9)
        cover += lo <= site.g2_0 <= hi
        # c0 follows its prior (the data add little to a 1 % prior)
        assert abs(post.params["c0"].mean() - 1.0) < 0.01
    assert cover >= 11          # P(cover <= 10 | 0.9, 15) is about 1 %


def test_posterior_agrees_with_profile_likelihood():
    cfg = HBTConfig()
    site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                            rho=0.95, blinking=False), 2)
    h = np.random.default_rng(4).poisson(
        expected_histogram(site, 30.0, cfg)).astype(float)
    post = bayesian_g2_model(h, 30.0, 150e3, cfg, c0_prior=(1.0, 0.01),
                             n_samples=800, n_burn=400, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prof = profile_likelihood_ci(h, 30.0, 150e3, cfg,
                                     c0_prior=(1.0, 0.01))
    lo, hi = post.credible_interval(0.95)
    assert abs(post.median() - prof["g2_hat"]) < 0.03
    assert abs(lo - prof["lo"]) < 0.04 and abs(hi - prof["hi"]) < 0.04
    assert post.rhat < 1.1 and post.ess > 100
    assert post.prob_below(0.5) < 0.5          # a pair is not certified


def test_without_a_flat_level_prior_the_interval_is_wider():
    cfg = HBTConfig()
    site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                            rho=0.95, blinking=False), 1)
    h = np.random.default_rng(3).poisson(
        expected_histogram(site, 30.0, cfg)).astype(float)
    kw = dict(n_samples=600, n_burn=400, seed=2)
    pinned = bayesian_g2_model(h, 30.0, 150e3, cfg, c0_prior=(1.0, 0.01),
                               **kw)
    free = bayesian_g2_model(h, 30.0, 150e3, cfg, **kw)
    w_p = np.subtract(*pinned.credible_interval(0.9)[::-1])
    w_f = np.subtract(*free.credible_interval(0.9)[::-1])
    assert w_f > 1.5 * w_p
    with pytest.raises(ValueError):
        bayesian_g2_model(h[:-1], 30.0, 150e3, cfg)
    with pytest.raises(ValueError):
        bayesian_g2_model(h, 30.0, 150e3, cfg, c0_prior=(1.0, -0.1))
