"""Anchors for the exact-Poisson Bayesian g2 posterior: the hand-built
beta-prime law against SciPy's independent implementation, unit mass
and closed-form moments by quadrature, the closed form against a
direct numerical marginalization of the exact Poisson likelihood,
Monte-Carlo Gamma-ratio agreement, credible-interval coverage on
twin-generated histograms, and the triage verdict ordering."""
import numpy as np
import pytest
from scipy import integrate, stats

from sparq import EmitterSite, HBTConfig, bayesian_g2, expected_histogram
from sparq.bayes import G2Posterior

CFG = HBTConfig()


def _params(rate=200.0, tau1=15.0, tau2=250.0, a=0.3, rho=0.95):
    return dict(tau1=tau1, tau2=tau2, a=a, rho=rho, rate_kcps=rate,
                blinking=False)


def _hist(site, T_s, rng):
    return rng.poisson(expected_histogram(site, T_s, CFG)).astype(float)


def _post(k0=7.0, n0=1, kr=41300.0, nr=44):
    return G2Posterior(alpha0=0.5 + k0, beta0=float(n0),
                       alphar=0.5 + kr, betar=float(nr),
                       k0=k0, n0=n0, kr=kr, nr=nr)


def test_pdf_cdf_ppf_match_scipy_betaprime():
    p = _post()
    ref = stats.betaprime(p.alpha0, p.alphar,
                          scale=p.betar / p.beta0)
    g = np.linspace(1e-4, 3.0, 300)
    assert np.allclose(p.pdf(g), ref.pdf(g), rtol=1e-12, atol=1e-300)
    assert np.allclose(p.cdf(g), ref.cdf(g), rtol=0, atol=1e-12)
    q = np.array([0.025, 0.5, 0.975])
    assert np.allclose(p.ppf(q), ref.ppf(q), rtol=1e-9)
    # inverse identity
    assert np.allclose(p.cdf(p.ppf(q)), q, rtol=0, atol=1e-10)


def test_density_normalizes_and_matches_closed_form_moments():
    p = _post(k0=4.0, kr=9000.0, nr=40)
    total, _ = integrate.quad(p.pdf, 0.0, np.inf)
    assert abs(total - 1.0) < 1e-8
    m_quad, _ = integrate.quad(lambda g: g * p.pdf(g), 0.0, np.inf)
    assert abs(m_quad - p.mean()) < 1e-8
    # mode is the stationary point of the density
    g0 = p.mode()
    eps = 1e-6
    assert p.pdf(g0) >= p.pdf(g0 + eps) and p.pdf(g0) >= p.pdf(g0 - eps)


def test_closed_form_equals_direct_poisson_marginalization():
    """p(g) = int Gamma_pdf(g*lam; a0, b0) * lam * Gamma_pdf(lam) dlam:
    the ratio density by 1-D quadrature over the nuisance rate, an
    integral the module never evaluates, equals the closed form."""
    p = _post(k0=7.0, n0=1, kr=850.0, nr=44)
    fa = stats.gamma(p.alpha0, scale=1.0 / p.beta0)
    fb = stats.gamma(p.alphar, scale=1.0 / p.betar)
    lo, hi = fb.ppf(1e-13), fb.ppf(1.0 - 1e-13)
    peak = p.alphar / p.betar
    for g in (0.05, 0.3, 0.7, 1.2):
        val, err = integrate.quad(
            lambda lam: fa.pdf(g * lam) * lam * fb.pdf(lam), lo, hi,
            points=[peak], limit=200)
        assert abs(val - p.pdf(g)) < 1e-9 + 10 * err


def test_monte_carlo_gamma_ratio_matches_cdf():
    p = _post(k0=12.0, kr=4000.0, nr=44)
    rng = np.random.default_rng(0)
    x = rng.gamma(p.alpha0, 1.0 / p.beta0, 200000)
    y = rng.gamma(p.alphar, 1.0 / p.betar, 200000)
    r = x / y
    for g in (p.ppf(0.1), p.ppf(0.5), p.ppf(0.9)):
        emp = float(np.mean(r <= g))
        assert abs(emp - p.cdf(g)) < 5e-3


def test_windows_and_refusals():
    site = EmitterSite(_params(), 1)
    rng = np.random.default_rng(1)
    h = _hist(site, 50.0, rng)
    p = bayesian_g2(h, CFG)
    assert p.n0 == 1 and p.nr == int(
        (np.abs(CFG.bin_centers) >= 0.65 * CFG.tau_max).sum())
    assert p.k0 == float(h[CFG.n_bins // 2])
    p3 = bayesian_g2(h, CFG, n_center_bins=3)
    assert p3.n0 == 3 and p3.k0 >= p.k0
    with pytest.raises(ValueError):
        bayesian_g2(h[:-1], CFG)                     # wrong length
    with pytest.raises(ValueError):
        bayesian_g2(h, CFG, n_center_bins=2)         # even window
    bad = h.copy()
    bad[0] = -1.0
    with pytest.raises(ValueError):
        bayesian_g2(bad, CFG)
    with pytest.raises(ValueError):
        bayesian_g2(h, CFG, prior=(0.0, 0.0))        # improper shape
    with pytest.raises(ValueError):
        bayesian_g2(h, CFG, lo_frac=1.5)             # empty reference
    with pytest.raises(ValueError):
        bayesian_g2(h, CFG, n_center_bins=CFG.n_bins)  # overlap


def test_posterior_concentrates_and_covers_on_twin_histograms():
    """On histograms drawn from the twin the credible interval covers
    the true central-bin ratio at about its nominal level, and the
    posterior mean converges to the high-count ratio."""
    site = EmitterSite(_params(), 1)
    mu = expected_histogram(site, 200.0, CFG)
    m0 = CFG.n_bins // 2
    mr = np.abs(CFG.bin_centers) >= 0.65 * CFG.tau_max
    g_true = float(mu[m0] / mu[mr].mean())
    rng = np.random.default_rng(3)
    hits = 0
    reps = 200
    for _ in range(reps):
        p = bayesian_g2(rng.poisson(mu).astype(float), CFG)
        lo, hi = p.credible_interval(0.95)
        hits += int(lo <= g_true <= hi)
    assert 0.90 <= hits / reps <= 1.0
    # high-count limit: posterior mean -> observed ratio
    h = rng.poisson(mu * 50).astype(float)
    p = bayesian_g2(h, CFG)
    raw = (p.k0 / p.n0) / (p.kr / p.nr)
    assert abs(p.mean() - raw) / raw < 1e-3


def test_verdict_probability_orders_single_against_double():
    rng = np.random.default_rng(5)
    single = EmitterSite(_params(), 1)
    double = EmitterSite(_params(), 2)
    p1 = bayesian_g2(_hist(single, 100.0, rng), CFG)
    p2 = bayesian_g2(_hist(double, 100.0, rng), CFG)
    assert p1.prob_below(0.5) > 0.9
    assert p2.prob_below(0.5) < 0.5
    assert p1.prob_below(0.5) > p2.prob_below(0.5)
