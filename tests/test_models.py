"""Dip models beyond the three-level formula (new in 0.10.0): each model
against an independent calculation, and the fits' goodness-of-fit
p-values: uniform when the model is right, tiny when it is wrong."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.stats import kstest

from sparq import EmitterSite, HBTConfig, expected_histogram, g2_measured
from sparq.models import (exp_conv_gauss_complex, fit_model, g2_model,
                          goodness_of_fit, model_exponentials)
from sparq.physics import _exp_conv_gauss


def test_complex_blur_matches_real_formula_and_numerics():
    tau = np.linspace(-40, 40, 801)
    for T, s in ((7.0, 0.6), (0.3, 0.05), (250.0, 0.5), (0.05, 0.495),
                 (0.001, 0.495)):
        got = exp_conv_gauss_complex(tau, -1.0 / T, s)
        assert np.abs(got.imag).max() < 1e-15
        assert np.abs(got.real - _exp_conv_gauss(tau, T, s)).max() < 1e-13
    # complex exponent against a brute-force convolution
    lam, s = -0.2 + 1.3j, 0.5
    x = np.linspace(-80, 80, 160001)
    dx = x[1] - x[0]
    f = np.exp(lam * np.abs(x))
    k = np.exp(-x ** 2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi))
    num = np.convolve(f, k, mode="same") * dx
    ana = exp_conv_gauss_complex(tau, lam, s)
    assert np.abs(ana.real - np.interp(tau, x, num.real)).max() < 1e-6
    assert np.abs(ana.imag - np.interp(tau, x, num.imag)).max() < 1e-6
    # lifetimes far below the jitter (0.9.1 gave 0 or NaN here): the
    # fast simulator stays finite
    from sparq import EmitterSite, expected_histogram as eh
    site = EmitterSite(dict(tau1=0.05, tau2=5.0, a=0.4, rho=0.97,
                            rate_kcps=400.0, blinking=False), 1)
    assert np.all(np.isfinite(eh(site, 1.0, HBTConfig())))
    x5 = np.linspace(-20, 20, 400001)
    n5 = np.convolve(np.exp(-np.abs(x5) / 0.05), np.exp(
        -x5 ** 2 / (2 * 0.495 ** 2)) / (0.495 * np.sqrt(2 * np.pi)),
        mode="same") * (x5[1] - x5[0])
    t5 = np.linspace(-5, 5, 41)
    assert np.abs(_exp_conv_gauss(t5, 0.05, 0.495)
                  - np.interp(t5, x5, n5)).max() < 1e-6
    # far out, where exp and erfc alone would overflow, it stays finite
    far = exp_conv_gauss_complex(np.array([1e4]), -0.01 + 2j, 0.3)
    assert np.all(np.isfinite(far))


def test_three_level_and_multi_exponential_match_the_package_formula():
    tau = np.linspace(-60, 60, 241)
    for sig in (0.0, 0.35):
        want = g2_measured(tau, 15.0, 250.0, 0.3, 1, 1.0, sig)
        got = g2_model(tau, "three_level", dict(tau1=15.0, a=0.3,
                                                tau2=250.0), sig)
        assert np.abs(got - want).max() < 1e-12
        got = g2_model(tau, "multi_exponential", (15.0, 0.3, 250.0), sig)
        assert np.abs(got - want).max() < 1e-12
    # two shoulders: the sum of exponentials, 0 at zero delay
    t = np.linspace(0, 100, 101)
    g = g2_model(t, "multi_exponential", (5.0, 0.4, 50.0, 0.2, 400.0))
    want = 1 - 1.6 * np.exp(-t / 5) + 0.4 * np.exp(-t / 50) \
        + 0.2 * np.exp(-t / 400)
    assert np.abs(g - want).max() < 1e-14 and abs(g[0]) < 1e-14
    with pytest.raises(ValueError):
        model_exponentials("multi_exponential", (5.0, 0.4))
    with pytest.raises(ValueError):
        model_exponentials("nonsense", (1.0,))


def _kimble_mandel(t, om, ga):
    mu = np.sqrt(om ** 2 - ga ** 2 / 16 + 0j)
    return np.real(1 - np.exp(-3 * ga * t / 4)
                   * (np.cos(mu * t) + 3 * ga / (4 * mu) * np.sin(mu * t)))


def _bloch(t, om, ga, gd, de):
    """Independent path: integrate the optical Bloch equations for
    rho_ee and rho_eg from the ground state, and divide by their value
    after a long time."""
    def rhs(_, y):
        ree, re_r, re_i = y
        reg = re_r + 1j * re_i
        rge = np.conj(reg)
        dree = np.real(-1j * (om / 2) * (rge - reg)) - ga * ree
        dreg = (1j * de * reg - 1j * (om / 2) * (1 - 2 * ree)
                - (ga / 2 + gd) * reg)
        return [dree, dreg.real, dreg.imag]
    sol = solve_ivp(rhs, (0, t.max()), [0, 0, 0], t_eval=t, rtol=1e-11,
                    atol=1e-13, method="DOP853")
    late = solve_ivp(rhs, (0, 400.0 / ga), [0, 0, 0], rtol=1e-11,
                     atol=1e-13, method="DOP853").y[0, -1]
    return sol.y[0] / late


def test_coherent_model_matches_closed_form_and_bloch_equations():
    t = np.linspace(0, 30, 301)
    for om, ga in ((2.0, 0.5), (0.3, 0.5), (0.1, 1.0), (3.0, 1.4)):
        g = g2_model(t, "coherent", dict(omega=om, gamma=ga,
                                         gamma_deph=0.0, delta=0.0))
        assert np.abs(g - _kimble_mandel(t, om, ga)).max() < 1e-12
    for om, ga, gd, de in ((3.0, 1.4, 0.3, 0.0), (1.0, 0.5, 0.1, 0.7)):
        g = g2_model(t, "coherent", dict(omega=om, gamma=ga,
                                         gamma_deph=gd, delta=de))
        assert np.abs(g - _bloch(t, om, ga, gd, de)).max() < 1e-7
        assert abs(g[0]) < 1e-12


def test_fit_p_values_are_uniform_when_the_model_is_right():
    cfg = HBTConfig()
    site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                            rho=0.95, blinking=False), 1)
    mu = expected_histogram(site, 30.0, cfg)
    ps, errs = [], []
    for s in range(20):
        h = np.random.default_rng(s).poisson(mu).astype(float)
        r = fit_model(h, 30.0, 150e3, cfg, c0_prior=(1.0, 0.01))
        ps.append(r["p_value"])
        errs.append(r["g2_0"] - site.g2_0)
    assert kstest(ps, "uniform").pvalue > 0.01
    assert np.mean(np.array(ps) < 0.05) <= 0.2
    assert np.abs(errs).max() < 0.06


def test_wrong_model_is_rejected_right_model_recovers_g2():
    """A coherently driven emitter (Rabi oscillations) on a 0.1 ns grid:
    the three-level fit is rejected and misreads g2(0); the coherent
    fit is accepted and recovers it."""
    cfg = HBTConfig(tau_max=10.05, n_bins=201, sigma_irf=0.05)
    p = dict(omega=3.0, gamma=1.4, gamma_deph=0.1, delta=0.0)
    nodes, w = cfg.bin_nodes()
    T, r = 120.0, 2e5
    flat = (r / 2) ** 2 * cfg.bin_width * 1e-9 * T
    mu = flat * (g2_model(nodes, "coherent", p, cfg.sigma_irf, rho2=0.9)
                 @ w)
    h = np.random.default_rng(0).poisson(mu).astype(float)
    f3 = fit_model(h, T, r, cfg, "three_level", c0_prior=(1.0, 0.01))
    fc = fit_model(h, T, r, cfg, "coherent", fixed=dict(delta=0.0),
                   c0_prior=(1.0, 0.01))
    assert f3["p_value"] < 1e-3
    assert fc["p_value"] > 0.01
    assert abs(fc["g2_0"] - 0.1) < 0.05
    assert abs(f3["g2_0"] - 0.1) > abs(fc["g2_0"] - 0.1)
    assert fc["aic"] < f3["aic"]
    # the simulation-based p-value agrees with the chi-square one on
    # which model to reject
    b = goodness_of_fit(f3, n_boot=19, seed=1)
    assert b["p_value"] == pytest.approx(1 / 20)      # none as large
    with pytest.raises(ValueError):
        fit_model(h, T, r, cfg, "coherent", fixed=dict(nonsense=1.0))
