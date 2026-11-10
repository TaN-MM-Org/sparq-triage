"""Dip models beyond the three-level formula, their fits, and a test of
whether a fit describes the data (new in 0.10.0).

Every model here is written as a sum of exponentials in the delay,

    g2_1(tau) = 1 + sum_k c_k exp(lambda_k |tau|),   Re(lambda_k) < 0,

with g2_1(0) = 0 for one emitter. The exponents may be complex (a
damped oscillation). The package's three-level formula is the special
case of two real exponents. Written this way, every model gets the
same exact treatment of the detectors' timing jitter (the Gaussian blur
of each exponential has a closed form, also for complex exponents; see
`exp_conv_gauss_complex`), of bin averaging (`HBTConfig.bin_nodes`),
and of background and emitter number:

    g2(tau) = c0 [1 + rho2 (g2_1_blurred(tau) - 1)],   g2(0) = 1 - rho2,

so rho2 = rho^2 / N and, as in `fit_g2_histogram`, the fitted g2(0) is
the value without jitter.

Models (each a function of named parameters in ns or 1/ns returning
(c, lambda)):

* ``"three_level"`` -- tau1, a, tau2: the package's formula.
* ``"multi_exponential"`` -- tau1 and m shoulders (a_1, tau_b1, ...,
  a_m, tau_bm): the form of any emitter with several dark states
  described by rate equations whose rates give real exponents.
* ``"coherent"`` -- omega (Rabi angular frequency), gamma (1/T1),
  gamma_deph (pure dephasing), delta (detuning), all in 1/ns: a
  two-level emitter driven coherently (resonance fluorescence),
  from the exact Lindblad master equation. Without dephasing and on
  resonance it is the closed form
      g2 = 1 - exp(-3 gamma tau/4) [cos(mu tau) + 3 gamma/(4 mu) sin(mu tau)],
      mu = sqrt(omega^2 - gamma^2/16)
  (Kimble, Dagenais and Mandel, Phys. Rev. Lett. 39, 691 (1977);
  Carmichael and Walls, J. Phys. B 9, 1199 (1976)), which the tests
  check to 1e-12.

`fit_model` fits any of them by the exact Poisson likelihood, and
reports the Poisson deviance with its chi-square p-value and, on
request, a parametric-bootstrap p-value (`goodness_of_fit`) and (new in
0.15.0, `level`) a profile-likelihood interval for g2(0). A small
p-value says the model does not describe the dip; the tests check
that the p-values are uniform when the model is right and tiny when a
coherently driven emitter is fitted with the three-level formula.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import wofz
from scipy.stats import chi2 as _chi2

from .physics import HBTConfig

__all__ = ["MODELS", "model_exponentials", "g2_model",
           "exp_conv_gauss_complex", "coherent_liouvillian",
           "fit_model", "goodness_of_fit"]


# ----------------------------------------------------------------------
# exponential representations
# ----------------------------------------------------------------------

def _three_level(tau1, a, tau2):
    return (np.array([-(1.0 + a), a], complex),
            np.array([-1.0 / tau1, -1.0 / tau2], complex))


def _multi_exponential(tau1, *shoulders):
    if len(shoulders) % 2:
        raise ValueError("multi_exponential takes tau1 and pairs "
                         "(a_k, tau_bk)")
    a = np.asarray(shoulders[0::2], float)
    tb = np.asarray(shoulders[1::2], float)
    c = np.concatenate([[-(1.0 + a.sum())], a]).astype(complex)
    lam = np.concatenate([[-1.0 / tau1], -1.0 / tb]).astype(complex)
    return c, lam


def _eig_terms(M, p0, obs, pss_obs):
    """c_k, lambda_k of obs . exp(M t) p0 / pss_obs over the non-zero
    eigenvalues of M (the zero eigenvalue gives the constant 1)."""
    w, V = np.linalg.eig(M)
    coef = np.linalg.solve(V, p0)
    amp = (obs @ V) * coef / pss_obs
    i0 = int(np.argmin(np.abs(w)))
    keep = np.arange(w.size) != i0
    return amp[keep].astype(complex), w[keep].astype(complex)


def coherent_liouvillian(omega, gamma, gamma_deph=0.0, delta=0.0):
    """4x4 Lindblad superoperator of a driven two-level emitter (1/ns).

    Basis |g> = 0, |e> = 1, density matrix stacked column by column.
    H = -delta |e><e| + (omega/2)(|e><g| + |g><e|) in the frame of the
    laser; decay |g><e| at rate gamma; pure dephasing |e><e| at rate
    2 gamma_deph, so that the coherence decays at gamma/2 + gamma_deph.
    """
    sm = np.array([[0, 1], [0, 0]], complex)          # |g><e|
    see = np.array([[0, 0], [0, 1]], complex)
    H = -delta * see + 0.5 * omega * (sm + sm.conj().T)
    eye = np.eye(2)

    def spre(A):
        return np.kron(eye, A)

    def spost(A):
        return np.kron(A.T, eye)

    L = -1j * (spre(H) - spost(H))
    for c, r in ((sm, gamma), (see, 2.0 * gamma_deph)):
        cd = c.conj().T
        L = L + r * (spre(c) @ spost(cd) - 0.5 * spre(cd @ c)
                     - 0.5 * spost(cd @ c))
    return L


def _coherent(omega, gamma, gamma_deph=0.0, delta=0.0):
    L = coherent_liouvillian(omega, gamma, gamma_deph, delta)
    w, V = np.linalg.eig(L)
    i0 = int(np.argmin(np.abs(w)))
    ss = V[:, i0].reshape(2, 2, order="F")
    ss = ss / np.trace(ss)
    rho0 = np.array([1, 0, 0, 0], complex)            # |g><g|
    obs = np.array([0, 0, 0, 1], complex)             # rho_ee (index 3)
    return _eig_terms(L, rho0, obs, np.real(ss[1, 1]))


MODELS = {
    "three_level": (_three_level, ("tau1", "a", "tau2")),
    "coherent": (_coherent, ("omega", "gamma", "gamma_deph", "delta")),
    "multi_exponential": (_multi_exponential, None),  # tau1, a1, tb1, ...
}


def model_exponentials(model, params):
    """(c, lambda) of a model; params a dict (or sequence in the model's
    parameter order)."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from "
                         f"{sorted(MODELS)}")
    fn, names = MODELS[model]
    if isinstance(params, dict):
        if names is None:
            raise ValueError("give multi_exponential parameters as a "
                             "sequence (tau1, a1, tau_b1, ...)")
        args = [float(params[n]) for n in names]
    else:
        args = [float(x) for x in params]
    return fn(*args)


# ----------------------------------------------------------------------
# jitter and bins
# ----------------------------------------------------------------------

def exp_conv_gauss_complex(tau, lam, s):
    """exp(lam |tau|) blurred by a normalized Gaussian of std s, for
    complex lam with Re(lam) < 0 (closed form via the Faddeeva function
    w(z) = exp(-z^2) erfc(-i z), which stays finite where exp and erfc
    alone overflow). Equals `sparq.physics._exp_conv_gauss` for real
    lam = -1/T (tested to 1e-13, also for lifetimes far below the
    jitter)."""
    tau = np.asarray(tau, dtype=float)
    lam = complex(lam)
    if s <= 1e-12:
        return np.exp(lam * np.abs(tau))
    mu = -lam
    out = np.zeros(tau.shape, complex)
    g = np.exp(-tau ** 2 / (2.0 * s * s))
    for sg in (1.0, -1.0):
        u = (mu * s - sg * tau / s) / np.sqrt(2.0)
        pos = np.real(u) >= 0.0
        term = np.empty(tau.shape, complex)
        # e^{mu^2 s^2/2 - sg mu tau} erfc(u) = e^{-tau^2/2s^2} w(i u)
        term[pos] = g[pos] * wofz(1j * u[pos])
        # erfc(u) = 2 - erfc(-u) where Re(u) < 0
        n = ~pos
        term[n] = (2.0 * np.exp(mu * mu * s * s / 2.0 - sg * mu * tau[n])
                   - g[n] * wofz(-1j * u[n]))
        out += 0.5 * term
    return out


def g2_model(tau, model, params, sigma_irf=0.0, rho2=1.0):
    """Normalized g2 of one emitter model, blurred by two detectors'
    jitter (sigma_pair = sqrt(2) sigma_irf), with dip depth rho2:
    1 + rho2 (g2_1 - 1)."""
    c, lam = model_exponentials(model, params)
    s = np.sqrt(2.0) * float(sigma_irf)
    tau = np.asarray(tau, dtype=float)
    acc = np.zeros(tau.shape, complex)
    for ck, lk in zip(c, lam):
        acc += ck * exp_conv_gauss_complex(tau, lk, s)
    return 1.0 + rho2 * np.real(acc)


# ----------------------------------------------------------------------
# fits by the exact Poisson likelihood
# ----------------------------------------------------------------------

_DEFAULT_STARTS = {
    "three_level": [dict(tau1=15.0, a=0.3, tau2=250.0),
                    dict(tau1=5.0, a=1.0, tau2=100.0),
                    dict(tau1=1.0, a=0.3, tau2=30.0)],
    "coherent": [dict(omega=1.0, gamma=0.5, gamma_deph=0.05, delta=0.0),
                 dict(omega=0.3, gamma=0.3, gamma_deph=0.05, delta=0.0),
                 dict(omega=3.0, gamma=1.0, gamma_deph=0.1, delta=0.0)],
}


def _poisson_deviance(h, mu):
    mu = np.maximum(mu, 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(h > 0, h * np.log(h / mu), 0.0)
    return float(2.0 * np.sum(t - (h - mu)))


def fit_model(hist, T_s: float, r_hat: float,
              cfg: HBTConfig | None = None, model: str = "three_level",
              starts=None, fixed=None, c0_prior=None,
              n_shoulders: int = 1, level: float | None = None) -> dict:
    """Fit a dip model to a histogram on the grid `cfg` by the exact
    Poisson likelihood.

    hist : Poisson counts per bin, centered on the dip: the native
        bins, or bins re-binned with `rebin_real(..., method="whole")`
        at a whole multiple of the input width. (Split bins vary less
        than Poisson counts, which makes the p-values too large.)
    T_s, r_hat : acquisition time (s) and the singles rate (counts/s,
        both detectors) that set the flat level (r_hat/2)^2 w T_s.
    model : "three_level", "multi_exponential" (with `n_shoulders`)
        or "coherent".
    starts : list of dicts (or sequences for multi_exponential) of model
        parameters to start from; defaults cover NV- to quantum-dot-like
        scales. Each start is tried with dip depth 0.9 and 0.5.
    fixed : dict of model parameters held fixed (e.g. delta=0.0 or
        gamma_deph=0.0 for "coherent").
    c0_prior : optional (mean, sd) Gaussian constraint on the flat-level
        scale c0, as in `profile_likelihood_ci`.
    level : confidence level of a profile-likelihood interval for g2(0)
        (new in 0.15.0; None, the default, gives none). The interval is
        where the profile deviance 2 [NLL(g2) - NLL_min], with every
        other parameter (the model's and c0) fitted again at each g2,
        stays below the chi-square(1) quantile of `level` (Wilks), within
        0 <= g2(0) <= 1. It belongs to the chosen model: for a model that
        the p-value rejects it means little. The profile is searched
        outward from the best fit, each point started from its
        neighbor, and each end refined by brentq to 1e-4. If the profile
        finds a lower NLL than the fit (the local search had stopped
        short), the ends are measured from that lower value and
        `profile_improved` is True.

    All free model parameters are positive and fitted on a log scale;
    rho2 lies in [0, 1] and c0 in [1e-3, 1e3]. The fit is a local
    search (L-BFGS-B from the starts) of half the Poisson deviance (from
    0.15.0; up to 0.14.0 of a number of the order of the total counts,
    on which the search could stop short of the best fit). When the
    model fits badly its likelihood has several minima, and which one
    the search ends in depends on the SciPy version: for the three-level
    fit of README example 10 the deviance is 345.74 with SciPy 1.18 and
    348.81 with SciPy 1.10 (p-values 2.2e-10 and 1.1e-10; both reject
    the model).

    Returns a dict: model, params (the model's), rho2, c0, g2_0
    (= 1 - rho2), mu (the fitted mean counts), deviance (Poisson),
    dof (bins minus free parameters), p_value (the chi-square tail
    probability of the deviance: approximate, good when bins hold more
    than a few counts; see `goodness_of_fit` for an exact-by-simulation
    version), aic, n_free, ok; with `level` also g2_0_lo, g2_0_hi (0 or
    NaN, respectively, when the interval reaches 0 or does not close
    below 1), level and profile_improved.
    """
    if cfg is None:
        cfg = HBTConfig()
    h = np.asarray(hist, dtype=float)
    if h.ndim != 1 or h.size != cfg.n_bins:
        raise ValueError(f"hist must have cfg.n_bins = {cfg.n_bins} bins")
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        raise ValueError("hist must be finite and non-negative")
    flat = (0.5 * float(r_hat)) ** 2 * (cfg.bin_width * 1e-9) * float(T_s)
    if not (float(r_hat) > 0 and float(T_s) > 0 and np.isfinite(flat)):
        raise ValueError("r_hat and T_s must be positive and finite")
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}")
    if level is not None:
        try:
            level = float(level)
        except (TypeError, ValueError):
            raise ValueError("level must be a number in (0, 1)") from None
        if not (0.0 < level < 1.0):
            raise ValueError(f"level must lie in (0, 1); got {level}")
    fixed = dict(fixed or {})
    if model == "multi_exponential":
        names = ["tau1"]
        for k in range(1, int(n_shoulders) + 1):
            names += [f"a{k}", f"tau_b{k}"]
        if starts is None:
            starts = []
            for t1, tb0 in ((15.0, 100.0), (2.0, 20.0), (0.5, 5.0)):
                st = [t1]
                for k in range(int(n_shoulders)):
                    st += [0.3, tb0 * 5.0 ** k]
                starts.append(st)
        starts = [dict(zip(names, st)) if not isinstance(st, dict) else st
                  for st in starts]
    else:
        names = list(MODELS[model][1])
        if starts is None:
            starts = _DEFAULT_STARTS[model]
    for k in fixed:
        if k not in names:
            raise ValueError(f"{k!r} is not a parameter of {model!r}")
    free = [n for n in names if n not in fixed]
    nodes, wts = cfg.bin_nodes()
    s_pair = np.sqrt(2.0) * float(cfg.sigma_irf)

    def params_of(x):
        p = dict(fixed)
        p.update({n: float(np.exp(v)) for n, v in zip(free, x[2:])})
        return p

    def g1(p):
        args = [p[n] for n in names]
        c, lam = MODELS[model][0](*args)
        if np.any(np.real(lam) >= 0) or not np.all(np.isfinite(lam)):
            return None
        acc = np.zeros(nodes.shape, complex)
        for ck, lk in zip(c, lam):
            acc += ck * exp_conv_gauss_complex(nodes, lk, s_pair)
        return np.real(acc) @ wts           # g2_1 - 1, bin-averaged

    if c0_prior is not None:
        try:
            c0_m, c0_sd = float(c0_prior[0]), float(c0_prior[1])
        except (TypeError, IndexError, ValueError):
            raise ValueError("c0_prior must be a (mean, sd) pair") from None
        if not (c0_m > 0 and c0_sd > 0 and np.isfinite(c0_m)
                and np.isfinite(c0_sd)):
            raise ValueError("c0_prior must be (mean, sd), both positive "
                             "and finite")

    # sum(h log h - h): adding it makes the objective half the Poisson
    # deviance (plus the c0 prior term), a number of the order of the
    # bins instead of the total counts, so that the optimizer's relative
    # stopping rule acts on the differences that matter (from 0.15.0, as
    # `profile_likelihood_ci` from 0.12.0; up to 0.14.0 the fit could stop
    # short of its best point, by 0.05 in deviance and 0.002 in g2(0) in
    # the tests, and by an amount that depended on the SciPy version)
    pos = h > 0
    nll_off = float(np.sum(h[pos] * np.log(h[pos])) - h.sum())

    def nll(x):
        rho2, lc0 = x[0], x[1]
        p = params_of(x)
        d = g1(p)
        if d is None:
            return 1e300
        mu = flat * np.exp(lc0) * (1.0 + rho2 * d)
        if np.any(mu <= 0) or not np.all(np.isfinite(mu)):
            return 1e300
        v = float(np.sum(mu - h * np.log(mu))) + nll_off
        if c0_prior is not None:
            v += 0.5 * ((np.exp(lc0) - c0_m) / c0_sd) ** 2
        return v

    c0g = float(np.clip(np.median(h[np.abs(cfg.bin_centers)
                                    >= 0.65 * cfg.tau_max]) / flat,
                        1e-3, 1e3)) if h.sum() > 0 else 1.0
    bounds = [(0.0, 1.0), (np.log(1e-3), np.log(1e3))] + \
        [(np.log(1e-6), np.log(1e6))] * len(free)
    best = None
    for st in starts:
        for r0 in (0.9, 0.5):
            x0 = [r0, np.log(c0g)] + [np.log(max(float(st[n]), 1e-6))
                                      for n in free]
            res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
            if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                best = res
    ok = best is not None and best.fun < 1e299
    if not ok:
        return dict(model=model, ok=False)
    x = best.x
    p = params_of(x)
    mu = flat * np.exp(x[1]) * (1.0 + x[0] * g1(p))
    dev = _poisson_deviance(h, mu)
    n_free = 2 + len(free)
    dof = int(h.size - n_free)
    out = dict(model=model, params=p, rho2=float(x[0]),
               c0=float(np.exp(x[1])), g2_0=float(1.0 - x[0]), mu=mu,
               deviance=dev, dof=dof,
               p_value=float(_chi2.sf(dev, dof)) if dof > 0 else np.nan,
               aic=float(2 * (best.fun - nll_off) + 2 * n_free),
               n_free=n_free,
               ok=True, _fit_args=dict(T_s=T_s, r_hat=r_hat, cfg=cfg,
                                       model=model, fixed=fixed,
                                       c0_prior=c0_prior,
                                       n_shoulders=n_shoulders))
    if level is not None:
        lo, hi, improved = _profile_interval(nll, x, float(best.fun),
                                             bounds[1:], level)
        out.update(g2_0_lo=lo, g2_0_hi=hi, level=level,
                   profile_improved=improved)
    return out


def _profile_interval(nll, x_hat, f_hat, bounds, level):
    """Profile-likelihood interval for g2(0) = 1 - x[0] of `fit_model`
    (new in 0.15.0): nll(x) with x = (rho2, log c0, log params...), the
    best fit x_hat with value f_hat, and the bounds of x[1:].

    From g2_hat the profile is followed outward in steps that start at
    0.01 and grow by half each time, every point minimized over x[1:]
    from the solution at its inner neighbor; the first point beyond the
    chi-square quantile brackets the end, which brentq refines (each
    evaluation again started from the inner neighbor). The ends are
    measured from the smallest NLL seen, so a profile that finds a better
    fit than the local search did moves them. Returns (lo, hi, improved):
    lo is 0 when the interval reaches g2 = 0, hi NaN when it does not
    close below g2 = 1."""
    from scipy.optimize import brentq
    crit = float(_chi2.ppf(level, 1))
    g_hat = float(1.0 - x_hat[0])
    known = {g_hat: (f_hat, np.asarray(x_hat[1:], dtype=float))}

    def prof(g, start):
        g = float(g)
        if g in known:
            return known[g][0]
        res = minimize(lambda y: nll(np.r_[1.0 - g, y]), start,
                       method="L-BFGS-B", bounds=bounds)
        y, v = res.x, float(res.fun)
        if not np.isfinite(v):
            v = 1e300
        known[g] = (v, y)
        return v

    def walk(sign):
        """The profile outward from g_hat on one side: returns the inner
        and outer ends of the bracket of the crossing (None when the
        profile stays below the quantile up to the edge of [0, 1])."""
        g_in, step = g_hat, 0.01
        edge = 0.0 if sign < 0 else 1.0
        while True:
            g_out = g_in + sign * step
            if (g_out - edge) * sign >= 0.0:
                g_out = edge
            v = prof(g_out, known[g_in][1])
            if 2.0 * (v - f_ref[0]) > crit:
                return g_in, g_out
            if g_out == edge:
                return None
            g_in, step = g_out, 1.5 * step

    f_ref = [f_hat]
    brackets = {}
    for sign in (-1.0, 1.0):
        brackets[sign] = walk(sign)
    f_low = min(v for v, _ in known.values())
    improved = f_low < f_hat - 1e-4
    if improved:
        # a better fit turned up: walk again from the profile's minimum
        f_ref[0] = f_low
        g_best = min(known, key=lambda q: known[q][0])
        g_hat = g_best
        for sign in (-1.0, 1.0):
            brackets[sign] = walk(sign)
    ends = {}
    for sign in (-1.0, 1.0):
        b = brackets[sign]
        if b is None:
            ends[sign] = 0.0 if sign < 0 else float("nan")
            continue
        g_in, g_out = b
        start = known[g_in][1]

        def excess(g, start=start):
            return 2.0 * (prof(g, start) - f_ref[0]) - crit

        lo_g, hi_g = sorted((g_in, g_out))
        ends[sign] = float(brentq(excess, lo_g, hi_g, xtol=1e-4))
    return ends[-1.0], ends[1.0], bool(improved)


def goodness_of_fit(fit, n_boot=200, seed=0):
    """Parametric-bootstrap p-value of a `fit_model` result.

    Draws `n_boot` Poisson histograms from the fitted mean, refits each
    (starting from the fitted parameters) and returns
    dict(p_value, deviance, deviances): the share of simulated
    deviances at least as large as the observed one, with the usual
    (1 + count) / (1 + n_boot) so that it is never exactly 0.
    """
    if not fit.get("ok"):
        raise ValueError("the fit failed; nothing to test")
    args = fit["_fit_args"]
    rng = np.random.default_rng(seed)
    start = [dict(fit["params"])]
    free_start = [{k: v for k, v in start[0].items()
                   if k not in args["fixed"]}]
    free_start[0].update({k: v for k, v in args["fixed"].items()})
    devs = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        h = rng.poisson(fit["mu"]).astype(float)
        r = fit_model(h, args["T_s"], args["r_hat"], args["cfg"],
                      args["model"], starts=free_start,
                      fixed=args["fixed"], c0_prior=args["c0_prior"],
                      n_shoulders=args["n_shoulders"])
        devs[i] = r["deviance"] if r.get("ok") else np.inf
    p = (1.0 + np.sum(devs >= fit["deviance"])) / (1.0 + devs.size)
    return dict(p_value=float(p), deviance=fit["deviance"],
                deviances=devs)
