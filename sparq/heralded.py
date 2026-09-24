"""Heralded single-photon sources: heralded g2(0) and the
coincidence-to-accidental ratio (CAR).

Solid-state emitters are one route to single photons; the other
workhorse is the heralded pair source (parametric down-conversion or
four-wave mixing), where detecting one photon of a pair heralds its
twin. Its purity metric is the heralded second-order correlation
g2_h(0), and one routinely measured number ties to it: the
coincidence-to-accidental ratio CAR = C / A, the coincidences between
signal and idler detectors in the same time window over those between
different windows (the accidentals).

Low-efficiency relations. When every detection probability is small
(each detector's click probability proportional to the number of
photons it receives) and there are no dark counts, a source whose pair
number has the statistics of K independent thermal modes (K = 1:
single-mode, "thermal"; K -> infinity: Poissonian, the many-mode
limit) gives, with c = CAR,

    c = 1 + 1/K + 1/mu,        g2_h(0) = (1 + 1/K) (2 c - 1) / c^2,

mu the mean pair number per window. The derivation uses only the
factorial moments of the pair number (E[n(n-1)] = (1 + 1/K) mu^2,
E[n(n-1)(n-2)] = (1 + 1/K)(1 + 2/K) mu^3); the tests check both
relations against the exact model below at tiny efficiencies. For
K -> infinity this is (2 c - 1)/c^2, the Poissonian formula of H. Wang
et al., arXiv:2404.03236. For K = 1 it is (4 c - 2)/c^2 with c >= 2.

Correction in 0.10.0: up to 0.9.1 the thermal branch returned
(4 c + 2)/(c + 1)^2. That is the same single-mode result written with
the NET ratio c' = (C - A)/A = c - 1, while the Poissonian branch used
the raw ratio C / A; both branches now use the raw ratio, and
car_definition="net" takes (C - A)/A for either.

Beyond the low-efficiency limit. `heralded_source` computes CAR and
g2_h(0) exactly for threshold (click / no-click) detectors with given
efficiencies and dark-count probabilities per window, and
`heralded_from_car` inverts it. Up to 0.9.1 the closed forms were
described as lower limits that real sources can only exceed; the exact
model shows that this does not hold: at the same measured CAR, an
efficient herald detector gives a LOWER g2_h (a threshold detector
counts a multi-pair window once, so multi-pair windows weigh less than
in the linear limit), and dark counts can give a higher one. The tests
show both. The closed forms are the low-efficiency values, and
`heralded_from_car` with the measured efficiencies and dark counts is
the estimate to quote.

Neither applies to a single-emitter (antibunched) source; that is what
the rest of this package is for.
"""
from __future__ import annotations

import numpy as np

__all__ = ["heralded_g2_limit", "car_for_purity", "heralded_source",
           "heralded_from_car"]


def _modes(statistics, modes):
    if modes is not None:
        K = float(modes)
        if not (K > 0):
            raise ValueError("modes must be positive (np.inf: Poissonian)")
        return K
    if statistics == "poissonian":
        return np.inf
    if statistics == "thermal":
        return 1.0
    raise ValueError('statistics must be "poissonian" or "thermal" '
                     '(or give modes)')


def _raw_car(car, car_definition):
    c = np.asarray(car, dtype=float)
    if car_definition == "raw":
        return c
    if car_definition == "net":
        return c + 1.0
    raise ValueError('car_definition must be "raw" (C/A) or "net" '
                     '((C-A)/A)')


def heralded_g2_limit(car, statistics="poissonian", modes=None,
                      car_definition="raw"):
    """Heralded g2(0) from the CAR in the low-efficiency limit.

    car : the coincidence-to-accidental ratio, C/A (car_definition
        "raw", the default) or (C - A)/A ("net").
    statistics : "poissonian" (many modes, K -> infinity, the usual
        laser-pumped source) or "thermal" (one mode, K = 1); or give
        `modes` = K directly (np.inf for Poissonian).

    Returns (1 + 1/K)(2 c - 1)/c^2 with c the raw CAR. Refuses a raw
    CAR below 1 + 1/K, which this pair statistics cannot produce
    without dark counts (for K = 1 the raw CAR is at least 2). See the
    module docstring for what the value does and does not bound.
    """
    K = _modes(statistics, modes)
    c = _raw_car(car, car_definition)
    c_min = 1.0 + (0.0 if np.isinf(K) else 1.0 / K)
    if np.any(~np.isfinite(c)) or np.any(c < c_min):
        raise ValueError(
            f"the raw CAR must be finite and >= {c_min:.6g} for this pair "
            "statistics (1 + 1/K; below it the source cannot produce "
            "it without dark counts)")
    f = 1.0 + (0.0 if np.isinf(K) else 1.0 / K)
    out = f * (2.0 * c - 1.0) / c ** 2
    return float(out) if np.ndim(out) == 0 else out


def car_for_purity(g2_target, statistics="poissonian", modes=None,
                   car_definition="raw"):
    """The CAR at which the low-efficiency heralded g2(0) equals
    g2_target: the inverse of `heralded_g2_limit`.

    With f = 1 + 1/K, g c^2 - 2 f c + f = 0; the root on the branch
    c >= f is c = (f + sqrt(f^2 - g f)) / g. Refuses targets outside
    (0, (1 + 2/K)/(1 + 1/K)], the values the relation takes for
    c >= 1 + 1/K (1 for Poissonian, 3/2 for thermal). Returns the raw
    CAR, or the net one with car_definition="net".
    """
    K = _modes(statistics, modes)
    g = float(g2_target)
    f = 1.0 + (0.0 if np.isinf(K) else 1.0 / K)
    g_max = (1.0 + (0.0 if np.isinf(K) else 2.0 / K)) / f
    if not (np.isfinite(g) and 0.0 < g <= g_max):
        raise ValueError(f"g2_target must lie in (0, {g_max:.6g}] for "
                         "this pair statistics: zero needs an infinite "
                         "CAR, and the largest value is reached at the "
                         "smallest possible CAR")
    c = (f + np.sqrt(max(f * f - g * f, 0.0))) / g
    c = max(c, f)
    if car_definition == "net":
        return float(c - 1.0)
    if car_definition != "raw":
        raise ValueError('car_definition must be "raw" or "net"')
    return float(c)


def _pair_pmf(mu, K):
    """Pair-number probabilities, truncated where the tail is below
    1e-34 of the largest term."""
    from scipy.stats import nbinom, poisson
    N = int(mu) + 64
    while True:
        n = np.arange(N)
        P = poisson.pmf(n, mu) if np.isinf(K) else \
            nbinom.pmf(n, K, K / (K + mu))
        if P[-1] < 1e-34 * P.max() and N > 2 * mu + 10:
            return n, P
        N *= 2


def heralded_source(mu, modes=np.inf, eta_s=0.1, eta_i=0.1, dark_s=0.0,
                    dark_i=0.0):
    """Exact CAR and heralded g2(0) of a pair source seen by threshold
    detectors (new in 0.10.0).

    mu : mean number of pairs per coincidence window (or per pulse).
    modes : K, the number of independent thermal modes of the pair
        number (1: single-mode; np.inf: Poissonian).
    eta_s, eta_i : detection efficiency of a signal photon (whole
        signal arm) and of an idler photon, including all losses.
    dark_s, dark_i : probability of a dark or background click per
        window of each signal detector and of the idler (herald)
        detector.

    Model: the pair number n has the stated statistics; each idler
    photon is detected independently with eta_i and the herald clicks
    if at least one is detected or a dark count occurs. For the CAR the
    whole signal arm goes to one detector (efficiency eta_s); for g2_h
    it is split 50/50 onto two detectors A and B (each photon reaches A
    with probability 1/2 and is then detected with eta_s), and
        g2_h = P(A B H) P(H) / (P(A H) P(B H)).
    CAR = P(S H) / (P(S) P(H)), the accidentals being coincidences
    between independent windows. Every probability is a sum over n of
    positive terms written with expm1/log1p, so it stays accurate at
    small efficiencies (the tests check the low-efficiency relations
    of the module docstring to 1e-5 and a Monte Carlo simulation of
    the photons within its statistical error).

    Returns dict(car, car_net, g2_h, p_herald, p_signal, p_coinc).
    """
    mu = float(mu)
    K = float(modes)
    for name, v in (("eta_s", eta_s), ("eta_i", eta_i)):
        if not (0.0 < v <= 1.0):
            raise ValueError(f"{name} must lie in (0, 1]")
    for name, v in (("dark_s", dark_s), ("dark_i", dark_i)):
        if not (0.0 <= v < 1.0):
            raise ValueError(f"{name} must lie in [0, 1)")
    if not (mu > 0 and np.isfinite(mu)):
        raise ValueError("mu must be positive")
    if not K > 0:
        raise ValueError("modes must be positive")
    n, P = _pair_pmf(mu, K)
    lDs, lDi = np.log1p(-dark_s), np.log1p(-dark_i)
    fH = -np.expm1(lDi + n * np.log1p(-eta_i)) if eta_i < 1 else \
        np.where(n > 0, 1.0, dark_i)
    fS = -np.expm1(lDs + n * np.log1p(-eta_s)) if eta_s < 1 else \
        np.where(n > 0, 1.0, dark_s)
    lx = np.log1p(-eta_s / 2.0)
    fA = -np.expm1(lDs + n * lx)
    # P(A and B | n) = (1 - D x^n)^2 + D^2 (y^n - x^2n), y = x^2 - e^2/4
    if eta_s < 1.0:
        x = 1.0 - eta_s / 2.0
        delta = -np.log1p(-(eta_s ** 2) / (4.0 * x * x))
        fAB = fA ** 2 + np.exp(2.0 * lDs + 2.0 * n * lx) * np.expm1(
            -n * delta)
    else:                        # y = 0: 1 - 2 D 2^-n + D^2 [n == 0]
        fAB = (1.0 - 2.0 * np.exp(lDs + n * lx)
               + np.exp(2.0 * lDs) * (n == 0))
    PH, PS = float(P @ fH), float(P @ fS)
    PSH, PAH, PABH = float(P @ (fS * fH)), float(P @ (fA * fH)), \
        float(P @ (fAB * fH))
    car = PSH / (PS * PH)
    return dict(car=car, car_net=car - 1.0, g2_h=PABH * PH / PAH ** 2,
                p_herald=PH, p_signal=PS, p_coinc=PSH)


def heralded_from_car(car, modes=np.inf, eta_s=0.1, eta_i=0.1,
                      dark_s=0.0, dark_i=0.0, car_definition="raw"):
    """Heralded g2(0) from a measured CAR with the measured efficiencies
    and dark counts (new in 0.10.0): the pair numbers mu at which
    `heralded_source` gives this CAR, and g2_h there.

    Without dark counts the CAR falls steadily with mu and there is one
    solution. With dark counts the CAR also falls at small mu (the
    accidentals are then mostly dark counts), so there can be two; the
    measured herald probability per window (`p_herald` of each
    solution) tells them apart. Solutions are found by scanning mu on a
    log grid from 1e-8 to 1e3 and refining each sign change with
    brentq. Returns a list of dicts (mu plus the `heralded_source`
    output), sorted by mu; empty when no mu gives this CAR.
    """
    from scipy.optimize import brentq
    c = float(_raw_car(car, car_definition))
    if not np.isfinite(c):
        raise ValueError("CAR must be finite")

    def f(lm):
        return heralded_source(np.exp(lm), modes, eta_s, eta_i, dark_s,
                               dark_i)["car"] - c

    grid = np.linspace(np.log(1e-8), np.log(1e3), 400)
    vals = np.array([f(g) for g in grid])
    out = []
    for i in range(grid.size - 1):
        if vals[i] == 0.0:
            roots = [grid[i]]
        elif vals[i] * vals[i + 1] < 0:
            roots = [brentq(f, grid[i], grid[i + 1], xtol=1e-14)]
        else:
            continue
        for r in roots:
            mu = float(np.exp(r))
            d = heralded_source(mu, modes, eta_s, eta_i, dark_s, dark_i)
            d["mu"] = mu
            out.append(d)
    return out
