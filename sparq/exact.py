"""Numerically exact three-level CW g2 from the master equation.

The emitter Liouvillian for populations p = (p_g, p_e, p_s) is

    dp/dt = M p,   M = [[-k_exc,  k_r,        k_se],
                        [ k_exc, -(k_r+k_es), 0   ],
                        [ 0,      k_es,      -k_se]].

Under CW excitation, g2(tau) = p_e(tau | p(0) = g) / p_e(ss): after a
detection the emitter is projected to |g>, and the conditional re-excitation
probability normalized by the steady state is the intensity correlation.
Because M is 3x3 with one zero eigenvalue, g2 is *exactly* a sum of two
exponentials — the analytic form used by the histogram twin — with
(tau1, tau2, a) given by the eigen-decomposition below.
"""
import numpy as np


def liouvillian(k_exc, k_r, k_es, k_se):
    return np.array([
        [-k_exc,  k_r,          k_se],
        [ k_exc, -(k_r + k_es), 0.0 ],
        [ 0.0,    k_es,        -k_se],
    ])


def steady_state(M):
    w, V = np.linalg.eig(M)
    i = np.argmin(np.abs(w))
    p = np.real(V[:, i])
    return p / p.sum()


def g2_exact(tau, k_exc, k_r, k_es, k_se):
    """Exact g2(tau) by eigen-decomposition (tau in ns, rates in 1/ns)."""
    M = liouvillian(k_exc, k_r, k_es, k_se)
    w, V = np.linalg.eig(M)
    Vi = np.linalg.inv(V)
    p0 = np.array([1.0, 0.0, 0.0])            # projected to ground state
    pss = steady_state(M)
    tau = np.atleast_1d(np.abs(tau)).astype(float)
    # p(t) = V diag(e^{w t}) V^{-1} p0 ; take the e-component
    c = Vi @ p0
    pe = np.real(sum(V[1, k] * c[k] * np.exp(np.outer(tau, w[k]))[:, 0]
                     for k in range(3)))
    return pe / pss[1]


def effective_params(k_exc, k_r, k_es, k_se):
    """Exact (tau1, tau2, a) of the two-exponential form from the rates."""
    M = liouvillian(k_exc, k_r, k_es, k_se)
    w, V = np.linalg.eig(M)
    Vi = np.linalg.inv(V)
    pss = steady_state(M)
    c = Vi @ np.array([1.0, 0.0, 0.0])
    # nonzero eigenvalues, sorted by magnitude (fast = antibunching)
    idx = np.argsort(np.abs(w))[1:]
    amps = {i: np.real(V[1, i] * c[i]) / pss[1] for i in idx}
    i_fast = max(idx, key=lambda i: abs(np.real(w[i])))
    i_slow = min(idx, key=lambda i: abs(np.real(w[i])))
    tau1 = -1.0 / np.real(w[i_fast])
    tau2 = -1.0 / np.real(w[i_slow])
    a = amps[i_slow]
    return float(tau1), float(tau2), float(a)


def rates_from_site(tau1, tau2, a):
    """The approximate mapping (tau1, tau2, a) -> rates used by the
    photon-by-photon simulator up to 0.9.1.

    It does NOT reproduce (tau1, tau2, a): `effective_params` of its
    rates differ (for tau1 = 15, tau2 = 250, a = 0.3 ns they are
    14.6 ns, 191 ns and 0.336). Kept for reproducing old results; use
    `rates_for_params` for rates whose g2 is exactly the requested one.
    """
    k_tot = 1.0 / tau1
    k_exc, k_r = 0.4 * k_tot, 0.6 * k_tot
    k_se = 1.0 / tau2
    k_es = a * k_se * (k_exc + k_r) / k_exc
    return k_exc, k_r, k_es, k_se


def rates_for_params(tau1, tau2, a, pump_fraction=0.4):
    """Rates (k_exc, k_r, k_es, k_se) whose exact g2 is
    1 - (1 + a) exp(-|tau|/tau1) + a exp(-|tau|/tau2)  (new in 0.10.0).

    The ratio k_exc / (k_exc + k_r) is fixed at `pump_fraction`; the
    other three rates then follow in closed form. With l1 = 1/tau1,
    l2 = 1/tau2, S = l1 + l2, P = l1 l2 and D = (1 + a) l1 - a l2 (the
    slope of g2 at zero delay), matching the eigenvalues of the rate
    matrix and that slope gives
        k_se = P / D,
        k_exc + k_r + k_es = S - k_se =: Q,
        pump_fraction (k_exc + k_r) k_es = k_se (D - S + k_se) =: R,
    a quadratic for k_es. Of its two roots (both give the same g2)
    the smaller shelving rate k_es is returned. The tests check the
    round trip through `effective_params` to 1e-9.

    Raises ValueError when tau1 >= tau2 (with a > 0), a < 0, or when no
    rate set with this pump fraction exists; the message then names
    the smallest pump fraction that works, if any below 1 does.
    """
    tau1 = float(tau1)
    tau2 = float(tau2)
    a = float(a)
    f = float(pump_fraction)
    if not (tau1 > 0 and tau2 > 0 and np.isfinite(tau1) and np.isfinite(tau2)):
        raise ValueError("tau1 and tau2 must be positive and finite")
    if not (a >= 0 and np.isfinite(a)):
        raise ValueError("a must be finite and >= 0")
    if not (0.0 < f < 1.0):
        raise ValueError("pump_fraction must lie in (0, 1)")
    l1, l2 = 1.0 / tau1, 1.0 / tau2
    if a == 0.0:
        return f * l1, (1.0 - f) * l1, 0.0, l2
    if not tau1 < tau2:
        raise ValueError("with a > 0 the shelving time tau2 must exceed "
                         "the antibunching time tau1")
    S, P = l1 + l2, l1 * l2
    D = (1.0 + a) * l1 - a * l2
    k_se = P / D
    Q = S - k_se
    R = k_se * (D - S + k_se)
    if not (R > 0.0 and Q > 0.0):
        raise ValueError("no three-level rate model has this g2")
    f_min = 4.0 * R / Q ** 2
    if f < f_min:
        hint = (f"; the smallest pump_fraction that works is {f_min:.4g}"
                if f_min < 1.0 else "; no pump fraction below 1 works")
        raise ValueError(f"no rate set with pump_fraction {f} has this "
                         f"g2{hint}")
    k_es = 0.5 * (Q - np.sqrt(max(Q * Q - 4.0 * R / f, 0.0)))
    K = Q - k_es
    return f * K, (1.0 - f) * K, k_es, k_se
