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
    """The twin's nominal mapping (tau1,tau2,a) -> CTMC rates (see physics)."""
    k_tot = 1.0 / tau1
    k_exc, k_r = 0.4 * k_tot, 0.6 * k_tot
    k_se = 1.0 / tau2
    k_es = a * k_se * (k_exc + k_r) / k_exc
    return k_exc, k_r, k_es, k_se
