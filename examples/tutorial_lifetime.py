"""Tutorial: the excited-state lifetime of an emitter from a TCSPC decay
histogram, with 95 % intervals.

With a pulsed laser, a time-correlated single-photon counting (TCSPC)
card records each detected photon's delay after the laser pulse (in T3
mode `read_ptu` returns it as `dtime`). This script makes such delays
photon by photon (illustrative numbers, a fixed random seed), histograms
them, and fits the decay with `sparq.fit_lifetime`. It shows why the
laser period must be given when the decay lasts into the next period,
and how a second lifetime is found and checked.

Run it with `python examples/tutorial_lifetime.py`. It printed
(sparq 0.14.0, Python 3.13, NumPy 2.5.3, SciPy 1.18.1):

lifetime 5.938 ns, 95 % interval [5.606, 6.225] (true 6); IRF FWHM 276 ps (true 283); p-value 0.04
without the laser period: lifetime 4.074 ns, p-value 2.8e-09
one lifetime: 1.159 ns, p-value 1.5e-49
two lifetimes: 0.924 ns, 95 % interval [0.887, 0.953] (true 0.9), share of photons 0.704
two lifetimes: 5.821 ns, 95 % interval [4.213, 6.728] (true 5.0), share of photons 0.296
two lifetimes: p-value 0.02
"""
import numpy as np

from sparq import fit_lifetime

rng = np.random.default_rng(14)
t_rep = 12.5                       # ns, an 80 MHz laser
edges = np.linspace(0.0, t_rep, 501)              # 25 ps bins
centers = 0.5 * (edges[1:] + edges[:-1])


def delays(n, lifetimes, shares, t0=2.0, sigma=0.12, n_dark=2000):
    """n photon delays after the laser pulse: an exponential decay (a
    lifetime drawn per photon with the given shares), Gaussian timing
    jitter, folded into one laser period (a photon that arrives after
    the next pulse is timed from that pulse), plus dark counts spread
    evenly over the period."""
    tau = rng.choice(lifetimes, size=n, p=shares)
    t = t0 + rng.exponential(tau) + rng.normal(0.0, sigma, n)
    dark = rng.uniform(0.0, t_rep, n_dark)
    return np.concatenate([t % t_rep, dark])


# 1. An emitter with a 6 ns lifetime: 60 000 photons. Its decay has not
#    died out by the next pulse (e^(-12.5/6) = 12 % is left).
h = np.histogram(delays(60_000, [6.0], [1.0]), bins=edges)[0]
fit = fit_lifetime(centers, h, t_rep=t_rep)
print(f"lifetime {fit['tau'][0]:.3f} ns, 95 % interval "
      f"[{fit['tau_lo'][0]:.3f}, {fit['tau_hi'][0]:.3f}] (true 6); "
      f"IRF FWHM {2.3548 * fit['sigma_irf'] * 1e3:.0f} ps (true 283); "
      f"p-value {fit['p_value']:.2f}")

# 2. The same histogram fitted as if each decay ended within its period:
#    the tails of earlier pulses are taken for background, the lifetime
#    comes out short, and the p-value rejects the fit.
naive = fit_lifetime(centers, h, interval=False)
print(f"without the laser period: lifetime {naive['tau'][0]:.3f} ns, "
      f"p-value {naive['p_value']:.1e}")

# 3. An emitter with two decays: 0.9 ns (70 % of the photons) and 5 ns.
h2 = np.histogram(delays(60_000, [0.9, 5.0], [0.7, 0.3]), bins=edges)[0]
one = fit_lifetime(centers, h2, t_rep=t_rep, interval=False)
two = fit_lifetime(centers, h2, t_rep=t_rep, n_components=2)
print(f"one lifetime: {one['tau'][0]:.3f} ns, p-value {one['p_value']:.1e}")
for j, true in enumerate((0.9, 5.0)):
    print(f"two lifetimes: {two['tau'][j]:.3f} ns, 95 % interval "
          f"[{two['tau_lo'][j]:.3f}, {two['tau_hi'][j]:.3f}] (true "
          f"{true}), share of photons {two['fraction'][j]:.3f}")
print(f"two lifetimes: p-value {two['p_value']:.2f}")
