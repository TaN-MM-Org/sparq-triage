"""Differentiable (PyTorch) twin: expected HBT histogram as a smooth
function of the *measurement protocol* — excitation saturation parameter s
and correlation-window half-width tau_max — enabling adjoint (pathwise-
gradient) optimization of the protocol through the physics.

Power model (standard saturation photophysics):
    detected rate   R(s)   = 2 R_1 s/(1+s)          (R_1 = rate at s = 1)
    antibunching    tau1(s)= 2 tau1_1 /(1+s)        (pump-rate shortening)
    bunching amp    a(s)   = a_1 s                  (shelving pumping)
    background      B(s)   = B_1 s                  (linear in pump)
Counts are reparameterized as mu + sqrt(mu) * eps (Gaussian approximation
of Poisson), so gradients flow through both the mean and the noise scale.
"""
import numpy as np
import torch


def torch_expected_hist(theta_s, theta_w, base, T_s, n_bins=121,
                        sigma_irf=0.35):
    """theta_s: log saturation parameter (scalar tensor);
    theta_w: log tau_max; base: dict of per-site base-parameter tensors
    (tau1, tau2, a, rate_kcps, rho, n_emitters) at s = 1; returns [B, K]."""
    s = torch.exp(theta_s)
    tau_max = torch.exp(theta_w)
    B = base["tau1"].shape[0]
    k = torch.arange(n_bins, dtype=torch.float32)
    # bin centers scale with the window
    centers = ((k + 0.5) / n_bins * 2.0 - 1.0)[None, :] * tau_max
    width = 2.0 * tau_max / n_bins

    tau1 = base["tau1"][:, None] * 2.0 / (1.0 + s)
    a = base["a"][:, None] * 2.0 * s / (1.0 + s)     # shelving saturates
    tau2 = base["tau2"][:, None]
    S_rate = base["rate_kcps"][:, None] * 1e3 * base["rho"][:, None] \
        * 2.0 * s / (1.0 + s)
    B_rate = base["rate_kcps"][:, None] * 1e3 * (1 - base["rho"][:, None]) * s
    rho = S_rate / (S_rate + B_rate + 1e-9)
    r_tot = S_rate + B_rate

    at = centers.abs()
    # IRF-smoothed exponential ~ exp(-|t|/T) with soft rounding near 0
    smear = torch.sqrt(at ** 2 + 2.0 * sigma_irf ** 2)
    dip = (1 + a) * torch.exp(-smear / tau1) - a * torch.exp(-at / tau2)
    g2 = 1.0 - dip / base["n"][:, None]
    g2 = 1.0 + rho ** 2 * (g2 - 1.0)
    flat = (0.5 * r_tot) ** 2 * (width * 1e-9) * T_s
    return torch.clamp(flat * g2, min=1e-8)


def reparam_counts(mu, gen):
    eps = torch.randn(mu.shape, generator=gen)
    return torch.clamp(mu + torch.sqrt(mu) * eps, min=0.0)


def base_tensors(sites):
    return dict(
        tau1=torch.tensor([s.params["tau1"] for s in sites]),
        tau2=torch.tensor([s.params["tau2"] for s in sites]),
        a=torch.tensor([s.params["a"] for s in sites]),
        rate_kcps=torch.tensor([s.params["rate_kcps"] for s in sites]),
        rho=torch.tensor([s.params["rho"] for s in sites]),
        n=torch.tensor([float(s.n_emitters) for s in sites]),
    )


def fisher_info_g2zero(s, base_mean, T_s=1.0, n_bins=121, tau_max=60.5,
                       profile=True):
    """Numerical Fisher information for estimating g2(0) at saturation s.

    With profile=True the nuisance parameters of the estimation problem
    (antibunching time, bunching amplitude, and overall normalization)
    are profiled out: I_eff = 1/[I^{-1}]_{DD}, where D is the
    rho^2-driven dip depth. This is the information actually available
    to an estimator that does not know the photophysics a priori; the
    naive single-parameter FI (profile=False) grows monotonically with
    brightness and is an overestimate.
    """
    th_s = torch.tensor(np.log(s), dtype=torch.float32)
    th_w = torch.tensor(np.log(tau_max), dtype=torch.float32)

    def mu_of(base):
        return torch_expected_hist(th_s, th_w, base, T_s, n_bins)

    base = {k: v.clone() for k, v in base_mean.items()}
    mu0 = mu_of(base)
    eps = 1e-3

    def pert(key, mult=False, val=eps):
        b = {k: v.clone() for k, v in base.items()}
        if key == "rho2":
            b["rho"] = torch.sqrt(torch.clamp(base["rho"] ** 2 + val,
                                              max=1.0))
        elif mult:
            b[key] = base[key] * (1 + val)
        else:
            b[key] = base[key] + val
        return (mu_of(b) - mu0) / val

    derivs = [pert("rho2")]                              # D (target)
    if profile:
        derivs += [pert("tau1", mult=True),              # nuisances
                   pert("a", val=0.05),
                   pert("tau2", mult=True),
                   mu0 / 1.0]                            # normalization
    Dm = torch.stack(derivs)                             # [P, B, K]
    # average per-site Fisher matrices over the population
    W = 1.0 / torch.clamp(mu0, min=1e-9)
    I = torch.einsum("pbk,qbk,bk->pq", Dm, Dm, W) / Dm.shape[1]
    if not profile:
        return float(I[0, 0])
    Iinv = torch.linalg.inv(I + 1e-9 * torch.eye(I.shape[0]))
    return float(1.0 / Iinv[0, 0])
