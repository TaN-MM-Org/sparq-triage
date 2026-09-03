"""Purity estimators: Levenberg–Marquardt fit baseline, CNN on integrated
histograms (Kudyshev-style), and an event-driven spiking neural network
(surrogate-gradient LIF) that consumes sliced coincidence streams and
supports anytime (early-decision) readout.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import curve_fit

from .physics import HBTConfig

torch.set_num_threads(2)


# ----------------------------------------------------------------------
# 1. Curve-fit baseline (the conventional workflow)
# ----------------------------------------------------------------------

from .analysis import fit_g2_histogram  # noqa: F401  (moved; torch-free home)


# ----------------------------------------------------------------------
# 2. CNN on the integrated histogram
# ----------------------------------------------------------------------

class HistCNN(nn.Module):
    """1-D CNN on log(1+counts) + auxiliary scalars; classification +
    g2(0) regression heads."""

    def __init__(self, n_bins=121, n_aux=5, cond_dim=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 8 + n_aux + cond_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.cls = nn.Linear(64, 2)
        self.reg = nn.Linear(64, 1)
        self.n_aux = n_aux

    def forward(self, hist, aux, cond=None):
        x = torch.log1p(hist).unsqueeze(1)
        z = self.conv(x).flatten(1)
        z = torch.cat([z, aux] + ([cond] if cond is not None else []), 1)
        z = self.head(z)
        return self.cls(z), F.softplus(self.reg(z)).squeeze(-1)

    def macs_per_inference(self, n_bins=121):
        """Multiply-accumulates of one forward pass (dense arithmetic)."""
        m = 0
        L = n_bins
        m += L * 16 * 7
        L //= 2
        m += L * 16 * 32 * 5
        L //= 2
        m += L * 32 * 32 * 3
        m += (32 * 8 + self.n_aux) * 128 + 128 * 64 + 64 * 3
        return m


class TriageCNN(HistCNN):
    """HistCNN with an additional *triage* head predicting the full
    'good emitter' label (pure AND bright AND non-blinking) — blinking is
    visible to it through the bunching pedestal of the histogram. The
    default forward returns the triage head so the closed-loop
    environment and baselines consume P(good) directly."""

    def __init__(self, n_bins=121, n_aux=5, cond_dim=0):
        super().__init__(n_bins, n_aux, cond_dim)
        self.cls_good = nn.Linear(64, 2)

    def _feat(self, hist, aux, cond=None):
        x = torch.log1p(hist).unsqueeze(1)
        z = self.conv(x).flatten(1)
        z = torch.cat([z, aux] + ([cond] if cond is not None else []), 1)
        return self.head(z)

    def forward(self, hist, aux, cond=None):
        z = self._feat(hist, aux, cond)
        return self.cls_good(z), F.softplus(self.reg(z)).squeeze(-1)

    def all_heads(self, hist, aux, cond=None):
        z = self._feat(hist, aux, cond)
        return (self.cls(z), self.cls_good(z),
                F.softplus(self.reg(z)).squeeze(-1))


def train_triage(steps=1500, batch=256, seed=9, lr=1e-3, gen_fn=None):
    """Train the triage estimator (purity + good + regression heads)."""
    from .datasets import make_batch, CFG
    torch.manual_seed(seed)
    net = TriageCNN(CFG.n_bins)
    opt = torch.optim.Adam(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)
    for step in range(steps):
        b = gen_fn(rng, batch) if gen_fn else make_batch(
            rng, batch, T_dist=("logu", 0.03, 30.0), platform="NV")
        hist = torch.from_numpy(b["hist"])
        aux = torch.from_numpy(b["aux"])
        y = torch.from_numpy(b["y_cls"])
        yg = torch.from_numpy(b["y_g2"])
        ygood = torch.from_numpy(b["y_good"])
        v = torch.from_numpy(b["y_valid"])
        cls, cls_good, reg = net.all_heads(hist, aux)
        loss = (F.cross_entropy(cls[v], y[v])
                + F.cross_entropy(cls_good, ygood)
                + 0.5 * F.mse_loss(reg, yg))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 300 == 0:
            print(f"  triage step {step}: loss {float(loss):.3f}",
                  flush=True)
    net.eval()
    return net


# ----------------------------------------------------------------------
# 3. Event-driven spiking network (surrogate-gradient LIF)
# ----------------------------------------------------------------------

class _SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v > 0).float()

    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors
        # fast-sigmoid surrogate (Neftci et al. 2019)
        return g / (1.0 + 10.0 * v.abs()) ** 2


spike_fn = _SpikeFn.apply


class SpikingG2Net(nn.Module):
    """Two hidden LIF layers + leaky readout integrator.

    Input: [B, S, K] coincidence counts per (time-slice, delay-bin), plus
    static aux scalars injected as a constant current. Anytime output:
    readout state after every slice.
    """

    def __init__(self, n_bins=121, n_aux=5, h1=196, h2=128, cond_dim=0,
                 beta=0.90, beta_out=0.95):
        super().__init__()
        self.fc1 = nn.Linear(n_bins + n_aux + cond_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fco = nn.Linear(h2, 3)      # 2 class logits + 1 regression
        self.beta, self.beta_out = beta, beta_out
        self.h1, self.h2 = h1, h2

    def forward(self, stream, aux, cond=None, count_spikes=False):
        B, S, K = stream.shape
        dev = stream.device
        m1 = torch.zeros(B, self.h1, device=dev)
        m2 = torch.zeros(B, self.h2, device=dev)
        out = torch.zeros(B, 3, device=dev)
        outs, n_spk = [], 0.0
        extra = [aux] + ([cond] if cond is not None else [])
        static = torch.cat(extra, 1)
        for s in range(S):
            x = torch.cat([stream[:, s], static], 1)
            m1 = self.beta * m1 + self.fc1(x)
            s1 = spike_fn(m1 - 1.0)
            m1 = m1 - s1
            m2 = self.beta * m2 + self.fc2(s1)
            s2 = spike_fn(m2 - 1.0)
            m2 = m2 - s2
            out = self.beta_out * out + self.fco(s2)
            outs.append(out)
            if count_spikes:
                n_spk = n_spk + s1.sum() + s2.sum()
        traj = torch.stack(outs, 1)                      # [B, S, 3]
        logits, reg = traj[..., :2], F.softplus(traj[..., 2])
        if count_spikes:
            return logits, reg, n_spk
        return logits, reg

    def synops_per_inference(self, stream, aux, cond=None):
        """Measured synaptic operations (event-driven cost model):
        input-event synops + hidden-spike synops + readout synops. The
        static auxiliary scalars are per-inference bias currents on
        neuromorphic hardware (one configuration write per neuron per
        inference), counted once, not as per-slice synaptic events."""
        with torch.no_grad():
            B, S, K = stream.shape
            dev = stream.device
            m1 = torch.zeros(B, self.h1, device=dev)
            m2 = torch.zeros(B, self.h2, device=dev)
            extra = [aux] + ([cond] if cond is not None else [])
            static = torch.cat(extra, 1)
            syn = torch.full((B,), float(self.h1 + self.h2 + 3), device=dev)
            for s in range(S):
                x = torch.cat([stream[:, s], static], 1)
                m1 = self.beta * m1 + self.fc1(x)
                s1 = spike_fn(m1 - 1.0)
                m1 = m1 - s1
                m2 = self.beta * m2 + self.fc2(s1)
                s2 = spike_fn(m2 - 1.0)
                m2 = m2 - s2
                # events: input coincidences fan out to h1; hidden spikes
                # to h2 and the readout
                syn += stream[:, s].sum(1) * self.h1
                syn += s1.sum(1) * self.h2
                syn += s2.sum(1) * 3
        return syn


# ----------------------------------------------------------------------
# Training loop (physics-in-the-loop)
# ----------------------------------------------------------------------

def train_model(model, gen_fn, steps, lr=1e-3, batch=256, device="cpu",
                is_snn=False, log_every=200, seed=0, anytime_w=0.3):
    """gen_fn(rng, batch) -> batch dict; trains through the stochastic twin."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)
    model.train()
    hist_loss = []
    for step in range(steps):
        b = gen_fn(rng, batch)
        aux = torch.from_numpy(b["aux"])
        y = torch.from_numpy(b["y_cls"])
        yg = torch.from_numpy(b["y_g2"])
        v = torch.from_numpy(b["y_valid"]) if "y_valid" in b else \
            torch.ones(len(y), dtype=torch.bool)
        cond = b.get("cond")
        cond = torch.from_numpy(cond) if cond is not None else None
        if v.sum() == 0:
            continue
        if is_snn:
            stream = torch.from_numpy(b["stream"])
            logits, reg = model(stream, aux, cond)
            # deep supervision: weight late slices more
            S = logits.shape[1]
            w = torch.linspace(0.2, 1.0, S)
            w = w / w.sum()
            ce = sum(w[s] * F.cross_entropy(logits[v][:, s], y[v])
                     for s in range(S))
            loss = ce + 0.5 * F.mse_loss(reg[:, -1], yg)
            if anytime_w > 0:
                loss = loss + anytime_w * F.mse_loss(
                    reg[:, S // 2], yg)
        else:
            hist = torch.from_numpy(b["hist"])
            logits, reg = model(hist, aux, cond)
            loss = F.cross_entropy(logits[v], y[v]) \
                + 0.5 * F.mse_loss(reg, yg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist_loss.append(float(loss))
        if log_every and step % log_every == 0:
            print(f"  step {step:5d}  loss {np.mean(hist_loss[-100:]):.4f}")
    model.eval()
    return hist_loss


@torch.no_grad()
def evaluate(model, batch_dict, is_snn=False):
    aux = torch.from_numpy(batch_dict["aux"])
    y = batch_dict["y_cls"]
    yg = batch_dict["y_g2"]
    cond = batch_dict.get("cond")
    cond = torch.from_numpy(cond) if cond is not None else None
    if is_snn:
        stream = torch.from_numpy(batch_dict["stream"])
        logits, reg = model(stream, aux, cond)
        logits, reg = logits[:, -1], reg[:, -1]
    else:
        hist = torch.from_numpy(batch_dict["hist"])
        logits, reg = model(hist, aux, cond)
    pred = logits.argmax(1).numpy()
    prob = torch.softmax(logits, 1)[:, 1].numpy()
    v = batch_dict.get("y_valid", np.ones(len(y), bool))
    acc = balanced_accuracy(y[v], pred[v])
    mae = float(np.mean(np.abs(reg.numpy() - yg)))
    return dict(bal_acc=acc, mae_g2=mae, pred=pred, prob=prob,
                g2_hat=reg.numpy())


def balanced_accuracy(y, pred):
    accs = []
    for c in (0, 1):
        m = y == c
        if m.sum():
            accs.append(float((pred[m] == c).mean()))
    return float(np.mean(accs))
