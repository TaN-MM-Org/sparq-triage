"""Graph encoding of emitter level structures for cross-platform transfer.

Each emitter platform is described by the *published* template of its
photophysical level structure: nodes = electronic states (ground, excited,
shelving), edges = transitions annotated with the literature range of
their rates (log-mid and log-span) and a radiative flag.  A small
message-passing network (Gilmer et al., 2017) embeds the template into a
conditioning vector consumed by the estimator (concatenated to its inputs,
i.e., FiLM-style bias conditioning).  Because the embedding is a function
of the physics graph — not a platform ID — the estimator can zero-shot to
platforms never seen in training.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from .physics import PLATFORMS


def template_graph(platform):
    """Platform-level template graph from published parameter ranges.
    Accepts a platform name or a Platform instance (e.g., a procedurally
    generated synthetic platform)."""
    p = PLATFORMS[platform] if isinstance(platform, str) else platform
    t1m = np.sqrt(p.tau1_rng[0] * p.tau1_rng[1])     # geometric mid
    t2m = np.sqrt(p.tau2_rng[0] * p.tau2_rng[1])
    am = 0.5 * (p.a_rng[0] + p.a_rng[1])
    rm = np.sqrt(p.rate_rng[0] * p.rate_rng[1])
    k_tot = 1.0 / t1m
    k_exc, k_r = 0.4 * k_tot, 0.6 * k_tot
    k_se = 1.0 / t2m
    k_es = am * k_se * (k_exc + k_r) / k_exc
    span = lambda rng: np.log10(rng[1] / rng[0])
    # nodes: [is_g, is_e, is_s, radiative]
    node = np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0]], np.float32)
    edges = np.array([[0, 1], [1, 0], [1, 2], [2, 0]], np.int64)
    # edge features: log10 rate mid, spans of tau1/tau2/a, log10 brightness
    ef = np.array([
        [np.log10(k_exc), span(p.tau1_rng), 0.0,        np.log10(rm)],
        [np.log10(k_r),   span(p.tau1_rng), 0.0,        np.log10(rm)],
        [np.log10(k_es),  span(p.tau2_rng), am,         p.blink_p],
        [np.log10(k_se),  span(p.tau2_rng), am,         p.blink_p],
    ], np.float32)
    return node, edges, ef


class GraphEncoder(nn.Module):
    """Two rounds of edge-conditioned message passing + mean pool."""

    def __init__(self, node_dim=4, edge_dim=4, hidden=32, out_dim=8):
        super().__init__()
        self.embed = nn.Linear(node_dim, hidden)
        self.msg1 = nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden),
                                  nn.ReLU(), nn.Linear(hidden, hidden))
        self.msg2 = nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden),
                                  nn.ReLU(), nn.Linear(hidden, hidden))
        self.upd1 = nn.GRUCell(hidden, hidden)
        self.upd2 = nn.GRUCell(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, node, edges, ef):
        h = torch.relu(self.embed(node))                    # [3, H]
        for msg, upd in ((self.msg1, self.upd1), (self.msg2, self.upd2)):
            src, dst = edges[:, 0], edges[:, 1]
            m = msg(torch.cat([h[src], h[dst], ef], -1))    # [E, H]
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)
            h = upd(agg, h)
        return self.out(h.mean(0))                          # [out_dim]


def platform_embeddings(encoder, names):
    out = {}
    for n in names:
        node, edges, ef = template_graph(n)
        z = encoder(torch.from_numpy(node), torch.from_numpy(edges),
                    torch.from_numpy(ef))
        out[n] = z
    return out
