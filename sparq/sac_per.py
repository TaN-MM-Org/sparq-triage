"""Discrete-action Soft Actor–Critic with Prioritized Experience Replay.

SAC: Haarnoja et al., ICML 2018; discrete-action variant: Christodoulou,
arXiv:1910.07207. PER: Schaul et al., ICLR 2016 (proportional variant,
sum-tree).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)


class SumTree:
    def __init__(self, capacity):
        self.n = 1
        while self.n < capacity:
            self.n *= 2
        self.tree = np.zeros(2 * self.n)
        self.data = [None] * capacity
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

    def add(self, priority, item):
        idx = self.ptr
        self.data[idx] = item
        self.update(idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        i = idx + self.n
        delta = priority - self.tree[i]
        while i >= 1:
            self.tree[i] += delta
            i //= 2

    def sample(self, batch, rng):
        idxs, items = [], []
        total = self.tree[1]
        seg = total / batch
        for b in range(batch):
            v = rng.uniform(seg * b, seg * (b + 1))
            i = 1
            while i < self.n:
                if v <= self.tree[2 * i]:
                    i = 2 * i
                else:
                    v -= self.tree[2 * i]
                    i = 2 * i + 1
            idx = i - self.n
            idx = min(idx, self.size - 1)
            idxs.append(idx)
            items.append(self.data[idx])
        pr = np.array([self.tree[i + self.n] for i in idxs])
        return idxs, items, pr / max(total, 1e-12)


class PERBuffer:
    def __init__(self, capacity=200_000, alpha=0.6, beta0=0.4,
                 beta_steps=150_000, eps=1e-3, seed=0):
        self.tree = SumTree(capacity)
        self.alpha, self.beta0, self.beta_steps = alpha, beta0, beta_steps
        self.eps = eps
        self.max_p = 1.0
        self.step = 0
        self.rng = np.random.default_rng(seed)

    def push(self, transition):
        self.tree.add(self.max_p ** self.alpha, transition)

    def sample(self, batch):
        self.step += 1
        beta = min(1.0, self.beta0 + (1 - self.beta0) *
                   self.step / self.beta_steps)
        idxs, items, probs = self.tree.sample(batch, self.rng)
        w = (self.tree.size * probs) ** (-beta)
        w = w / w.max()
        return idxs, items, w.astype(np.float32)

    def update_priorities(self, idxs, td_errors):
        for i, e in zip(idxs, td_errors):
            p = (abs(float(e)) + self.eps)
            self.max_p = max(self.max_p, p)
            self.tree.update(i, p ** self.alpha)

    def __len__(self):
        return self.tree.size


class UniformBuffer:
    """Ablation: plain replay with the same interface."""

    def __init__(self, capacity=200_000, seed=0, **kw):
        self.data = []
        self.capacity = capacity
        self.ptr = 0
        self.rng = np.random.default_rng(seed)

    def push(self, transition):
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.ptr] = transition
            self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch):
        idxs = self.rng.integers(0, len(self.data), batch)
        return idxs, [self.data[i] for i in idxs], np.ones(batch, np.float32)

    def update_priorities(self, idxs, td):
        pass

    def __len__(self):
        return len(self.data)


class MLP(nn.Module):
    def __init__(self, i, o, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, o))

    def forward(self, x):
        return self.net(x)


class DiscreteSAC:
    def __init__(self, obs_dim, n_actions, gamma=0.98, lr=3e-4, tau=0.005,
                 target_entropy_scale=0.25, per=True, seed=0,
                 buffer_capacity=200_000):
        torch.manual_seed(seed)
        self.n_actions = n_actions
        self.gamma, self.tau = gamma, tau
        self.pi = MLP(obs_dim, n_actions)
        self.q1, self.q2 = MLP(obs_dim, n_actions), MLP(obs_dim, n_actions)
        self.q1t, self.q2t = MLP(obs_dim, n_actions), MLP(obs_dim, n_actions)
        self.q1t.load_state_dict(self.q1.state_dict())
        self.q2t.load_state_dict(self.q2.state_dict())
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.log_alpha = torch.tensor(np.log(0.3), requires_grad=True)
        self.opt_a = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = target_entropy_scale * np.log(n_actions)
        Buf = PERBuffer if per else UniformBuffer
        self.buf = Buf(capacity=buffer_capacity, seed=seed)
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, obs, greedy=False):
        logits = self.pi(torch.from_numpy(obs.astype(np.float32))[None])
        if greedy:
            return int(logits.argmax())
        p = torch.softmax(logits, -1).numpy()[0]
        return int(self.rng.choice(self.n_actions, p=p))

    def update(self, batch=256):
        if len(self.buf) < batch:
            return None
        idxs, items, w = self.buf.sample(batch)
        obs = torch.from_numpy(np.stack([t[0] for t in items]).astype(np.float32))
        act = torch.from_numpy(np.array([t[1] for t in items], np.int64))
        rew = torch.from_numpy(np.array([t[2] for t in items], np.float32))
        nobs = torch.from_numpy(np.stack([t[3] for t in items]).astype(np.float32))
        done = torch.from_numpy(np.array([t[4] for t in items], np.float32))
        w = torch.from_numpy(w)
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            nlogits = self.pi(nobs)
            npi = torch.softmax(nlogits, -1)
            nlogpi = torch.log_softmax(nlogits, -1)
            qmin = torch.min(self.q1t(nobs), self.q2t(nobs))
            v_next = (npi * (qmin - alpha * nlogpi)).sum(-1)
            target = rew + self.gamma * (1 - done) * v_next
        q1a = self.q1(obs).gather(1, act[:, None]).squeeze(1)
        q2a = self.q2(obs).gather(1, act[:, None]).squeeze(1)
        td = (q1a - target).detach()
        loss_q = (w * ((q1a - target) ** 2 + (q2a - target) ** 2)).mean()
        self.opt_q.zero_grad()
        loss_q.backward()
        self.opt_q.step()

        logits = self.pi(obs)
        pi = torch.softmax(logits, -1)
        logpi = torch.log_softmax(logits, -1)
        with torch.no_grad():
            qmin = torch.min(self.q1(obs), self.q2(obs))
        loss_pi = (w * (pi * (self.log_alpha.exp().detach() * logpi - qmin)
                        ).sum(-1)).mean()
        self.opt_pi.zero_grad()
        loss_pi.backward()
        self.opt_pi.step()

        entropy = -(pi * logpi).sum(-1).detach()
        loss_a = (self.log_alpha.exp() *
                  (entropy - self.target_entropy).detach()).mean()
        self.opt_a.zero_grad()
        loss_a.backward()
        self.opt_a.step()

        self.buf.update_priorities(idxs, td.numpy())
        with torch.no_grad():
            for tnet, net in ((self.q1t, self.q1), (self.q2t, self.q2)):
                for pt, p in zip(tnet.parameters(), net.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * p)
        return dict(loss_q=float(loss_q), loss_pi=float(loss_pi),
                    alpha=float(self.log_alpha.exp()),
                    entropy=float(entropy.mean()))
