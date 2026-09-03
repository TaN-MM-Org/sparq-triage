"""Prioritized replay: sum-tree bookkeeping and sampling proportionality."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")  # sparq.sac_per imports torch

from sparq.sac_per import PERBuffer, SumTree, UniformBuffer


def test_sumtree_total_tracks_priorities():
    tree = SumTree(8)
    for i, p in enumerate([1.0, 2.0, 3.0, 4.0]):
        tree.add(p, i)
    assert tree.tree[1] == 10.0
    tree.update(0, 5.0)
    assert tree.tree[1] == 14.0
    assert tree.size == 4


def test_sumtree_wraps_at_capacity():
    tree = SumTree(4)
    for i in range(6):
        tree.add(1.0, i)
    assert tree.size == 4
    assert sorted(tree.data) == [2, 3, 4, 5]


def test_sumtree_samples_proportionally_to_priority():
    tree = SumTree(4)
    for i, p in enumerate([1.0, 1.0, 1.0, 7.0]):
        tree.add(p, i)
    rng = np.random.default_rng(0)
    counts = np.zeros(4)
    for _ in range(500):
        idxs, _, _ = tree.sample(4, rng)
        for i in idxs:
            counts[i] += 1
    freq = counts / counts.sum()
    assert abs(freq[3] - 0.7) < 0.05
    assert np.all(np.abs(freq[:3] - 0.1) < 0.05)


def test_per_buffer_roundtrip_and_weights():
    buf = PERBuffer(capacity=64, seed=1)
    for i in range(32):
        buf.push(("s", i))
    assert len(buf) == 32
    idxs, items, w = buf.sample(8)
    assert len(idxs) == len(items) == len(w) == 8
    assert w.max() <= 1.0 + 1e-6 and (w > 0).all()
    buf.update_priorities(idxs, np.linspace(0.1, 2.0, 8))
    idxs2, _, _ = buf.sample(8)
    assert len(idxs2) == 8


def test_uniform_buffer_same_interface():
    buf = UniformBuffer(capacity=64, seed=1)
    for i in range(16):
        buf.push(("s", i))
    idxs, items, w = buf.sample(4)
    assert len(items) == 4 and np.allclose(w, 1.0)
