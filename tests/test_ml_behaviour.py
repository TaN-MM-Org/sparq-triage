"""What the machine-learning layer does, not just its shapes (new in
0.10.0): the estimators learn on the simulator, the agent's update
solves a problem with a known answer, the graph encoder ignores how
the levels are numbered, and the Fisher information matches the
scatter of an estimator."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scipy.optimize import minimize_scalar  # noqa: E402

from sparq import EmitterSite  # noqa: E402
from sparq.datasets import load_fisequr, make_batch, make_eval_set  # noqa
from sparq.estimators import (HistCNN, SpikingG2Net, evaluate,  # noqa
                              train_model)
from sparq.gnn import GraphEncoder, template_graph  # noqa: E402
from sparq.sac_per import DiscreteSAC  # noqa: E402
from sparq.twin_torch import (base_tensors, fisher_info_g2zero,  # noqa
                              torch_expected_hist)


def _gen(rng, b):
    return make_batch(rng, b, T_dist=("logu", 3.0, 30.0))


@pytest.mark.parametrize("snn", [False, True], ids=["cnn", "snn"])
def test_estimators_learn_on_the_simulator(snn):
    """An untrained network is at chance (balanced accuracy 0.5); after
    150 training steps on simulated histograms it classifies held-out
    sites (g2(0) above or below 0.5, boundary cases excluded) well and
    its g2(0) error is below the untrained one."""
    torch.manual_seed(0)
    ev = make_eval_set(np.random.default_rng(99), 400, 10.0)
    net = SpikingG2Net() if snn else HistCNN()
    before = evaluate(net, ev, is_snn=snn)
    losses = train_model(net, _gen, steps=150, batch=64, log_every=0,
                         seed=1, is_snn=snn)
    after = evaluate(net, ev, is_snn=snn)
    assert np.mean(losses[-20:]) < 0.5 * np.mean(losses[:20])
    assert after["bal_acc"] > 0.9 and before["bal_acc"] < 0.7
    assert after["mae_g2"] < 0.7 * before["mae_g2"]


def test_agent_update_solves_a_two_armed_problem():
    """Two states; in state 0 action 1 pays 1, in state 1 action 0 pays
    1, episodes end after one step. The soft actor-critic update must
    learn Q = reward and a policy that picks the paying action."""
    ag = DiscreteSAC(2, 2, seed=0, per=True, buffer_capacity=5000)
    rng = np.random.default_rng(0)
    eye = np.eye(2, dtype=np.float32)
    for _ in range(3000):
        s = int(rng.integers(2))
        a = int(rng.integers(2))
        ag.buf.push((eye[s], a, float(a == 1 - s), eye[s], 1.0))
    for _ in range(400):
        ag.update(128)
    with torch.no_grad():
        for s in range(2):
            q = ag.q1(torch.from_numpy(eye[s:s + 1])).numpy()[0]
            assert abs(q[1 - s] - 1.0) < 0.1 and abs(q[s]) < 0.1
            assert ag.act(eye[s], greedy=True) == 1 - s


def test_graph_encoder_ignores_level_numbering():
    torch.manual_seed(0)
    enc = GraphEncoder()
    node, edges, ef = template_graph("NV")
    z = enc(torch.from_numpy(node), torch.from_numpy(edges),
            torch.from_numpy(ef))
    for perm in ([2, 0, 1], [1, 2, 0], [0, 2, 1]):
        perm = np.array(perm)
        inv = np.argsort(perm)
        z2 = enc(torch.from_numpy(node[perm]), torch.from_numpy(inv[edges]),
                 torch.from_numpy(ef))
        assert (z - z2).abs().max().item() < 1e-6
    z.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in enc.parameters())
    # different platforms give different embeddings
    n2, e2, f2 = template_graph("GaN")
    z3 = enc(torch.from_numpy(n2), torch.from_numpy(e2),
             torch.from_numpy(f2))
    assert (z - z3).abs().max().item() > 1e-3


def test_fisher_information_matches_estimator_scatter():
    """Cramer-Rao check: for the dip parameter with everything else
    known, the scatter of the maximum-likelihood estimate over 300
    simulated histograms is 1 / I (within 20 %)."""
    site = EmitterSite(dict(tau1=15.0, tau2=250.0, a=0.3, rate_kcps=150.0,
                            rho=0.9, blinking=False), 1)
    base = base_tensors([site])
    I = fisher_info_g2zero(1.0, base, T_s=5.0, profile=False)
    th_s = torch.tensor(0.0)
    th_w = torch.tensor(float(np.log(60.5)))

    def mu_of(D):
        b = {k: v.clone() for k, v in base.items()}
        b["rho"] = torch.sqrt(torch.clamp(base["rho"] ** 2 + D, max=1.0))
        return torch_expected_hist(th_s, th_w, b, 5.0).double().numpy()[0]

    mu0 = mu_of(0.0)
    rng = np.random.default_rng(1)
    est = []
    for _ in range(300):
        h = rng.poisson(mu0)
        est.append(minimize_scalar(
            lambda D: float(np.sum(mu_of(D) - h * np.log(mu_of(D)))),
            bounds=(-0.3, 0.09), method="bounded",
            options=dict(xatol=1e-7)).x)
    assert 0.8 < np.var(est) * I < 1.25
    # profiling out the nuisance parameters can only lose information
    assert fisher_info_g2zero(1.0, base, T_s=5.0, profile=True) < I


def test_fisequr_loader_merges_parts(tmp_path):
    """Synthetic files in the layout the loader expects (first column
    delay, then one column per 10 s snapshot; '_part' files of one
    series are merged)."""
    d = np.linspace(-50, 50, 11)
    a = np.c_[d, np.ones((11, 2))]
    b = np.c_[d, 2 * np.ones((11, 3))]
    np.savetxt(os.path.join(tmp_path, "s1_part1.txt"), a)
    np.savetxt(os.path.join(tmp_path, "s1_part2.txt"), b)
    np.savetxt(os.path.join(tmp_path, "s2.txt"), a)
    out = {s["name"]: s for s in load_fisequr(str(tmp_path))}
    assert set(out) == {"s1", "s2"}
    assert out["s1"]["counts"].shape == (11, 5)
    assert out["s1"]["T_total"] == 50.0
    assert np.array_equal(out["s1"]["total"], np.full(11, 8.0))
