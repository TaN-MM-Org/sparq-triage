"""Shape and gradient contracts of the learned estimators, the
differentiable protocol twin, and the closed-loop triage environment."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sparq.datasets import CFG, make_batch
from sparq.estimators import HistCNN, SpikingG2Net, TriageCNN, evaluate
from sparq.physics import sample_site
from sparq.rl_env import (
    A_CERTIFY,
    A_MEAS0,
    A_REJECT,
    N_ACTIONS,
    OBS_DIM,
    TriageEnv,
    run_adaptive_heuristic,
    run_raster,
)
from sparq.twin_torch import base_tensors, reparam_counts, torch_expected_hist


def test_make_batch_shapes_and_labels():
    rng = np.random.default_rng(3)
    b = make_batch(rng, 8)
    assert b["stream"].shape == (8, 32, CFG.n_bins)
    assert b["hist"].shape == (8, CFG.n_bins)
    assert np.allclose(b["stream"].sum(1), b["hist"])
    assert set(np.unique(b["y_cls"])) <= {0, 1}
    assert np.all((b["y_g2"] >= 0) & (b["y_g2"] <= 3))


def test_cnn_and_snn_forward_shapes():
    rng = np.random.default_rng(3)
    b = make_batch(rng, 8)
    hist = torch.from_numpy(b["hist"])
    aux = torch.from_numpy(b["aux"])
    stream = torch.from_numpy(b["stream"])
    logits, reg = HistCNN()(hist, aux)
    assert logits.shape == (8, 2) and reg.shape == (8,)
    logits, reg = SpikingG2Net()(stream, aux)
    assert logits.shape == (8, 32, 2) and reg.shape == (8, 32)


def test_evaluate_returns_metrics():
    rng = np.random.default_rng(3)
    b = make_batch(rng, 16)
    out = evaluate(HistCNN(), b)
    assert 0.0 <= out["bal_acc"] <= 1.0
    assert out["mae_g2"] >= 0.0
    assert out["pred"].shape == (16,)


def test_protocol_twin_is_differentiable():
    rng = np.random.default_rng(3)
    base = base_tensors([sample_site(rng) for _ in range(4)])
    theta_s = torch.tensor(0.0, requires_grad=True)
    theta_w = torch.tensor(float(np.log(60.5)), requires_grad=True)
    mu = torch_expected_hist(theta_s, theta_w, base, 1.0)
    assert mu.shape == (4, 121)
    assert (mu > 0).all()
    mu.sum().backward()
    assert np.isfinite(theta_s.grad.item())
    assert np.isfinite(theta_w.grad.item())
    gen = torch.Generator().manual_seed(0)
    counts = reparam_counts(mu.detach(), gen)
    assert (counts >= 0).all()


def test_triage_env_step_contract():
    env = TriageEnv(TriageCNN(), n_sites=3, seed=0)
    obs = env.reset()
    assert obs.shape == (OBS_DIM,)
    assert 0 < N_ACTIONS
    obs, r, done, info = env.step(A_MEAS0)
    assert obs.shape == (OBS_DIM,) and not done
    env.step(A_REJECT)
    env.step(A_CERTIFY)
    env.step(A_CERTIFY)
    assert env.done
    s = env.summary()
    for key in ("time_s", "precision", "recall", "good_per_min"):
        assert key in s


def test_baseline_policies_run_to_completion():
    env = TriageEnv(TriageCNN(), n_sites=4, seed=0)
    field = env.new_field()
    s1 = run_raster(env, field, 1.0, np.random.default_rng(5))
    s2 = run_adaptive_heuristic(env, field, np.random.default_rng(5))
    assert s1["time_s"] > 0 and s2["time_s"] > 0
