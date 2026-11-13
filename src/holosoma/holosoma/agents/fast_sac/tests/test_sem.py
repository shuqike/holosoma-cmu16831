"""CPU checks for SEM geometry, gradients, baseline preservation, and preset isolation."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from holosoma.agents.fast_sac.fast_sac import Actor, SimplicialEmbedding
from holosoma.config_values.loco.g1.experiment import g1_29dof_fast_sac, g1_29dof_fast_sac_sem


def make_actor(**kwargs) -> Actor:
    return Actor(
        obs_indices={"actor_obs": {"start": 0, "end": 8, "size": 8}},
        obs_keys=["actor_obs"], n_act=3, num_envs=4, hidden_dim=512,
        log_std_min=-5.0, log_std_max=0.0, device="cpu", **kwargs,
    )


def test_simplex_geometry_and_gradient() -> None:
    features = torch.randn(4, 128, requires_grad=True)
    embedded = SimplicialEmbedding(64)(features)
    assert embedded.shape == features.shape
    assert torch.all(embedded > 0)
    torch.testing.assert_close(embedded.reshape(4, 2, 64).sum(-1), torch.ones(4, 2))
    (embedded * torch.randn_like(embedded)).sum().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0


def test_disabled_sem_preserves_default_actor() -> None:
    default = make_actor()
    disabled = make_actor(use_actor_sem=False)
    disabled.load_state_dict(default.state_dict())
    obs = torch.randn(4, 8)
    for actual, expected in zip(disabled(obs), default(obs)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert isinstance(default.net[-1], torch.nn.SiLU)


def test_sem_changes_features_without_adding_parameters() -> None:
    baseline = make_actor()
    sem = make_actor(use_actor_sem=True)
    sem.load_state_dict(baseline.state_dict())
    assert sum(p.numel() for p in sem.parameters()) == sum(p.numel() for p in baseline.parameters())
    obs = torch.randn(4, 8)
    assert not torch.allclose(sem.net(obs), baseline.net(obs))
    actions, log_probs = sem.get_actions_and_log_probs(obs)
    assert actions.shape == (4, 3)
    assert log_probs.shape == (4,)
    assert torch.isfinite(log_probs).all()
    (-log_probs.mean()).backward()
    assert sem.fc_logstd.weight.grad is not None
    assert torch.isfinite(sem.fc_logstd.weight.grad).all()


@pytest.mark.parametrize("group_size", [0, 1, 7, 65])
def test_invalid_actor_group_size(group_size: int) -> None:
    with pytest.raises(ValueError, match="divisible"):
        make_actor(use_actor_sem=True, sem_group_size=group_size)


def test_preset_changes_only_sem_and_run_name() -> None:
    baseline = g1_29dof_fast_sac
    sem = g1_29dof_fast_sac_sem
    assert sem.algo.config.use_actor_sem
    assert not baseline.algo.config.use_actor_sem
    restored = replace(
        sem,
        training=baseline.training,
        algo=replace(sem.algo, config=replace(sem.algo.config, use_actor_sem=False)),
    )
    assert restored == baseline


def test_teammate_task_defaults() -> None:
    preset = g1_29dof_fast_sac_sem
    assert preset.simulator.config.name == "isaacgym"
    assert preset.training.num_envs == 4096
    assert preset.terrain.terrain_term.scale_factor == 1.0
    assert not preset.algo.config.use_cnn_encoder
    assert preset.randomization.setup_terms["push_randomizer_state"].params["enabled"]
    assert preset.randomization.setup_terms["push_randomizer_state"].params["push_interval_s"] == [5, 10]


def test_legacy_scene_cli_preserves_empty_defaults() -> None:
    import tyro

    from holosoma.config_types.simulator import SceneConfig
    from holosoma.utils.tyro_utils import TYRO_CONIFG

    scene = tyro.cli(SceneConfig, args=[], config=TYRO_CONIFG)
    assert scene.scene_files is None
    assert scene.rigid_objects is None


def test_sem_onnx_actions_match_pytorch(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    actor = make_actor(use_actor_sem=True).eval()
    # The zero-initialized heads would conceal an export that accidentally skipped SEM.
    with torch.no_grad():
        actor.fc_mu[0].weight.normal_()
    obs = torch.randn(4, 8)
    model_path = tmp_path / "sem.onnx"
    torch.onnx.export(
        actor, (obs,), str(model_path), input_names=["obs"],
        output_names=["actions", "mean", "log_std"], opset_version=17,
    )
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    actual = torch.from_numpy(session.run(["actions"], {"obs": obs.numpy()})[0])
    torch.testing.assert_close(actual, actor(obs)[0], atol=1e-5, rtol=1e-4)
