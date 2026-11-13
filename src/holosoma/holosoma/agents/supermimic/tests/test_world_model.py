from __future__ import annotations

import pytest

from holosoma.agents.supermimic.world_model import Dynamics, SequenceReplay, lambda_returns
from holosoma.utils.safe_torch_import import nn, torch


def test_replay_window_preserves_world_and_order_after_wrap() -> None:
    replay = SequenceReplay(4)
    frame = torch.zeros(2, 1)
    for step in range(7):
        frame[:, 0] = torch.tensor([step, step + 100])
        replay.append({"state": frame})
    frame.fill_(-1)

    windows = replay.sample(batch_size=16, length=4, fields=("state",))["state"].squeeze(-1)
    torch.testing.assert_close(windows.diff(dim=1), torch.ones(16, 3))
    torch.testing.assert_close(windows[:, 0].remainder(100), torch.full((16,), 3.0))


def test_dynamics_fits_terminal_successor_and_masks_reset() -> None:
    model = Dynamics(9, 1, 1, hidden_dim=8)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    state = torch.zeros(2, 3, 9)
    state[..., 8] = -1  # Physical gravity direction is part of the state layout.
    successor = state.clone()
    successor[..., 0] = torch.tensor([[1.0, 1e3, -1e3], [1.0, 2.0, 3.0]])
    batch = {
        "state": state,
        "next_state": successor,
        "applied_action": torch.zeros(2, 3, 1),
        "context": torch.zeros(2, 3, 1),
        "done": torch.tensor([[True, False, False], [False, False, True]]),
        "terminated": torch.tensor([[True, False, False], [False, False, False]]),
    }
    state_loss, fall_loss = model.rollout_loss(batch)
    assert state_loss.item() == pytest.approx((1 + 1 + 4 + 9) / (4 * 9))
    fall_loss.backward()
    # One fall among four valid transitions. The other episode ends by timeout.
    assert model.network[-1].bias.grad[-1].item() == pytest.approx(0.25)


def test_frozen_dynamics_propagates_gradient_through_multiple_steps() -> None:
    model = Dynamics(9, 1, 1, hidden_dim=8)
    transition = nn.Linear(11, 10, bias=False)
    model.network = nn.Sequential(transition)
    with torch.no_grad():
        transition.weight.zero_()
        transition.weight[0, 0] = 1.0
        transition.weight[0, 9] = 1.0
    model.requires_grad_(False)
    zeros = torch.zeros(1, 1)
    action = torch.ones(1, 1, requires_grad=True)
    state = torch.zeros(1, 9)
    state[:, 8] = -1
    first, _ = model(state, action, zeros)
    second, _ = model(first, zeros, zeros)
    second.sum().backward()
    torch.testing.assert_close(action.grad, torch.full_like(action, 2))
    assert all(param.grad is None for param in model.parameters())


def test_lambda_returns_use_soft_termination_and_bootstrap() -> None:
    rewards = torch.tensor([[1.0], [2.0]])
    discounts = torch.tensor([[0.5], [0.25]])
    next_values = torch.tensor([[6.0], [8.0]])
    targets = lambda_returns(rewards, discounts, next_values, gae_lambda=0.75)
    torch.testing.assert_close(targets, torch.tensor([[3.25], [4.0]]))


def test_replay_samples_only_selected_worlds() -> None:
    replay = SequenceReplay(2)
    replay.append({"state": torch.arange(4)[:, None]})
    selected = torch.tensor([1, 3])
    sampled = replay.sample(128, 1, ("state",), selected)["state"]
    assert sampled.remainder(2).eq(1).all()


def test_joint_prior_integrates_predicted_velocities() -> None:
    model = Dynamics(11, 1, 1, hidden_dim=8, joint_position_start=9, joint_dt_scaled=0.5)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.network[-1].bias[10] = 2.0
    state = torch.zeros(1, 11)
    state[:, 8:] = torch.tensor([-1., 1., 4.])
    action = context = torch.zeros(1, 1)
    successor, _ = model(state, action, context)
    expected = state.clone()
    expected[:, 9:] = torch.tensor([3.5, 6.])
    torch.testing.assert_close(successor, expected)
