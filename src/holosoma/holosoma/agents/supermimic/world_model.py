"""Replay and supervised dynamics fitting for SuperMimic."""

from __future__ import annotations

from holosoma.utils.safe_torch_import import F, nn, torch
from torch.utils.checkpoint import checkpoint


def imagination_step(model, state, action, context, recompute: bool):
    # The deterministic, frozen WM is recomputed during backward instead of
    # retaining all hidden activations for every step of the 32-step rollout.
    # This preserves the complete BPTT objective and batch; there is no detach.
    if recompute:
        return checkpoint(model, state, action, context, use_reentrant=False, preserve_rng_state=False)
    return model(state, action, context)


class SequenceReplay:
    """A time-major ring buffer. Each window comes from a single simulator world."""

    def __init__(self, capacity: int, storage_device: str | None = None):
        self.capacity = capacity
        self.storage_device = storage_device
        self.sample_device = None
        self.steps = 0
        self.data: dict[str, torch.Tensor] = {}
        self._complete_cache = {}

    def __len__(self) -> int:
        return min(self.steps, self.capacity)

    @torch.no_grad()
    def append(self, transition: dict[str, torch.Tensor]) -> None:
        if not self.data:
            self.sample_device = transition["state"].device
            self.data = {name: torch.empty((self.capacity, *value.shape), dtype=value.dtype,
                                          device=self.storage_device or value.device)
                         for name, value in transition.items()}
        slot = self.steps % self.capacity
        for name, value in transition.items():
            self.data[name][slot].copy_(value)
        self.steps += 1
        self._complete_cache.clear()

    def sample(
        self, batch_size: int, length: int, fields: tuple[str, ...], world_ids: torch.Tensor | None = None,
        complete: bool = False,
    ) -> dict[str, torch.Tensor]:
        if not 1 <= length <= len(self):
            raise ValueError(f"Cannot sample a {length}-step window from {len(self)} replay steps.")
        states = self.data["state"]
        if complete:
            # Uniform eligible world, then uniform complete window in that world,
            # as in the public replay buffer. Explicit successor storage allows
            # a terminal last transition; no earlier boundary may be crossed.
            # Cache on the sampling device for all model updates per collection.
            if length not in self._complete_cache:
                chronological = torch.arange(self.steps-len(self), self.steps, device=states.device)
                done = self.data["done"][chronological % self.capacity].to(self.sample_device).long()
                prefix = torch.cat((done.new_zeros(1, done.shape[1]), done.cumsum(0)), 0)
                count = len(self)-length+1
                valid = (prefix[length-1:length-1+count]-prefix[:count]) == 0
                valid = valid.T
                counts = valid.sum(-1)
                positions = torch.arange(count, device=self.sample_device).expand_as(valid)
                packed = torch.where(valid, positions, count).sort(-1).values
                self._complete_cache[length] = (packed, counts)
            packed, counts = self._complete_cache[length]
            available = torch.arange(states.shape[1],device=self.sample_device) if world_ids is None else world_ids.to(self.sample_device)
            available = available[counts[available] > 0]
            if not available.numel():
                raise ValueError("No complete replay window; collect more real transitions")
            worlds = available[torch.randint(available.numel(), (batch_size,),device=self.sample_device)]
            offsets = (torch.rand(batch_size,device=self.sample_device)*counts[worlds]).long()
            starts = packed[worlds, offsets]+self.steps-len(self)
            slots = ((starts[:,None]+torch.arange(length,device=self.sample_device)) % self.capacity).to(states.device)
            return {name:self.data[name][slots,worlds[:,None].to(states.device)].to(self.sample_device) for name in fields}
        # Draw indices on the original training device so CPU offload leaves the
        # CUDA RNG stream and sampled transitions identical to GPU storage.
        starts = torch.randint(self.steps - len(self), self.steps - length + 1, (batch_size, 1), device=self.sample_device)
        world_count = states.shape[1] if world_ids is None else world_ids.numel()
        worlds = torch.randint(world_count, (batch_size, 1), device=self.sample_device)
        if world_ids is not None:
            worlds = world_ids.to(self.sample_device)[worlds]
        slots = ((starts + torch.arange(length, device=self.sample_device)) % self.capacity).to(states.device)
        worlds = worlds.to(states.device)
        return {name: self.data[name][slots, worlds].to(self.sample_device) for name in fields}


def mlp(input_dim: int, hidden_dim: int, output_dim: int, activation=nn.SiLU) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        activation(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        activation(),
        nn.Linear(hidden_dim, output_dim),
    )


class Dynamics(nn.Module):
    """Predict the next scaled state and the probability of falling in one control step."""

    def __init__(
        self, state_dim: int, action_dim: int, context_dim: int, hidden_dim: int,
        joint_position_start: int = 0, joint_dt_scaled: float = 0.0,
        geometry: nn.Module | None = None,
    ):
        super().__init__()
        self.network = nn.Sequential(
            *list(mlp(state_dim + action_dim + context_dim, hidden_dim, hidden_dim).children()),
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, state_dim + 1),
        )
        self.joint_position = slice(joint_position_start, joint_position_start + action_dim)
        self.joint_velocity = slice(joint_position_start + action_dim, joint_position_start + 2 * action_dim)
        self.joint_dt_scaled = joint_dt_scaled
        self.geometry = geometry
        nn.init.uniform_(self.network[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad():
            self.network[-1].bias[-1] = -4.0

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.network(torch.cat((state, action, context), dim=-1))
        successor = state + output[..., :-1]
        if self.joint_dt_scaled:
            # Integrate scaled joint velocity and let the network learn a residual for the substeps.
            joint_step = 0.5 * self.joint_dt_scaled * (state[..., self.joint_velocity] + successor[..., self.joint_velocity])
            start, stop = self.joint_position.start, self.joint_position.stop
            successor = torch.cat(
                (successor[..., :start], successor[..., start:stop] + joint_step, successor[..., stop:]), dim=-1
            )
        # Gravity is a direction, not three independent Euclidean coordinates.
        # Interpret the learned increment as a rotation. Rodrigues' formula with
        # sinc avoids a singular derivative at zero rotation.
        rotation = output[..., 6:9]
        angle = rotation.norm(dim=-1, keepdim=True)
        gravity = F.normalize(state[..., 6:9], dim=-1)
        cross = torch.cross(rotation, gravity, dim=-1)
        gravity = (gravity + torch.sinc(angle / torch.pi) * cross
                   + 0.5 * torch.sinc(angle / (2 * torch.pi)).square()
                   * torch.cross(rotation, cross, dim=-1))
        successor = torch.cat((successor[..., :6], gravity, successor[..., 9:]), dim=-1)
        if self.geometry is not None:
            separation, orientation, _ = self.geometry(successor[..., self.joint_position], gravity)
            # Foot separation/orientation cannot vary independently of the joints.
            # Clearance remains terrain-dependent and is learned separately.
            index = self.joint_velocity.stop + 2
            successor = torch.cat((successor[..., :index], separation * 5.0,
                                   orientation, successor[..., index + 2:]), dim=-1)
        return successor, output[..., -1]

    def rollout_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        state = batch["state"][:, 0]
        valid = torch.ones_like(batch["done"][:, 0], dtype=torch.bool)
        state_loss = state.new_zeros(())
        fall_loss = state.new_zeros(())
        valid_steps = state.new_zeros(())
        for step in range(batch["done"].shape[1]):
            target = batch["next_state"][:, step]
            state, fall_logit = self(state, batch["applied_action"][:, step], batch["context"][:, step])
            state_loss = state_loss + ((state - target).square().mean(dim=-1) * valid).sum()
            fall_bce = F.binary_cross_entropy_with_logits(
                fall_logit, batch["terminated"][:, step].float(), reduction="none"
            )
            fall_loss = fall_loss + (fall_bce * valid).sum()
            valid_steps = valid_steps + valid.sum()
            # The final transition is valid. The reset state is not its successor.
            valid = valid & ~batch["done"][:, step]
            state = torch.where(valid[:, None], state, target)
        return state_loss / valid_steps, fall_loss / valid_steps


def lambda_returns(
    rewards: torch.Tensor, discounts: torch.Tensor, next_values: torch.Tensor, gae_lambda: float
) -> torch.Tensor:
    """TD(lambda) targets for a time-major rollout with a value bootstrap."""
    target = next_values[-1]
    targets = []
    for reward, discount, next_value in zip(rewards.flip(0), discounts.flip(0), next_values.flip(0)):
        target = reward + discount * torch.lerp(next_value, target, gae_lambda)
        targets.append(target)
    return torch.stack(targets[::-1])
