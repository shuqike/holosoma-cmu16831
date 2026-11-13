"""CPU reservoirs of real, contiguous windows across the entire training run.

Random priorities retain a uniform sample of all candidate windows, independently
of their age or error. Train and held-out worlds never share a reservoir. Only WM
supervision uses history; actor/critic imagination seeds remain in recent replay.
"""
from __future__ import annotations

from holosoma.utils.safe_torch_import import torch


class WindowReservoir:
    def __init__(self, capacity: int, length: int, seed: int):
        if capacity < 1 or length < 1:
            raise ValueError("History capacity and window length must be positive")
        self.capacity, self.length = capacity, length
        self.size = self.seen = 0
        self.data = {}
        self.keys = torch.empty(capacity, dtype=torch.float64)
        self.worlds = torch.empty(capacity, dtype=torch.long)
        self.starts = torch.empty(capacity, dtype=torch.long)
        # Separate RNGs keep retention independent of how often fitting/evaluation runs.
        self.capture_rng = torch.Generator().manual_seed(seed)
        self.sample_rng = torch.Generator().manual_seed(seed + 1)

    def __len__(self):
        return self.size

    @torch.no_grad()
    def capture(self, replay, fields, world_ids, candidates: int, global_end: int):
        if len(replay) < self.length:
            raise ValueError("History cannot capture an incomplete window")
        worlds = torch.as_tensor(world_ids, device="cpu", dtype=torch.long)
        if not worlds.numel():
            raise ValueError("History needs at least one source world")
        chosen = worlds[torch.randint(worlds.numel(), (candidates,), generator=self.capture_rng)]
        priorities = torch.rand(candidates, dtype=torch.float64, generator=self.capture_rng)
        total_keys = torch.cat((self.keys[:self.size], priorities))
        keep = total_keys.topk(min(self.capacity, total_keys.numel()), largest=False).indices
        incoming = keep[keep >= self.size] - self.size
        available = torch.ones(self.capacity, dtype=torch.bool)
        available[keep[keep < self.size]] = False
        destination = available.nonzero().flatten()[:incoming.numel()]
        if incoming.numel():
            device = replay.data["state"].device
            slots = torch.arange(replay.steps-self.length, replay.steps, device=device) % replay.capacity
            world_index = chosen[incoming].to(device)
            for name in fields:
                batch = replay.data[name][slots[None, :], world_index[:, None]].cpu()
                if name not in self.data:
                    self.data[name] = torch.empty((self.capacity, *batch.shape[1:]), dtype=batch.dtype)
                self.data[name][destination] = batch
            self.keys[destination] = priorities[incoming]
            self.worlds[destination] = chosen[incoming]
            self.starts[destination] = global_end - self.length
        self.size = min(self.capacity, self.size + candidates)
        self.seen += candidates

    def sample(self, count: int, length: int, fields, device):
        if not self.size or not 1 <= length <= self.length:
            raise ValueError("Invalid history sample size or horizon")
        windows = torch.randint(self.size, (count, 1), generator=self.sample_rng)
        offsets = torch.randint(self.length-length+1, (count, 1), generator=self.sample_rng)
        times = offsets + torch.arange(length)
        return {name: self.data[name][windows, times].to(device) for name in fields}

    def coverage(self, global_end: int, recent_steps: int):
        if not self.size:
            return dict(windows=0, candidates_seen=self.seen)
        age = global_end - self.starts[:self.size]
        return dict(windows=self.size, candidates_seen=self.seen,
                    unique_worlds=int(self.worlds[:self.size].unique().numel()),
                    oldest_start=int(self.starts[:self.size].min()),
                    newest_start=int(self.starts[:self.size].max()),
                    older_than_recent_fraction=float((age > recent_steps).float().mean()),
                    terminal_windows=int(self.data["terminated"][:self.size].any(1).sum()))

    def state_dict(self):
        return dict(capacity=self.capacity, length=self.length, size=self.size, seen=self.seen,
                    keys=self.keys[:self.size].clone(), worlds=self.worlds[:self.size].clone(),
                    starts=self.starts[:self.size].clone(),
                    data={name: value[:self.size].clone() for name, value in self.data.items()},
                    capture_rng=self.capture_rng.get_state(), sample_rng=self.sample_rng.get_state())

    def load_state_dict(self, saved):
        if (saved["capacity"], saved["length"]) != (self.capacity, self.length):
            raise ValueError("History checkpoint capacity/horizon differs from configuration")
        self.size, self.seen = saved["size"], saved["seen"]
        for name in ("keys", "worlds", "starts"):
            getattr(self, name)[:self.size].copy_(saved[name])
        self.data = {name: torch.empty((self.capacity, *value.shape[1:]), dtype=value.dtype)
                     for name, value in saved["data"].items()}
        for name, value in saved["data"].items():
            self.data[name][:self.size].copy_(value)
        self.capture_rng.set_state(saved["capture_rng"].cpu())
        self.sample_rng.set_state(saved["sample_rng"].cpu())
