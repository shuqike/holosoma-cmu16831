"""Settings shared by the current trainer and the Isaac Gym release adapter."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class SuperMimicConfig:
    # Version the algorithm separately from the immutable G1 task protocol.
    implementation: str = "public_gencoord_v3"
    num_learning_iterations: int = 25000
    num_steps_per_env: int = 2
    replay_steps: int = 1000
    replay_storage_device: str = "cpu"
    # Historical replay is opt-in: the 50% ablation slowed policy learning.
    model_history_fraction: float = 0.0
    history_capacity: int = 16384
    history_holdout_capacity: int = 2048
    history_window: int = 32
    history_candidates: int = 512
    history_seed: int = 20261111
    model_horizon: int = 8
    imagination_horizon: int = 32
    checkpoint_imagination: bool = True
    batch_size: int = 4096
    model_hidden_dim: int = 2048
    actor_hidden_dim: int = 256
    # Public source uses this stack for each encoder and the shared decoder.
    actor_hidden_dims: Tuple[int, ...] = (2048, 1024, 512, 256)
    actor_latent_dim: int = 32
    critic_hidden_dim: int = 256
    critic_second_hidden_dim: int = 128
    policy_warmup_iterations: int = 0
    # Optional legacy guard. Algorithm A.1 updates actor/critic every iteration
    # once a model-fitting window exists; validation must not silently change UTD.
    model_readiness_gate: bool = False
    learning_rate: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0
    exploration_noise: float = 0.1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    actor_beta2: float = 0.95
    actor_grad_clip_value: float = 0.1
    lr_decay_iterations: int = 250000
    lr_final_fraction: float = 0.03
    update_to_data_ratio: float = 0.4
    logging_interval: int = 50
    save_interval: int = 500
    eval_callbacks: Any = None
    model_batch_size: Optional[int] = 2048
    model_learning_rate: Optional[float] = None
    model_updates: int = 10
    holdout_stride: int = 0
    kinematic_joint_prior: bool = True
    normalize_action_history: bool = True
    detach_fall_discount: bool = False
    real_critic_weight: float = 0.0
    exploration_joint_std: Optional[float] = None
    exploration_correlation: float = 0.0


@dataclass(frozen=True)
class SuperMimicAlgoConfig:
    _target_: str
    _recursive_: bool
    config: SuperMimicConfig
