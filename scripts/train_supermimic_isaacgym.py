"""Run the velocity-only SuperMimic adaptation on Holosoma's Isaac Gym release."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Union

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src/holosoma"))
DEFAULT_CONFIG = REPO / "configs/supermimic/isaacgym_g1.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--algorithm-config", type=Path, default=DEFAULT_CONFIG, help="Algorithm configuration JSON; task settings are unchanged")
    args = parser.parse_args()
    config = make_config(args.base_dir, args.num_envs, args.seed, args.iterations)
    if args.algorithm_config:
        algorithm = json.loads(args.algorithm_config.read_text())
        algorithm["num_learning_iterations"] = args.iterations
        config = replace(config, algo=replace(config.algo, config=replace(config.algo.config, **algorithm)))
    from holosoma.train_agent import train

    train(config)


def make_config(base_dir, num_envs, seed, iterations):

    # This branch pins both the task and algorithm to the Isaac Gym release.
    import isaacgym  # noqa: F401
    import holosoma.agents
    from holosoma.config_types import algo as legacy_algo

    repo = Path(__file__).resolve().parents[1]
    expected = (repo / "src/holosoma/holosoma/agents").resolve()
    if expected not in [Path(p).resolve() for p in holosoma.agents.__path__]:
        raise RuntimeError("A different Holosoma checkout was imported; start a fresh interpreter")
    from holosoma.agents.supermimic.config import SuperMimicAlgoConfig, SuperMimicConfig

    # The release's experiment schema validates the algorithm against this union.
    legacy_algo.AlgoConfig = Union[legacy_algo.AlgoConfig, SuperMimicAlgoConfig]
    from holosoma.config_values import logger, simulator
    from holosoma.config_values.loco.g1.experiment import g1_29dof_fast_sac

    fastsac = g1_29dof_fast_sac
    algorithm = SuperMimicAlgoConfig(
        _target_="holosoma.agents.supermimic.supermimic.SuperMimicAgent",
        _recursive_=False,
        config=SuperMimicConfig(
            num_learning_iterations=iterations,
            holdout_stride=16,
            policy_warmup_iterations=0,
            model_readiness_gate=False,
            learning_rate=2e-4,
            model_learning_rate=2e-4,
            kinematic_joint_prior=True,
            normalize_action_history=True,
            detach_fall_discount=False,
            real_critic_weight=0.0,
            exploration_joint_std=0.025,
            exploration_correlation=0.0,
            logging_interval=25,
            save_interval=250,
        ),
    )
    algorithm = replace(algorithm, config=replace(algorithm.config, **{
        **json.loads(DEFAULT_CONFIG.read_text()), "num_learning_iterations": iterations}))
    return replace(
        fastsac,
        env_class="holosoma.agents.supermimic.locomotion.SuperMimicLocomotion",
        algo=algorithm,
        simulator=simulator.isaacgym,
        training=replace(fastsac.training, name="g1_29dof_supermimic", num_envs=num_envs, seed=seed),
        logger=replace(logger.disabled, base_dir=str(base_dir)),
        nightly=None,
    )


if __name__ == "__main__":
    main()
