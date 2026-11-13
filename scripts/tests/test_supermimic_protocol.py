"""The standalone experiment must preserve the released G1 task."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from train_supermimic_isaacgym import make_config


def test_task_matches_released_fastsac(tmp_path):
    config = make_config(tmp_path, 4096, 2, 5000)
    from holosoma.config_values.loco.g1.experiment import g1_29dof_fast_sac
    from holosoma.config_values import simulator
    assert asdict(config.simulator) == asdict(simulator.isaacgym)
    for field in ("robot", "terrain", "observation", "action", "command", "reward",
                  "termination", "randomization", "curriculum", "scene", "sensor", "plugin"):
        left, right = getattr(config, field, None), getattr(g1_29dof_fast_sac, field, None)
        assert left == right, field
    assert config.training.num_envs == 4096


def test_selected_algorithm_config_and_checkout(tmp_path):
    config = make_config(tmp_path, 64, 1, 17)
    expected = json.loads((ROOT / "configs/supermimic/isaacgym_g1.json").read_text())
    expected["num_learning_iterations"] = 17
    assert json.loads(json.dumps(asdict(config.algo.config))) == expected
    assert config.algo.config.model_hidden_dim == 1024
    assert config.algo.config.model_history_fraction == 0
    import holosoma.agents.supermimic.supermimic as implementation
    assert ROOT in Path(implementation.__file__).resolve().parents
