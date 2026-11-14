"""Protocol checks with fake clocks/files; no simulator, policies, or CUDA."""

import ast
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.sem_experiment_utils import DEFAULT_ITERATIONS, PROTOCOL, discover_results_root, timestamps

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "timed_recorder", ROOT / "src/holosoma/holosoma/agents/fast_sac/timed_experiment.py"
)
recorder_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder_module)


def test_default_and_custom_schedules_include_end():
    assert timestamps(1800) == [900, 1800]
    assert timestamps(120) == [120]
    assert timestamps(1200, [900, 300, 600, 300]) == [300, 600, 900, 1200]
    assert timestamps(100000) == [900, 100000]


@pytest.mark.parametrize("budget, marks", [
    (0, None), (math.inf, None), (900, [901]), (900, [0]), (900, [math.nan]),
])
def test_invalid_schedules(budget, marks):
    with pytest.raises(ValueError):
        timestamps(budget, marks)


def test_results_directory_is_inferred_from_budget(tmp_path):
    expected = tmp_path / "budget_1800s"
    expected.mkdir()
    assert discover_results_root(tmp_path, 1800, [1, 2]) == expected


def custom_result(root, study, run, budget=1800, seed=1):
    path = root / study / run
    path.mkdir(parents=True)
    (path / "run.json").write_text(json.dumps(dict(
        budget_s=budget, seed=seed, variant="baseline", smoke_test=False, status="finished", return_code=0)))
    return path


def test_custom_directory_is_discovered_without_an_input_argument(tmp_path):
    custom_result(tmp_path, "my_study", "attempt")
    assert discover_results_root(tmp_path, 1800, [1]) == tmp_path / "my_study"


def test_custom_directory_discovery_filters_budget_and_seeds(tmp_path):
    custom_result(tmp_path, "wrong_budget", "a", budget=900)
    custom_result(tmp_path, "wrong_seed", "b", seed=5)
    custom_result(tmp_path, "selected", "c")
    assert discover_results_root(tmp_path, 1800, [1]) == tmp_path / "selected"


def test_custom_directory_discovery_does_not_choose_between_studies(tmp_path):
    custom_result(tmp_path, "first", "a")
    custom_result(tmp_path, "second", "b")
    with pytest.raises(ValueError, match="Multiple custom studies"):
        discover_results_root(tmp_path, 1800, [1])
    assert discover_results_root(tmp_path, 1800, [1], ["a"]) == tmp_path / "second"


def test_missing_budget_does_not_fall_back_to_other_data(tmp_path):
    custom_result(tmp_path, "wrong_budget", "a", budget=900)
    with pytest.raises(ValueError, match="No matching results"):
        discover_results_root(tmp_path, 1800, [1])


def test_snapshot_clock_and_dense_window(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(recorder_module.time, "monotonic", lambda: clock[0])
    recorder = recorder_module.TimedExperimentRecorder(tmp_path, 1000, 1800, [900, 1800], 5)
    assert not recorder.should_log(894.99)
    assert recorder.should_log(895)
    recorder.observe(100, 899.9, {"Train/num_samples": 101 * 4096})
    calls = []

    def save(path):
        Path(path).write_bytes(b"checkpoint")
        calls.append(path)
        return {"elapsed_wall_time_s": clock[0] - 1000}

    def export(path):
        Path(path).write_bytes(b"onnx")
        clock[0] += 2  # Export delay is distinct from snapshot time.

    clock[0] = 1900.125
    assert recorder.capture_due(100, save, export)
    record = recorder.records[0]
    assert record["checkpoint_offset_s"] == pytest.approx(.125)
    assert not record["checkpoint_within_requested_time"]
    assert record["latest_observation"]["elapsed_wall_time_s"] == 899.9
    assert not recorder.capture_due(100, save, export)
    assert len(calls) == 1
    recorder.publish(finished=True)
    manifest = json.loads((tmp_path / "timestamps.json").read_text())
    assert manifest["unreached_timestamps_s"] == [1800]
    assert manifest["finished"]


def test_blocked_iteration_does_not_invent_intermediate_policies(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder_module.time, "monotonic", lambda: 1000)
    recorder = recorder_module.TimedExperimentRecorder(tmp_path, 0, 1800, [300, 600, 900, 1800])
    recorder.observe(10, 1000, {"Train/num_samples": 40960})
    calls = []

    def save(path):
        calls.append(path)
        return {"elapsed_wall_time_s": 1000}

    recorder.capture_due(10, save, lambda path: None)
    assert len(calls) == 1
    assert [r["requested_elapsed_s"] for r in recorder.records] == [300, 600, 900]
    assert [r["checkpoint_offset_s"] for r in recorder.records] == [700, 400, 100]


def test_pre_target_snapshot_preserves_a_strictly_eligible_policy(tmp_path, monkeypatch):
    clock = [895.1]
    monkeypatch.setattr(recorder_module.time, "monotonic", lambda: clock[0])
    recorder = recorder_module.TimedExperimentRecorder(tmp_path, 0, 900, [900], 5)
    save = lambda path: {"elapsed_wall_time_s": clock[0]}
    assert recorder.capture_due(100, save, lambda path: None)
    assert not recorder.records
    assert recorder.pre_records[0]["checkpoint_within_requested_time"]
    assert not recorder.capture_due(101, save, lambda path: None)
    clock[0] = 900.1
    assert recorder.capture_due(102, save, lambda path: None)
    assert not recorder.records[0]["checkpoint_within_requested_time"]
    assert recorder.pre_records[0]["checkpoint"] != recorder.records[0]["checkpoint"]


def test_partial_diagnostic_does_not_hide_latest_complete_observation(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder_module.time, "monotonic", lambda: 900.1)
    recorder = recorder_module.TimedExperimentRecorder(tmp_path, 0, 900, [900])
    recorder.observe(100, 899.5, {tag: 1.0 for tag in recorder.REQUIRED_SCALARS})
    recorder.observe(101, 900.1, {"Train/num_samples": 4096})
    recorder.capture_due(101, lambda path: {"elapsed_wall_time_s": 900.1}, lambda path: None)
    record = recorder.records[0]
    assert record["observation_available"]
    assert record["latest_observation"]["elapsed_wall_time_s"] == 900.1
    assert record["latest_complete_observation"]["elapsed_wall_time_s"] == 899.5


def test_scalar_objects_are_normalized_before_observation_and_manifest_writes(tmp_path, monkeypatch):
    class TensorScalar:
        def detach(self):
            return self

        def item(self):
            return 0.5

    monkeypatch.setattr(recorder_module.time, "monotonic", lambda: 900.1)
    recorder = recorder_module.TimedExperimentRecorder(tmp_path, 0, 900, [900])
    recorder.observe(100, 899.5, {tag: TensorScalar() for tag in recorder.REQUIRED_SCALARS})
    recorder.capture_due(100, lambda path: {"elapsed_wall_time_s": 900.1}, lambda path: None)
    observation = json.loads((tmp_path / "observations.jsonl").read_text())
    manifest = json.loads((tmp_path / "timestamps.json").read_text())
    assert observation["scalars"]["Train/mean_reward"] == 0.5
    assert manifest["records"][0]["latest_complete_observation"] == observation


@pytest.mark.parametrize("value", [math.nan, math.inf, object()])
def test_invalid_diagnostics_do_not_leave_partial_observations(tmp_path, value):
    recorder = recorder_module.TimedExperimentRecorder(tmp_path, 0, 900, [900])
    with pytest.raises((TypeError, ValueError), match="bad_metric"):
        recorder.observe(1, 1, {"bad_metric": value})
    assert not (tmp_path / "observations.jsonl").exists()
    assert recorder.latest is None


def batch_classifier(tmp_path):
    # The batch stays a shell entry point. Load only its pure classification
    # functions, without executing its activation, process checks, or launches.
    text = (ROOT / "scripts/run_matched_sem_seeds.sh").read_text()
    source = text.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("argument", "classify")]
    context = dict(json=json, math=math, RESULTS=tmp_path, PROTOCOL=PROTOCOL,
                   MARKS=[900, 1800], TRAINING_SOURCE="same-source",
                   ARGS=SimpleNamespace(seconds=1800, iterations=DEFAULT_ITERATIONS,
                                        logging_interval=100, timestamp_window_seconds=5))
    exec(compile(ast.Module(body=functions, type_ignores=[]), "batch-classifier", "exec"), context)
    return context["classify"]


def full_run(tmp_path):
    run = tmp_path / "attempt_baseline_seed1"
    train = run / "training/project/experiment"
    train.mkdir(parents=True)
    metadata = dict(seed=1, variant="baseline", smoke_test=False, budget_s=1800,
                    protocol=PROTOCOL, timestamps_s=[900, 1800], timestamp_window_s=5,
                    training_source_sha256="same-source", base_revision="9c238cf8",
                    simulator="Isaac Gym Preview 4", torch_compile_cache="cold", simulator_cache="shared",
                    status="finished", return_code=0,
                    command=["exp:g1-29dof-fast-sac", "simulator:isaacgym",
                             "--training.num-envs", "4096", "--training.seed", "1",
                             "--terrain.terrain-term.scale-factor", "1.0",
                             "--algo.config.num-learning-iterations", str(DEFAULT_ITERATIONS),
                             "--algo.config.logging-interval", "100"])
    (run / "run.json").write_text(json.dumps(metadata))
    (train / "timing.json").write_text(json.dumps(dict(
        budget_s=1800.000000000001, training_elapsed_wall_time_s=1800.1, global_step=9999)))
    (train / "model_0009999.pt").write_bytes(b"pt")
    (train / "model_0009999.onnx").write_bytes(b"onnx")
    (train / "events.out.tfevents.test").write_bytes(b"x" * 100)
    records = [dict(requested_elapsed_s=t, observation_available=True,
                    checkpoint="model_0009999.pt", onnx="model_0009999.onnx") for t in (900, 1800)]
    (train / "timestamps.json").write_text(json.dumps(dict(
        finished=True, timestamps_s=[900, 1800], records=records)))
    return run, train, metadata


def test_batch_skips_complete_arbitrary_budget_with_roundoff(tmp_path):
    run, _, _ = full_run(tmp_path)
    complete, unfinished = batch_classifier(tmp_path)(1, "baseline")
    assert complete == [run.name]
    assert not unfinished


def test_batch_retries_missing_timestamp_or_final_artifact(tmp_path):
    _, train, _ = full_run(tmp_path)
    (train / "model_0009999.onnx").unlink()
    complete, unfinished = batch_classifier(tmp_path)(1, "baseline")
    assert not complete and unfinished


def test_batch_does_not_reuse_different_source_or_schedule(tmp_path):
    run, _, metadata = full_run(tmp_path)
    metadata["training_source_sha256"] = "different-source"
    (run / "run.json").write_text(json.dumps(metadata))
    assert batch_classifier(tmp_path)(1, "baseline") == ([], [])
