"""Source-bound validation gates for the Isaac Gym experiment."""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_report(path):
    report = json.loads(path.read_text())
    checks = (report.get("validation_schema") == "complete_actor_update_v1",
              report.get("status") == "passed",
              report.get("protocol_equal"), report.get("physical_gate"), report.get("critic_gate"),
              report.get("numerical_model_gradient_gate"), report.get("full_actor_update_gate"),
              report.get("initial_state_match"),
              report.get("benchmark_authorized_by_gates"))
    if not all(checks):
        raise ValueError("Validation gates did not all pass; no benchmark will be launched.")
    source = REPO / "src/holosoma/holosoma/agents/supermimic"
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")}
    if actual != report["source_sha256"]:
        raise ValueError("Algorithm source changed since validation; validate these exact sources first.")
    scripts = report.get("validation_script_sha256", {})
    if not scripts or any(hashlib.sha256((REPO / "scripts" / name).read_bytes()).hexdigest() != digest
                          for name, digest in scripts.items()):
        raise ValueError("Validation scripts changed or were not recorded; rerun complete validation.")
    checkpoint = Path(report["checkpoint"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
        raise ValueError("Validated checkpoint contents changed.")
    preparation = json.loads(Path(report["preparation"]).read_text())
    if preparation["checkpoint_sha256"] != report["checkpoint_sha256"] or preparation["source_sha256"] != actual:
        raise ValueError("Complete optimizer update preparation does not match the validated checkpoint/source.")
    if not preparation.get("actor_update_from_training_replay") or preparation["production_replay_snapshot"]["retained_replay_steps"] != preparation["algorithm_config"]["replay_steps"]:
        raise ValueError("Complete update must use the filled production FIFO snapshot.")
    return report
