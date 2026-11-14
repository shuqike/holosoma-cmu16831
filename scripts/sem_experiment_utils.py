"""Shared standard-library protocol helpers; no simulator or plotting imports."""

import hashlib
import json
import math
from pathlib import Path

PROTOCOL = "timed-milestones-v1"
DEFAULT_ITERATIONS = 2147483647  # Wall time, rather than 50k iterations, limits long runs.


def timestamps(seconds, requested=None):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Run duration must be finite and positive")
    marks = ([900.0] if seconds >= 900 else []) if requested is None else list(requested)
    if any(not math.isfinite(t) or t <= 0 or t > seconds for t in marks):
        raise ValueError("Timestamps must be finite, positive, and at most the run duration")
    return sorted(set(marks + [float(seconds)]))


def default_output_root(seconds):
    return Path("logs/sem_experiments") / ("budget_%ss" % format(seconds, ".15g"))


def discover_results_root(results_root, seconds, seeds, excluded=()):
    """Infer a local study from its budget, with metadata fallback for custom roots."""
    root = Path(results_root)
    canonical = root / default_output_root(seconds).name
    if canonical.is_dir():
        return canonical
    candidates = set()
    for path in root.rglob("run.json"):
        if path.parent.name in excluded:
            continue
        try:
            metadata = json.loads(path.read_text())
            matches = (
                math.isclose(float(metadata.get("budget_s", 0)), seconds, rel_tol=0, abs_tol=1e-6)
                and metadata.get("seed") in seeds
                and metadata.get("variant") in ("baseline", "sem")
                and metadata.get("smoke_test") is False
                and metadata.get("status") == "finished" and metadata.get("return_code") == 0
            )
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        if matches:
            candidates.add(path.parent.parent)
    if len(candidates) == 1:
        return candidates.pop()
    if candidates:
        choices = ", ".join(str(path.relative_to(root)) for path in sorted(candidates))
        raise ValueError(
            "Multiple custom studies match this budget/seeds: %s. "
            "Use --exclude-run to exclude unwanted attempts, or organize the selected study in %s."
            % (choices, canonical)
        )
    raise ValueError("No matching results found for budget %gs. Expected %s; sync results first." % (seconds, canonical))


def training_source_fingerprint(repo):
    digest = hashlib.sha256()
    root = Path(repo) / "src/holosoma/holosoma"
    for path in sorted(root.rglob("*.py")):
        if "tests" not in path.relative_to(root).parts:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
