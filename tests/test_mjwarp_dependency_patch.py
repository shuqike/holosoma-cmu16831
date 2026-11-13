"""The installer must validate all replacements before modifying a dependency."""

import hashlib
import runpy
from pathlib import Path

import pytest

PATCHER = Path(__file__).resolve().parents[1] / "scripts" / "patch_mujoco_warp.py"
pytestmark = pytest.mark.no_sim


def test_patch_is_idempotent_and_check_does_not_write(tmp_path):
    api = runpy.run_path(str(PATCHER))
    path = tmp_path / "collision_gjk.py"
    text = "\n\n".join(api[name] for name in ("FIXED_FUNCTION", "FIXED_INFLATE", "FIXED_EPA")) + "\n"
    path.write_text(text)
    before = path.stat().st_mtime_ns
    assert api["patch"](path, check=True) == "already patched"
    assert api["patch"](path) == "already patched"
    assert path.stat().st_mtime_ns == before
    assert path.read_text() == text
    assert not path.with_suffix(".py.pre_holosoma_affine").exists()


def test_unknown_second_function_cannot_leave_a_partial_patch(tmp_path, monkeypatch):
    api = runpy.run_path(str(PATCHER))
    patch = api["patch"]
    old_first = "def _tri_affine_coord(a, b, c, p):\n    return a"
    # Simulate a recognized first function but an unreviewed second function.
    monkeypatch.setitem(patch.__globals__, "ORIGINAL_SHA256", hashlib.sha256(old_first.encode()).hexdigest())
    text = old_first + "\n\ndef _inflate(x):\n    return x\n\n" + api["FIXED_EPA"] + "\n"
    path = tmp_path / "collision_gjk.py"
    path.write_text(text)
    with pytest.raises(RuntimeError, match="Unrecognized MJWarp _inflate"):
        patch(path)
    assert path.read_text() == text
    assert not path.with_suffix(".py.pre_holosoma_affine").exists()


def test_known_source_patch_keeps_backup_and_check_only_refuses_missing_fix(tmp_path, monkeypatch):
    api = runpy.run_path(str(PATCHER))
    patch = api["patch"]
    first = "def _tri_affine_coord(a, b, c, p):\n    return a"
    second = "def _inflate(x):\n    return x"
    monkeypatch.setitem(patch.__globals__, "ORIGINAL_SHA256", hashlib.sha256(first.encode()).hexdigest())
    monkeypatch.setitem(patch.__globals__, "INFLATE_SHA256", hashlib.sha256(second.encode()).hexdigest())
    original = first + "\n\n" + second + "\n\n" + api["FIXED_EPA"] + "\n"
    path = tmp_path / "collision_gjk.py"
    path.write_text(original)
    with pytest.raises(RuntimeError, match="fix is missing"):
        patch(path, check=True)
    assert path.read_text() == original
    assert patch(path) == "patched"
    assert path.with_suffix(".py.pre_holosoma_affine").read_text() == original
    assert patch(path, check=True) == "already patched"
