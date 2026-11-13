"""Apply the reviewed heightfield-contact fixes to the pinned MJWarp dependency.

The original function is checked by SHA-256; unknown upstream code is never
rewritten. Re-running is idempotent. No CUDA device is required for installation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path

ORIGINAL_SHA256 = "87be4d4b19c3fc914f64ab0e653c957d6ad8be1739972e9bc08ec8fe3478db5c"
INFLATE_SHA256 = "bc56715bf453eee58262fcd5bb0f0a7dbed50bfbca5df29ab034bc75a89299d3"
EPA_SHA256 = "914424f0e84c916a2daa79229e49e5acdc2bcff478c85197d7ec4a56e8f56534"
FIXED_FUNCTION = """def _tri_affine_coord(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3, p: wp.vec3) -> wp.vec3:
  # Work relative to one vertex: absolute-coordinate determinants lose the
  # small triangle area through cancellation on large height fields.
  e0 = v2 - v1
  e1 = v3 - v1
  q = p - v1
  normal = wp.cross(e0, e1)
  x = int(0)
  y = int(1)
  if wp.abs(normal[0]) >= wp.abs(normal[1]) and wp.abs(normal[0]) >= wp.abs(normal[2]):
    x = 1
    y = 2
  elif wp.abs(normal[1]) >= wp.abs(normal[2]):
    x = 0
    y = 2
  det = e0[x] * e1[y] - e0[y] * e1[x]
  b = (q[x] * e1[y] - q[y] * e1[x]) / det
  c = (e0[x] * q[y] - e0[y] * q[x]) / det
  return wp.vec3(1.0 - b - c, b, c)"""

FIXED_INFLATE = """def _inflate(
  result: GJKResult, geom1: Geom, geom2: Geom, geomtype1: int, geomtype2: int, margin1: float, margin2: float
) -> Tuple[float, wp.vec3, wp.vec3]:
  # Preserve the GJK witnesses and signed distance, as in CPU MuJoCo 3.10.
  # Re-projecting hfield side contacts using a position as a support direction
  # can pick the opposite capsule end and turn separation into penetration.
  dist = result.dist
  x1 = result.x1
  x2 = result.x2
  n = wp.normalize(x2 - x1)
  if margin1 > 0.0:
    x1 += margin1 * n
  if margin2 > 0.0:
    x2 -= margin2 * n
  dist -= margin1 + margin2
  return dist, x1, x2"""

FIXED_EPA = """def _epa_witness(
  pt: Polytope, geom1: Geom, geom2: Geom, geomtype1: int, geomtype2: int, face_idx: int
) -> Tuple[wp.vec3, wp.vec3, float]:
  face = _get_face_verts(pt.face[face_idx])
  # Preserve EPA's closest face and penetration depth for heightfields too.
  # Re-selecting support with a witness position as a direction changes the
  # capsule endpoint and no longer represents the converged EPA solution.
  v1 = pt.vert[2 * face[0]] - pt.vert[2 * face[0] + 1]
  v2 = pt.vert[2 * face[1]] - pt.vert[2 * face[1] + 1]
  v3 = pt.vert[2 * face[2]] - pt.vert[2 * face[2] + 1]
  coordinates = _tri_affine_coord(v1, v2, v3, pt.face_pr[face_idx])
  l1 = coordinates[0]
  l2 = coordinates[1]
  l3 = coordinates[2]

  v1 = pt.vert[2 * face[0] + 1]
  v2 = pt.vert[2 * face[1] + 1]
  v3 = pt.vert[2 * face[2] + 1]
  x2 = wp.vec3()
  x2[0] = v1[0] * l1 + v2[0] * l2 + v3[0] * l3
  x2[1] = v1[1] * l1 + v2[1] * l2 + v3[1] * l3
  x2[2] = v1[2] * l1 + v2[2] * l2 + v3[2] * l3

  v1 = pt.vert[2 * face[0]]
  v2 = pt.vert[2 * face[1]]
  v3 = pt.vert[2 * face[2]]
  x1 = wp.vec3()
  x1[0] = v1[0] * l1 + v2[0] * l2 + v3[0] * l3
  x1[1] = v1[1] * l1 + v2[1] * l2 + v3[1] * l3
  x1[2] = v1[2] * l1 + v2[2] * l2 + v3[2] * l3
  return x1, x2, -wp.sqrt(pt.face_norm2[face_idx])"""


def function_source(source: str, name: str = "_tri_affine_coord") -> str:
    matches = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one MJWarp {name} function; review the dependency version")
    return ast.get_source_segment(source, matches[0])


def patch(source_path: Path, *, check: bool = False) -> str:
    source = source_path.read_text(encoding="utf-8")
    replacements = []
    for name, expected_hash, fixed in (
        ("_tri_affine_coord", ORIGINAL_SHA256, FIXED_FUNCTION),
        ("_inflate", INFLATE_SHA256, FIXED_INFLATE),
        ("_epa_witness", EPA_SHA256, FIXED_EPA),
    ):
        old = function_source(source, name)
        if old == fixed:
            continue
        digest = hashlib.sha256(old.encode()).hexdigest()
        if digest != expected_hash:
            raise RuntimeError(f"Unrecognized MJWarp {name} ({digest}); refusing to rewrite {source_path}")
        replacements.append((old, fixed))
    if not replacements:
        return "already patched"
    if check:
        raise RuntimeError("MJWarp heightfield contact fix is missing; run scripts/patch_mujoco_warp.py")
    backup = source_path.with_suffix(".py.pre_holosoma_affine")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    for old, fixed in replacements:
        source = source.replace(old, fixed, 1)
    source_path.write_text(source, encoding="utf-8")
    return "patched"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.find_spec("mujoco_warp")
    if spec is None or spec.origin is None:
        raise RuntimeError("mujoco_warp is not installed in this Python environment")
    path = Path(spec.origin).parent / "_src" / "collision_gjk.py"
    print(f"MJWarp heightfield-contact fix: {patch(path, check=args.check)} ({path})")


if __name__ == "__main__":
    main()
