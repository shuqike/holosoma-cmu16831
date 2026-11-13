"""Replay a captured physics input without Holosoma or a policy."""

import argparse
import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--verify", action="store_true", help="Require each GPU contact depth to match a CPU contact")
args = parser.parse_args()
wp.init()
results = []
for centered in (False, True):
    model = mujoco.MjModel.from_binary_path(str(args.source.with_suffix(".mjb")))
    state = dict(np.load(args.source.with_suffix(".npz")))
    if centered:
        offset = state["qpos"][:2].copy()
        model.geom_pos[model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD, :2] -= offset
        state["qpos"][:2] -= offset
    data = mujoco.MjData(model)
    for name, value in state.items():
        getattr(data, name)[:] = value
    mujoco.mj_forward(model, data)
    with wp.ScopedDevice("cuda:0"):
        wm = mjw.put_model(model)
        wd = mjw.make_data(model, nworld=1, nconmax=96, njmax=576)
        for name, value in state.items():
            getattr(wd, name).assign(value.astype(np.float32)[None])
        mjw.forward(wm, wd)
        wp.synchronize()
        ncon = int(wd.nacon.numpy()[0])
        row = {
            "centered": centered,
            "cpu_dist": data.contact.dist.tolist(),
            "cpu_geom": data.contact.geom.tolist(),
            "gpu_dist": wd.contact.dist.numpy()[:ncon].tolist(),
            "gpu_geom": wd.contact.geom.numpy()[:ncon].tolist(),
            "cpu_max_qacc": float(np.abs(data.qacc).max()),
            "gpu_max_qacc": float(np.abs(wd.qacc.numpy()).max()),
        }
        if args.verify:
            if not np.isfinite(row["gpu_dist"]).all():
                raise AssertionError("GPU produced a non-finite contact depth")
            if data.ncon and not ncon:
                raise AssertionError("GPU dropped all contacts from the CPU contact state")
            errors = []
            for pair, depth in zip(row["gpu_geom"], row["gpu_dist"], strict=True):
                candidates = [
                    d for g, d in zip(row["cpu_geom"], row["cpu_dist"], strict=True) if sorted(g) == sorted(pair)
                ]
                if not candidates:
                    raise AssertionError(f"GPU-only contact pair: {pair}")
                errors.append(min(abs(depth - d) for d in candidates))
            row["max_contact_depth_error"] = max(errors, default=0.0)
            if not np.isfinite(wd.qacc.numpy()).all() or row["max_contact_depth_error"] > 2e-5:
                raise AssertionError(f"CPU/GPU contact regression: {row}")
        results.append(row)
        print(json.dumps(row), flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(results, indent=2) + "\n")
