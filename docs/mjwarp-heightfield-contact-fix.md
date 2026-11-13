# MJWarp heightfield contact fix

The local `heightfield-contact-v3` patch fixes two classes of collision errors in
MuJoCo Warp commit `ecaef88917a3c90cd238bf76681ca770f58033df`, used with MuJoCo
3.10.0 and Warp 1.15.0. It changes three functions in the installed dependency's
`mujoco_warp/_src/collision_gjk.py`. It does not reduce terrain size or alter the
task's physics timestep, gains, randomization, or action clipping.

## What is fixed

- `_tri_affine_coord`: compute barycentric coordinates from local edge vectors
  instead of expanded absolute-coordinate determinants. Select the projection
  with the largest normal component, solve for `b` and `c`, and return
  `(1 - b - c, b, c)`.
- `_inflate`: preserve GJK witness points and signed distance, then apply margins
  using the generic path. The old heightfield branch used a witness position as
  a support direction, could select the opposite capsule endpoint, and returned
  a negative distance even for separated objects.
- `_epa_witness`: preserve the converged EPA face, its interpolated witness
  points, and its penetration depth. Remove the analogous heightfield-specific
  support-point reselection in the deep-penetration path.

The original failure involved roughly 10 cm terrain cells at heightfield-local
coordinates around 37–45 m. A traced barycentric weight sum was 1.02078 rather
than 1. This amplified the coordinate offset and produced a spurious 1.205459 m
penetration; CPU MuJoCo reported about 0.002488 m for the same saved state.
The observed failure chain was incorrect contacts, abnormal constraint response
and acceleration, diverging velocities, and eventually non-finite state.

The two contact-reprojection fixes address separate geometry errors. In
particular, fixing affine coordinates and shallow-contact inflation alone still
failed the larger stress test; the EPA path also needed correction.

Upstream PR [google-deepmind/mujoco_warp#1953](https://github.com/google-deepmind/mujoco_warp/pull/1953)
contains the affine-coordinate fix and its regression only. This local patch
also includes the two witness fixes; it is not equivalent to that PR alone.

## Installation

The three MuJoCo setup scripts apply the patch automatically. Their environment
sentinels include `heightfield-contact-v3` so an older setup marker does not skip
the fix. For an existing environment, run from the repository root:

```bash
python scripts/patch_mujoco_warp.py
python scripts/patch_mujoco_warp.py --check
```

The patcher verifies all three function fingerprints before writing anything,
refuses unknown versions, preserves a `.py.pre_holosoma_affine` backup, and is
idempotent. An upstream version change requires review, not removal of the
fingerprint guard. Restart processes that already imported the dependency.

## Reproduce and validate

The installer tests need only Python and pytest. The contact tests additionally
need the pinned MuJoCo/Warp packages and a CUDA device; they do not import
Holosoma, Torch, or an experimental agent.

```bash
python -m pytest -q tests/test_mjwarp_dependency_patch.py tests/test_mjwarp_contacts.py
```

The three GPU tests cover 90 affine-coordinate cases (translations, projections,
windings, and interior/exterior/vertex queries), signed shallow contact across
zero separation, and a known EPA penetration with known witness points.

For the full Holosoma environment:

```bash
# Fixed random actions, full-size mixed terrain, no policy or optimizer.
bash scripts/validate_mjwarp_nan_fix.sh python logs/mjwarp-validation

# Capture a state preceding an abnormal velocity spike if one occurs.
python scripts/diagnose_mjwarp_nan.py --capture-spike --steps 5000 \
  --output logs/mjwarp-capture/state.json

# Requires the .mjb and .npz siblings produced by a captured spike.
python scripts/replay_mjwarp_contact.py logs/mjwarp-capture/state.json \
  --verify --output logs/mjwarp-capture/replay.json

# Bounded PPO training, including checkpoint/optimizer finiteness checks.
python scripts/validate_mjwarp_training.py \
  --output logs/mjwarp-ppo-validation/summary.json
```

The standalone diagnostic uses the checked-in G1 FastSAC task configuration with
MJWarp, without constructing an agent. Its Gaussian actions retain the original
probe's per-joint range scaling and clipping. Every physics substep is checked
before the environment can reset; control steps also check observations and
rewards. Checks synchronize the GPU, so these runs are not throughput benchmarks.
The replay compares each GPU contact depth with the closest CPU depth for the
same geometry pair, with a 20 micrometer tolerance. It does not prove identical
contact manifolds or trajectories.

## Recorded validation before branch extraction

The earlier investigation ran the exact v3 collision patch on the full 160 by
240 m heightfield at 200 Hz physics and 50 Hz control, with default randomization:

| Seed | Environments | Action standard deviation before joint scaling | Control steps | Result |
| --- | ---: | ---: | ---: | --- |
| 42 | 512 | 0.1 | 5000 | No NaN/Inf |
| 2718 | 512 | 0.1 | 5000 | No NaN/Inf |
| 1 | 512 | 0.1 | 5000 | No NaN/Inf |
| 42 | 4096 | 0.5 | 5000 | No NaN/Inf |

Together these comprise 28,160,000 environment control transitions. A separate
512-environment PPO run completed 100 iterations with finite physics,
observations, rewards, model parameters, and optimizer tensors. The original
and stress-failure snapshots passed CPU/GPU contact-depth comparison after the
fix. The investigation also passed 22 upstream GJK tests and 3 upstream
heightfield-driver cases, in addition to the 6 local tests.

These are historical validation results, not a claim of exhaustive terrain or
policy coverage. The earlier fixed-action probe used an experimental locomotion
wrapper; branch extraction replaces it with the base locomotion manager and
inlines the same action-bound calculation to remove that unrelated dependency.
Large binary snapshots, training checkpoints, and raw logs remain local under
`logs/mjwarp_nan_investigation/`; they are not required for the synthetic tests
and are not included in the commit.

## Branch extraction checks (2026-11-12)

A fresh installation of the pinned dependency failed all three local collision
tests before patching. After applying v3, all 6 local tests and all 22 upstream
GJK tests passed on an RTX 4090. Both saved failure states were replayed again,
each at its original position and after a common world translation. Maximum
contact-depth errors were 1.286/2.015 micrometers for the original snapshot and
13.965/5.072 micrometers for the stress snapshot, all below the 20 micrometer
tolerance. Ruff lint/format, shell syntax, and staged whitespace checks passed.

The full rollout and PPO runs above were not repeated during branch extraction;
in particular, the diagnostic's new base-manager entry point has not yet been
rerun through that full protocol. The collision patch itself is unchanged.
