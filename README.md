# Virtual Gym Buddy Simulator Harness

Python-first simulation and validation harness for Gym Buddy. Nothing here ships in the Android app.

**Canonical product/spec authority:** Google Drive docs `00–07`  
https://drive.google.com/drive/folders/1KcxnGbShccsJgoz8A2GU-2IkB85EiWxL

> This repository is public. Never commit licensed raw datasets, personal media/profile data, credentials, caches, or generated artifacts.

## Setup

```bash
git clone https://github.com/lazerta/virtual-gym-buddy-simulator-harness.git
cd virtual-gym-buddy-simulator-harness
uv sync --extra dev
uv run pytest
uv run python -m harness.doctor
uv run python -m harness.cli validate-profiles --quick --fps 10
```

## Codex handoff

> Pull latest `main`, read this README and Drive docs `00–07`, set up with `uv`, then run `pytest`, `python -m harness.doctor`, and `python -m harness.cli validate-profiles --quick --fps 10`. Fix the correct layer; do not weaken tests just to get green. Keep raw Fit3D/other datasets, personal data, secrets, caches, virtualenvs, generated videos/CSVs, and large artifacts out of Git. If `FIT3D_ROOT` is absent, real Fit3D replay is `NOT_RUN`.

## Architecture rules

- `PrimarySubjectLock` and `TrackingQualityGate` are separate.
- Multiple people do not automatically pause analysis.
- Rep detection is temporal and uses reusable movement primitives.
- Form analysis is separate from rep detection.
- Generator, production/reference logic, and oracle remain independent.
- External/anonymous quick validation must pass before the personal stage runs.
- The public repo contains only a non-personal synthetic fallback profile.

## Private personal profile

Copy `config/personal_profile.example.json` somewhere **outside this repo**, fill it locally, then set:

```powershell
$env:GYM_BUDDY_PERSONAL_PROFILE="D:\\Private\\gym-buddy-personal-profile.json"
```

The real personal profile is never committed.

## Public real-motion baseline: MM-Fit

MM-Fit is the default accessible public baseline. Gym Buddy reads the official `pose_3d.npy` + `labels.csv` files directly; PyTorch and the upstream MM-Fit codebase are not required. Set:\n\n```powershell\n$env:MMFIT_ROOT="D:\\Datasets\\MMFit"\n```\n\nThen validate:\n\n```powershell\nuv run python -m harness.cli mmfit-check --load\n```\n\nDirectly useful movement-family bridges include squat, dumbbell shoulder press, dumbbell row, and seated lateral raise. These are **movement-family references**, not exact substitutes for Smith/bench/machine equipment. Exact equipment-specific motions still come from retargeted real gym video / MuscleMimic trajectories.\n\n## Optional Fit3D benchmark\n\nFit3D is not required for setup, release gates, or the scientific simulation backend. If you happen to have local access, it can still be used as an additional benchmark:\n\nFit3D stays local:

```powershell
$env:FIT3D_ROOT="D:\\Datasets\\Fit3D"
```

Detector input uses `joints3d_25`; `rep_ann` is independent oracle truth only. See `datasets/FIT3D_LOCAL_SETUP.md`.


## Scientific 3D backend

The visual renderer is not the source of biomechanics or physics. The scientific simulation backend uses:

- **MuJoCo** for articulated dynamics, contacts, and equipment rigid-body physics.
- **MyoSim / MyoFullBody** for the musculoskeletal human model.
- **MuscleMimic-compatible retargeted trajectories** as an optional motion source.
- **Three.js** only as a renderer/visualization client consuming exported simulator state.

Install and verify:

```bash
uv sync --extra mujoco
uv run python -m harness.cli mujoco-smoke
```

The JSONL state bridge can be launched with:

```bash
uv run gym-buddy-mujoco-bridge
```

It accepts one JSON object per line with operations such as `reset`, `state`, `step`, and `set_qpos`.


## Native scientific viewer

For the authoritative simulation view, use MuJoCo directly rather than the legacy
Three.js-authored motion path:

```bash
uv sync --extra mujoco
uv run gym-buddy-sim
```

Replay an upstream MuscleMimic/LocoMuJoCo MyoFullBody trajectory:

```bash
uv run gym-buddy-sim --motion /path/to/trajectory.npz
```

Or put canonical exercise trajectories under `GYM_BUDDY_MOTION_ROOT`
(default `motions/myofullbody`) using names such as
`smith_squat.npz`, then run:

```bash
uv run gym-buddy-sim --exercise smith_squat
```

Check all ten canonical motion slots:

```bash
uv run python -m harness.cli motion-status
```

The runtime reads MuscleMimic's own serialized `qpos`, `qvel`,
`split_points`, and `frequency` fields directly. No separate Gym Buddy
retargeting format is required.


## Real gym video → MyoFullBody

For exact equipment-specific motions that MM-Fit does not cover, the scientific
motion path reuses upstream open-source projects rather than hand-authored
animation:

```text
gym video
  -> WHAM monocular SMPL motion
  -> Gym Buddy AMASS/SMPL-H compatibility adapter
  -> MuscleMimic GMR
  -> MyoFullBody qpos
  -> MuJoCo
```

One-time prerequisites:

- clone/setup WHAM and set `WHAM_ROOT`;
- clone/setup MuscleMimic and set `MUSCLEMIMIC_ROOT`;
- obtain the required SMPL-family model assets under their own license terms;
- if WHAM uses a separate Python environment, set `WHAM_PYTHON`.

Then run one Gym Buddy command:

```powershell
uv run gym-buddy-video-motion --video .\videos\incline_smith_press.mp4 --exercise incline_smith_press
```

The canonical output is:

```text
motions/myofullbody/incline_smith_press.npz
```

The converter selects the longest continuous WHAM subject track, prefers world
coordinates when available, and converts WHAM's 72D SMPL pose using the same
body-pose convention used by MuscleMimic's AMASS exporter: preserve the first
66 axis-angle dimensions, zero the terminal two SMPL hand/wrist slots, and pad
to the 156D SMPL-H layout.

Use `--dry-run` to print the external WHAM and MuscleMimic commands without
running them.
