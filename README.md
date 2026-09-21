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

## Fit3D

Fit3D stays local:

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
