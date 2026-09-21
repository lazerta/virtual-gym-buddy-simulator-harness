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

## Architecture rules

- `PrimarySubjectLock` and `TrackingQualityGate` are separate.
- Multiple people do not automatically pause analysis.
- Rep detection is temporal and uses reusable movement primitives.
- Form analysis is separate from rep detection.
- Generator, production/reference logic, and oracle remain independent.
- External/anonymous quick validation must pass before the personal stage runs.
- The public repo contains only a non-personal synthetic fallback profile.
- **MuJoCo/MyoFullBody is the authoritative scientific 3D backend.**
- Three.js is visualization only; it must not invent an independent biomechanics solution.
- Real exercise motion should come from public pose/motion data or real-video retargeting, not hand-authored animation.

## Private personal profile

Copy `config/personal_profile.example.json` somewhere **outside this repo**, fill it locally, then set:

```powershell
$env:GYM_BUDDY_PERSONAL_PROFILE="D:\\Private\\gym-buddy-personal-profile.json"
```

The real personal profile is never committed.

## Public real-motion baseline: MM-Fit

MM-Fit is the default accessible public baseline. Gym Buddy reads the official
`pose_3d.npy` + `labels.csv` files directly; PyTorch and the upstream MM-Fit
codebase are not required.

```powershell
$env:MMFIT_ROOT="D:\\Datasets\\MMFit"
uv run python -m harness.cli mmfit-check --load
```

Directly useful movement-family bridges include squat, dumbbell shoulder press,
dumbbell row, and lateral shoulder raise. These are **movement-family
references**, not exact substitutes for Smith/bench/machine equipment.

## Optional Fit3D benchmark

Fit3D is **not required** for setup, release gates, or the scientific simulation
backend. If local access happens to be available, it can still be used as an
additional benchmark:

```powershell
$env:FIT3D_ROOT="D:\\Datasets\\Fit3D"
```

Detector input uses `joints3d_25`; `rep_ann` is independent oracle truth only.

## Scientific 3D backend

The scientific backend uses:

- **MuJoCo** for articulated dynamics, contacts, and equipment rigid-body physics.
- **MyoSim / MyoFullBody** for the musculoskeletal human model.
- **MuscleMimic/GMR-compatible trajectories** for real movement.
- **Three.js** only as a renderer/visualization client.

Install and verify:

```bash
uv sync --extra mujoco
uv run python -m harness.cli mujoco-smoke
```

The JSONL state bridge:

```bash
uv run gym-buddy-mujoco-bridge
```

It accepts `reset`, `state`, `step`, and `set_qpos`.

## Native scientific viewer

Use MuJoCo directly for the authoritative simulation view:

```bash
uv run gym-buddy-sim
```

Replay one retargeted trajectory:

```bash
uv run gym-buddy-sim --motion /path/to/trajectory.npz
```

Or store canonical trajectories under `GYM_BUDDY_MOTION_ROOT` (default
`motions/myofullbody`) and run:

```bash
uv run gym-buddy-sim --exercise smith_squat
```

Check all ten canonical motion slots:

```bash
uv run python -m harness.cli motion-status
```

## Real gym video → MyoFullBody

Exact equipment-specific motion uses existing open-source projects instead of
hand-authored animation:

```text
real gym video / public video URL
  -> yt-dlp (URL input only)
  -> WHAM world-grounded SMPL motion
  -> Gym Buddy AMASS/SMPL-H compatibility adapter
  -> MuscleMimic GMR
  -> MyoFullBody qpos
  -> MuJoCo
```

The WHAM converter selects the longest continuous subject track, prefers world
coordinates when available, preserves the first 66 SMPL axis-angle dimensions,
zeros the terminal two SMPL hand/wrist slots, and pads to the 156D SMPL-H layout
used by the MuscleMimic path.

### Windows / WSL2 one-command setup

WHAM's upstream installation targets Ubuntu/Python 3.9/CUDA, and MuscleMimic
inference supports Linux. On Windows, use WSL2.

From PowerShell in the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_motion_stack.ps1
```

That wrapper enters WSL2 and runs `scripts/setup_motion_stack_wsl.sh`, which:

- installs Miniforge and `uv` if missing;
- clones WHAM and MuscleMimic;
- installs WHAM's upstream Python 3.9 / CUDA 11.3 dependency stack;
- installs MuscleMimic with `smpl` + `gmr` extras;
- installs the Gym Buddy MuJoCo runtime;
- writes `~/gym-buddy-motion-stack/gym-buddy-motion.env`.

**Licensed model files are the one part this script cannot fetch for you.**
WHAM's SMPL/SMPLify assets and MuscleMimic's SMPL-H/MANO assets require their
respective registrations/licenses. Credentials are never stored by Gym Buddy.

After those assets are installed, start a WSL shell and run:

```bash
source ~/gym-buddy-motion-stack/gym-buddy-motion.env
```

### One command per public video

```bash
uv run gym-buddy-video-motion \
  --url "https://www.youtube.com/watch?v=..." \
  --exercise incline_smith_press
```

Local video is also supported:

```bash
uv run gym-buddy-video-motion \
  --video ./videos/incline_smith_press.mp4 \
  --exercise incline_smith_press
```

Canonical output:

```text
motions/myofullbody/incline_smith_press.npz
```

Then inspect it in the scientific viewer:

```bash
uv run gym-buddy-sim --exercise incline_smith_press
```

Use `--dry-run` to print the WHAM and MuscleMimic commands without executing
them. Only download/process public videos when you have the right to do so and
in accordance with the source platform's terms.
