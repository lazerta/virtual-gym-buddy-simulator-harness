# Fit3D Local Setup

Fit3D is a licensed external dataset. **Do not commit or redistribute raw Fit3D files in this repository.**

Official sources:

- Dataset: https://fit3d.imar.ro/
- Download: https://fit3d.imar.ro/download
- Official tooling: https://github.com/sminchisescu-research/imar_vision_datasets_tools

The current license is intended for a single user for non-commercial scientific/research purposes. Accept the current upstream license yourself before downloading.

## Local layout

The adapter expects the public Fit3D tooling layout:

```text
fit3d/
  train/
    sXX/
      joints3d_25/
        <action>.json
      rep_ann.json
      camera_parameters/
        <camera>/
          <action>.json
  test/
    ...
```

Set:

```bash
export FIT3D_ROOT=/absolute/path/to/fit3d
```

PowerShell:

```powershell
$env:FIT3D_ROOT = "D:\datasets\fit3d"
```

## Verify

```bash
gym-buddy-harness fit3d-check
gym-buddy-harness fit3d-check --load --limit 10
gym-buddy-harness fit3d-index --out artifacts/fit3d_index.csv
```

## Rep replay

For simple one-primitive exercises, the harness can derive a cyclic movement signal from the 3D joints **without reading repetition intervals into the detector**:

```bash
gym-buddy-harness fit3d-rep-replay --limit 25
```

Data separation is intentional:

```text
joints3d_25 -> motion/primitive signal -> RepDetector
rep_ann     -> independent oracle only
```

Compound/unknown actions are skipped by the single-cycle replay and will later be decomposed into ordered movement primitives.

## Do not commit raw data

Do not add Fit3D archives, images, skeleton JSON, GHUM/SMPL-X data, or rep annotation files to this repository or shared artifact storage.
