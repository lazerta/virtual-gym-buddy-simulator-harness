# Virtual Gym Buddy Simulator Harness

Python-first simulation and validation harness for Gym Buddy.

**Canonical product/spec authority:** Google Drive docs `00–07`  
https://drive.google.com/drive/folders/1KcxnGbShccsJgoz8A2GU-2IkB85EiWxL

> This repository is public. Never commit raw licensed datasets, personal media/data, credentials, or generated private artifacts.

## Setup

```bash
git clone https://github.com/lazerta/virtual-gym-buddy-simulator-harness.git
cd virtual-gym-buddy-simulator-harness
uv sync --extra dev
uv run pytest
uv run python -m harness.doctor
uv run python -m harness.cli validate-profiles --quick --fps 10
```

## Codex handoff prompt

> Pull the latest `main`, read this README and canonical Google Drive docs `00–07`, set up the project with `uv`, then run `pytest`, `python -m harness.doctor`, and `python -m harness.cli validate-profiles --quick --fps 10`. Fix the correct layer; do not weaken tests just to get green. Keep raw Fit3D/other datasets, personal media, secrets, caches, virtualenvs, generated videos, CSV dumps, and large artifacts out of Git. If `FIT3D_ROOT` is absent, report real Fit3D replay as `NOT_RUN`.

## Core rules

- External/anonymous quick validation passes before Shawn-like validation runs.
- `PrimarySubjectLock` and `TrackingQualityGate` are separate.
- Multiple people do not automatically pause analysis.
- Rep detection is temporal and uses reusable movement primitives.
- Form analysis is separate from rep detection.
- Oracle truth never enters detector inputs.
- Raw licensed datasets and personal data never enter this repository.

## Fit3D

Fit3D is local-only:

```powershell
$env:FIT3D_ROOT="D:\Datasets\Fit3D"
```

3D joints are detector input; repetition annotations are independent oracle truth only.
