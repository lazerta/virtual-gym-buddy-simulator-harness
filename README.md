# Virtual Gym Buddy Simulator Harness

Python-first test harness for Gym Buddy: commercial-gym simulation, PrimarySubjectLock, temporal rep detection, trainer/spotter behavior, movement primitives, and Fit3D local replay.

## Local setup

```bash
git clone https://github.com/lazerta/virtual-gym-buddy-simulator-harness.git
cd virtual-gym-buddy-simulator-harness
uv sync
uv run pytest
uv run python -m harness.cli validate-profiles --quick --fps 10
```

Optional:

```bash
uv run python -m harness.cli doctor
```

## Fit3D

Fit3D raw data stays local and is never committed.

```powershell
$env:FIT3D_ROOT="D:\Datasets\Fit3D"
```

The harness uses Fit3D 3D joints as detector input and repetition annotations only as the independent oracle.

## Codex setup prompt

> Clone this repo and set up a clean local Python environment with `uv`. Install only dependencies declared by the project. Run `pytest`, `harness.cli doctor`, and `validate-profiles --quick --fps 10`. Fix environment or harness issues at the correct layer; do not loosen tests just to make them pass. Keep raw Fit3D/other datasets, personal media, secrets, caches, and generated artifacts out of Git. If `FIT3D_ROOT` is absent, report real Fit3D replay as `NOT_RUN`, not PASS.

## Core rules

- External/anonymous profile passes before Shawn-like profile runs.
- Multiple people do not automatically pause analysis; lock the primary subject and ignore others unless identity/observability becomes ambiguous.
- Rep detection is temporal and uses reusable movement primitives, not one detector per exercise variant.
- Reality-aware oracle: `MUST / MAY / MUST_NOT`; safety violations remain zero-tolerance.
- Raw licensed datasets and personal data never enter this repository.
