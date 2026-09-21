# Scientific 3D Simulation Backend

## Authority boundary

This backend owns physical and biomechanical simulation.

```text
real motion / retargeted trajectory
            |
            v
      MyoFullBody state
            |
            v
          MuJoCo
  human + equipment + contacts
            |
      simulator state
       /           \
      v             v
 ground truth    Three.js
                rendering only
```

### Reused open-source components

- **MuJoCo** — dynamics/contact engine.
- **MyoSim / MyoFullBody** — anatomically detailed musculoskeletal model.
- **MuscleMimic** — upstream reference implementation for SMPL-H -> MyoFullBody retargeting and muscle-driven imitation. This repository consumes retargeted trajectories rather than copying its implementation.
- **CMU MoCap / real-video pose pipelines** — candidate motion sources.
- **Three.js** — RGB visualization only.

## What this repository implements

Only glue and Gym Buddy-specific state contracts:

- MyoFullBody model loading.
- Gym-equipment physics proxy attachment.
- simulator step/reset/state API.
- retargeted trajectory adapter.
- JSONL state bridge.
- state export: qpos, qvel, bodies, sites, contacts.
- tests ensuring the upstream model actually compiles and steps.

## What this repository should NOT implement

- a bespoke full-body physics engine;
- custom musculoskeletal anatomy;
- hand-authored full exercise animation as the primary motion source;
- a duplicate SMPL-H retargeting stack;
- Three.js as the authoritative motion or physics engine.

## Motion source contract

The canonical interchange format is a MyoFullBody `qpos` trajectory:

```python
MotionTrajectory(
    qpos=np.ndarray[frames, model.nq],
    fps=float,
    source=str,
)
```

MuscleMimic/GMR, offline video retargeting, CMU-derived motion, or future sources all converge to that same contract.

## Rendering contract

Three.js should consume exported state and render it. It must not silently invent a second motion solution. Any visual avatar retargeting is a presentation transform derived from the authoritative MuJoCo state.
