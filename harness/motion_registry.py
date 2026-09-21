from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .motion_sources import MotionTrajectory, load_musclemimic_npz


CANONICAL_EXERCISES = (
    "incline_db_press",
    "flat_db_press",
    "barbell_bench",
    "incline_smith_press",
    "seated_ohp",
    "smith_squat",
    "leg_press",
    "lateral_raise",
    "lat_pulldown",
    "chest_supported_t_row",
)


@dataclass(frozen=True)
class MotionSpec:
    exercise_id: str
    filename: str
    source_pipeline: str
    notes: str = ""


MOTION_SPECS = {
    exercise_id: MotionSpec(
        exercise_id=exercise_id,
        filename=f"{exercise_id}.npz",
        source_pipeline="real-video/SMPL-H -> MuscleMimic GMR -> MyoFullBody qpos",
        notes="No hand-authored fallback is allowed for the scientific backend.",
    )
    for exercise_id in CANONICAL_EXERCISES
}


def motion_root(value: str | os.PathLike[str] | None = None) -> Path:
    root = value or os.environ.get("GYM_BUDDY_MOTION_ROOT") or "motions/myofullbody"
    return Path(root).expanduser()


def motion_path(exercise_id: str, root: str | os.PathLike[str] | None = None) -> Path:
    try:
        spec = MOTION_SPECS[exercise_id]
    except KeyError as exc:
        raise KeyError(f"unknown canonical exercise: {exercise_id}") from exc
    return motion_root(root) / spec.filename


def load_exercise_motion(
    exercise_id: str,
    *,
    root: str | os.PathLike[str] | None = None,
    trajectory_index: int = 0,
) -> MotionTrajectory:
    path = motion_path(exercise_id, root)
    if not path.exists():
        raise FileNotFoundError(
            f"motion for {exercise_id} not found: {path}. "
            "Populate it with a retargeted MyoFullBody trajectory."
        )
    return load_musclemimic_npz(path, trajectory_index=trajectory_index)


def motion_status(root: str | os.PathLike[str] | None = None) -> list[dict]:
    base = motion_root(root)
    return [
        {
            "exercise_id": exercise_id,
            "path": str(base / spec.filename),
            "available": (base / spec.filename).exists(),
            "source_pipeline": spec.source_pipeline,
        }
        for exercise_id, spec in MOTION_SPECS.items()
    ]
