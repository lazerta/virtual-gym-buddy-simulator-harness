from __future__ import annotations

from dataclasses import dataclass
import csv
import os
from pathlib import Path
from typing import Iterable

import numpy as np


MMFIT_FPS = 30.0

MMFIT_ACTIONS = (
    "squats",
    "lunges",
    "bicep_curls",
    "situps",
    "pushups",
    "tricep_extensions",
    "dumbbell_rows",
    "jumping_jacks",
    "dumbbell_shoulder_press",
    "lateral_shoulder_raises",
)

# These are movement-family bridges, not claims of exact equipment equivalence.
MMFIT_CANONICAL_BRIDGES = {
    "squats": ("smith_squat", "squat-family"),
    "dumbbell_shoulder_press": ("seated_ohp", "overhead-press-family"),
    "dumbbell_rows": ("chest_supported_t_row", "row-family"),
    "lateral_shoulder_raises": ("lateral_raise", "lateral-raise-family"),
}


class MMFitFormatError(ValueError):
    pass


@dataclass(frozen=True)
class MMFitSet:
    start_frame: int
    end_frame: int
    rep_count: int
    action: str

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError(f"invalid MM-Fit set interval: {self}")
        if self.rep_count < 0:
            raise ValueError("rep_count must be >= 0")


@dataclass(frozen=True)
class MMFitWorkoutRecord:
    workout_id: str
    root: Path
    pose_3d_path: Path
    labels_path: Path

    @property
    def id(self) -> str:
        return self.workout_id


@dataclass
class MMFitWorkout:
    workout_id: str
    fps: float
    frame_ids: np.ndarray
    joints_3d: np.ndarray
    sets: tuple[MMFitSet, ...]

    def __post_init__(self) -> None:
        if self.joints_3d.ndim != 3 or self.joints_3d.shape[1:] != (16, 3):
            raise MMFitFormatError(
                f"expected joints_3d [T,16,3], got {self.joints_3d.shape}"
            )
        if self.frame_ids.shape != (self.joints_3d.shape[0],):
            raise MMFitFormatError("frame_ids length must equal joint frame count")

    @property
    def frame_count(self) -> int:
        return int(self.joints_3d.shape[0])

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(sorted({x.action for x in self.sets}))

    def action_sets(self, action: str) -> tuple[MMFitSet, ...]:
        return tuple(x for x in self.sets if x.action == action)

    def set_slice(self, item: MMFitSet) -> np.ndarray:
        frame = self.frame_ids
        mask = (frame >= item.start_frame) & (frame <= item.end_frame)
        return self.joints_3d[mask]


def _find_one(workout_dir: Path, token: str, suffix: str) -> Path | None:
    matches = sorted(
        p for p in workout_dir.iterdir()
        if p.is_file() and token in p.name and p.suffix.lower() == suffix
    )
    return matches[0] if matches else None


def _parse_labels(path: Path) -> tuple[MMFitSet, ...]:
    out: list[MMFitSet] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if len(row) < 4:
                raise MMFitFormatError(f"label row must have 4 columns in {path}: {row}")
            item = MMFitSet(
                start_frame=int(row[0]),
                end_frame=int(row[1]),
                rep_count=int(row[2]),
                action=row[3].strip(),
            )
            out.append(item)
    return tuple(out)


def _load_pose3d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(np.load(path), dtype=np.float64)

    # Official MM-Fit starter code uses pose_3d[0, frame, 0] as frame id and
    # pose_3d[:, frame, 1:] as 16 xyz joints.
    if raw.ndim != 3 or raw.shape[0] != 3 or raw.shape[2] < 17:
        raise MMFitFormatError(
            f"expected official MM-Fit pose layout [3,T,>=17], got {raw.shape}"
        )

    frame_ids = raw[0, :, 0].astype(np.int64)
    joints = np.transpose(raw[:, :, 1:17], (1, 2, 0))
    if not np.isfinite(joints).all():
        raise MMFitFormatError(f"non-finite pose values in {path}")
    return frame_ids, joints


class MMFitLocalAdapter:
    """Read official MM-Fit pose_3d + labels without torch or upstream code."""

    def __init__(self, root: str | os.PathLike[str] | None = None, *, fps: float = MMFIT_FPS):
        value = root or os.environ.get("MMFIT_ROOT")
        if not value:
            raise ValueError("MM-Fit root not provided. Pass root=... or set MMFIT_ROOT.")
        self.root = Path(value).expanduser().resolve()
        self.fps = float(fps)

    def index(self) -> list[MMFitWorkoutRecord]:
        if not self.root.exists():
            return []
        records: list[MMFitWorkoutRecord] = []
        for workout_dir in sorted(p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("w")):
            pose = _find_one(workout_dir, "pose_3d", ".npy")
            labels = _find_one(workout_dir, "labels", ".csv")
            if pose is None or labels is None:
                continue
            records.append(
                MMFitWorkoutRecord(
                    workout_id=workout_dir.name,
                    root=workout_dir,
                    pose_3d_path=pose,
                    labels_path=labels,
                )
            )
        return records

    def load(self, record: MMFitWorkoutRecord) -> MMFitWorkout:
        frames, joints = _load_pose3d(record.pose_3d_path)
        sets = _parse_labels(record.labels_path)
        return MMFitWorkout(
            workout_id=record.workout_id,
            fps=self.fps,
            frame_ids=frames,
            joints_3d=joints,
            sets=sets,
        )

    def summary(self, *, load_records: bool = False) -> dict:
        records = self.index()
        out = {
            "root": str(self.root),
            "exists": self.root.exists(),
            "record_count": len(records),
            "workouts": [r.workout_id for r in records],
            "default_public_baseline": True,
            "fit3d_required": False,
        }
        if load_records:
            loaded = [self.load(r) for r in records]
            out.update(
                total_frames=sum(x.frame_count for x in loaded),
                actions=sorted({a for x in loaded for a in x.actions}),
                total_sets=sum(len(x.sets) for x in loaded),
                total_annotated_reps=sum(s.rep_count for x in loaded for s in x.sets),
            )
        return out

    def canonical_bridge_coverage(self) -> list[dict]:
        available_actions = set()
        for record in self.index():
            try:
                available_actions.update(self.load(record).actions)
            except Exception:
                continue

        rows = []
        for action, (canonical, family) in MMFIT_CANONICAL_BRIDGES.items():
            rows.append(
                {
                    "mmfit_action": action,
                    "canonical_exercise": canonical,
                    "movement_family": family,
                    "available": action in available_actions,
                    "exact_equipment_match": False,
                }
            )
        return rows
