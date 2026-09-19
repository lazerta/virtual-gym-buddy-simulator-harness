from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .movement_primitives import MovementPrimitive, infer_primitive_composition


FIT3D_FPS = 50.0


@dataclass(frozen=True)
class RepInterval:
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError(f"Invalid repetition interval: {self}")

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass(frozen=True)
class Fit3DRecord:
    split: str
    subject: str
    action: str
    joints_path: Path
    rep_ann_path: Path
    camera_names: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return f"{self.split}:{self.subject}:{self.action}"


@dataclass
class MovementPrimitiveSequence:
    source: str
    split: str
    subject: str
    action: str
    fps: float
    joints_3d: np.ndarray
    rep_intervals: tuple[RepInterval, ...]
    primitives: tuple[MovementPrimitive, ...]
    camera_names: tuple[str, ...] = ()

    @property
    def frame_count(self) -> int:
        return int(self.joints_3d.shape[0])

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps

    def oracle_rep_index(self, frame_index: int) -> int | None:
        for i, interval in enumerate(self.rep_intervals):
            if interval.start_frame <= frame_index <= interval.end_frame:
                return i
        return None

    def oracle_phase_t(self, frame_index: int) -> float | None:
        idx = self.oracle_rep_index(frame_index)
        if idx is None:
            return None
        interval = self.rep_intervals[idx]
        if interval.frame_count <= 1:
            return 0.0
        return (frame_index - interval.start_frame) / (interval.frame_count - 1)

    def rep_slice(self, rep_index: int) -> np.ndarray:
        interval = self.rep_intervals[rep_index]
        return self.joints_3d[interval.start_frame : interval.end_frame + 1]

    def centered_scaled_joints(self) -> np.ndarray:
        x = np.asarray(self.joints_3d, dtype=np.float64)
        center = np.nanmean(x, axis=1, keepdims=True)
        centered = x - center
        radius = np.nanmedian(np.linalg.norm(centered, axis=-1), axis=1)
        scale = float(np.nanmedian(radius[radius > 1e-9])) if np.any(radius > 1e-9) else 1.0
        return centered / max(scale, 1e-9)


class Fit3DFormatError(ValueError):
    pass


def _as_int_frame(value: Any, fps: float, *, unit: str) -> int:
    if isinstance(value, str):
        value = float(value)
    if unit == "seconds":
        return int(round(float(value) * fps))
    return int(round(float(value)))


def _pair_from_obj(obj: Any, fps: float, *, unit: str) -> tuple[int, int] | None:
    if isinstance(obj, (list, tuple)) and len(obj) >= 2 and all(isinstance(x, (int, float, str)) for x in obj[:2]):
        return _as_int_frame(obj[0], fps, unit=unit), _as_int_frame(obj[1], fps, unit=unit)
    if isinstance(obj, dict):
        key_pairs = (
            ("start_frame", "end_frame"),
            ("start", "end"),
            ("begin", "end"),
            ("frame_start", "frame_end"),
        )
        for a, b in key_pairs:
            if a in obj and b in obj:
                return _as_int_frame(obj[a], fps, unit=unit), _as_int_frame(obj[b], fps, unit=unit)
    return None


def parse_rep_intervals(raw: Any, frame_count: int, *, fps: float = FIT3D_FPS, unit: str = "frames") -> tuple[RepInterval, ...]:
    if unit not in {"frames", "seconds"}:
        raise ValueError("unit must be 'frames' or 'seconds'")

    items: Iterable[Any]
    if raw is None:
        return ()
    if isinstance(raw, dict):
        direct = _pair_from_obj(raw, fps, unit=unit)
        if direct is not None:
            items = [raw]
        else:
            items = raw.values()
    elif isinstance(raw, (list, tuple)):
        items = raw
    else:
        raise Fit3DFormatError(f"Unsupported repetition annotation type: {type(raw).__name__}")

    intervals: list[RepInterval] = []
    for item in items:
        pair = _pair_from_obj(item, fps, unit=unit)
        if pair is None:
            continue
        start, end = pair
        if end == frame_count:
            end -= 1
        if not (0 <= start <= end < frame_count):
            raise Fit3DFormatError(
                f"Rep interval {(start, end)} outside motion of {frame_count} frames"
            )
        intervals.append(RepInterval(start, end))

    intervals.sort(key=lambda x: (x.start_frame, x.end_frame))
    for prev, cur in zip(intervals, intervals[1:]):
        if cur.start_frame <= prev.end_frame:
            raise Fit3DFormatError(f"Overlapping rep intervals: {prev} and {cur}")
    return tuple(intervals)


class Fit3DLocalAdapter:
    def __init__(self, root: str | os.PathLike[str] | None = None, *, fps: float = FIT3D_FPS, rep_unit: str = "frames"):
        root_value = root or os.environ.get("FIT3D_ROOT")
        if not root_value:
            raise ValueError("Fit3D root not provided. Pass root=... or set FIT3D_ROOT.")
        p = Path(root_value).expanduser().resolve()
        if not (p / "train").exists() and (p / "fit3d" / "train").exists():
            p = p / "fit3d"
        self.root = p
        self.fps = float(fps)
        self.rep_unit = rep_unit

    def validate_root(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "exists": self.root.exists(),
            "train_exists": (self.root / "train").exists(),
            "test_exists": (self.root / "test").exists(),
            "info_candidates": [
                str(x) for x in (self.root / "info.json", self.root / "fit3d_info.json") if x.exists()
            ],
        }

    def index(self, splits: tuple[str, ...] = ("train", "test")) -> list[Fit3DRecord]:
        records: list[Fit3DRecord] = []
        for split in splits:
            split_dir = self.root / split
            if not split_dir.exists():
                continue
            for subject_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                joints_dir = subject_dir / "joints3d_25"
                if not joints_dir.exists():
                    continue
                rep_ann = subject_dir / "rep_ann.json"
                for joints_path in sorted(joints_dir.glob("*.json")):
                    action = joints_path.stem
                    camera_root = subject_dir / "camera_parameters"
                    cams: list[str] = []
                    if camera_root.exists():
                        for camera_dir in sorted(p for p in camera_root.iterdir() if p.is_dir()):
                            if (camera_dir / f"{action}.json").exists():
                                cams.append(camera_dir.name)
                    records.append(
                        Fit3DRecord(
                            split=split,
                            subject=subject_dir.name,
                            action=action,
                            joints_path=joints_path,
                            rep_ann_path=rep_ann,
                            camera_names=tuple(cams),
                        )
                    )
        return records

    def load(self, record: Fit3DRecord) -> MovementPrimitiveSequence:
        with record.joints_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "joints3d_25" not in payload:
            raise Fit3DFormatError(f"Missing joints3d_25 in {record.joints_path}")
        joints = np.asarray(payload["joints3d_25"], dtype=np.float64)
        if joints.ndim != 3 or joints.shape[1:] != (25, 3):
            raise Fit3DFormatError(
                f"Expected joints shape [T,25,3], got {joints.shape} in {record.joints_path}"
            )
        if not np.isfinite(joints).all():
            raise Fit3DFormatError(f"Non-finite 3D joints in {record.joints_path}")

        ann_raw = None
        if record.rep_ann_path.exists():
            with record.rep_ann_path.open("r", encoding="utf-8") as f:
                all_ann = json.load(f)
            ann_raw = all_ann.get(record.action)
        reps = parse_rep_intervals(ann_raw, len(joints), fps=self.fps, unit=self.rep_unit)
        comp = infer_primitive_composition(record.action)
        return MovementPrimitiveSequence(
            source="fit3d",
            split=record.split,
            subject=record.subject,
            action=record.action,
            fps=self.fps,
            joints_3d=joints,
            rep_intervals=reps,
            primitives=comp.primitives,
            camera_names=record.camera_names,
        )

    def summary(self, *, load_records: bool = False, limit: int | None = None) -> dict[str, Any]:
        records = self.index()
        if limit is not None:
            records = records[:limit]
        actions = sorted({r.action for r in records})
        subjects = sorted({r.subject for r in records})
        out: dict[str, Any] = {
            **self.validate_root(),
            "record_count": len(records),
            "subject_count": len(subjects),
            "action_count": len(actions),
            "subjects": subjects,
            "actions": actions,
            "primitive_compositions": {
                a: [p.value for p in infer_primitive_composition(a).primitives] for a in actions
            },
        }
        if load_records:
            loaded = [self.load(r) for r in records]
            out.update(
                loaded_records=len(loaded),
                total_frames=sum(x.frame_count for x in loaded),
                total_reps=sum(len(x.rep_intervals) for x in loaded),
            )
        return out

    def index_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.index():
            rows.append(
                {
                    "record_id": r.id,
                    "split": r.split,
                    "subject": r.subject,
                    "action": r.action,
                    "primitives": "+".join(p.value for p in infer_primitive_composition(r.action).primitives),
                    "camera_count": len(r.camera_names),
                    "joints_path": str(r.joints_path),
                    "rep_ann_path": str(r.rep_ann_path),
                }
            )
        return pd.DataFrame(rows)


def principal_motion_depth(sequence: MovementPrimitiveSequence, *, smooth_frames: int = 5) -> np.ndarray:
    x = sequence.centered_scaled_joints().reshape(sequence.frame_count, -1)
    if len(x) < 3:
        return np.zeros(len(x), dtype=np.float64)
    x = x - np.nanmedian(x, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    score = x @ vh[0]
    if smooth_frames > 1 and len(score) >= smooth_frames:
        k = int(max(1, smooth_frames))
        kernel = np.ones(k, dtype=np.float64) / k
        pad_l = k // 2
        pad_r = k - 1 - pad_l
        score = np.convolve(np.pad(score, (pad_l, pad_r), mode="edge"), kernel, mode="valid")

    lo, hi = np.nanpercentile(score, [2.0, 98.0])
    if hi - lo < 1e-9:
        return np.zeros(len(score), dtype=np.float64)
    d = np.clip((score - lo) / (hi - lo), 0.0, 1.0)
    head = float(np.nanmedian(d[: max(2, min(len(d), int(round(sequence.fps * 0.25))))]))
    if head > 0.5:
        d = 1.0 - d
    return d.astype(np.float64)


def as_rep_detector_frames(sequence: MovementPrimitiveSequence, *, smooth_frames: int = 5):
    from .rep_detection import MovementFrame

    depth = principal_motion_depth(sequence, smooth_frames=smooth_frames)
    return [
        MovementFrame(frame_index=i, t_s=i / sequence.fps, depth=float(depth[i]), valid=True, assistance_evidence=0.0)
        for i in range(sequence.frame_count)
    ]


def evaluate_rep_replay(sequence: MovementPrimitiveSequence) -> dict[str, Any]:
    from .rep_detection import detect_reps

    simple = len(sequence.primitives) == 1 and sequence.primitives[0] != MovementPrimitive.UNKNOWN
    if not simple:
        return {
            "subject": sequence.subject,
            "action": sequence.action,
            "primitives": "+".join(p.value for p in sequence.primitives),
            "expected_reps": len(sequence.rep_intervals),
            "detected_reps": None,
            "passed": None,
            "status": "SKIPPED_COMPOUND_OR_UNKNOWN",
        }
    events = detect_reps(as_rep_detector_frames(sequence))
    return {
        "subject": sequence.subject,
        "action": sequence.action,
        "primitives": "+".join(p.value for p in sequence.primitives),
        "expected_reps": len(sequence.rep_intervals),
        "detected_reps": len(events),
        "passed": len(events) == len(sequence.rep_intervals),
        "status": "RUN",
    }


def run_fit3d_rep_replay(adapter: Fit3DLocalAdapter, *, limit: int | None = None) -> pd.DataFrame:
    rows = []
    records = adapter.index()
    if limit is not None:
        records = records[:limit]
    for record in records:
        rows.append(evaluate_rep_replay(adapter.load(record)))
    return pd.DataFrame(rows)
