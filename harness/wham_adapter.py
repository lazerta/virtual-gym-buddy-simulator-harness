from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np


class WhamFormatError(ValueError):
    pass


@dataclass(frozen=True)
class WhamTrack:
    track_id: str
    frame_ids: np.ndarray
    pose_camera: np.ndarray
    trans_camera: np.ndarray
    betas: np.ndarray
    pose_world: np.ndarray | None = None
    trans_world: np.ndarray | None = None

    @property
    def frames(self) -> int:
        return int(self.frame_ids.shape[0])


@dataclass(frozen=True)
class WhamConversionResult:
    output_path: Path
    metadata_path: Path
    track_id: str
    coordinate_space: str
    source_frames: int
    exported_frames: int
    start_frame: int
    end_frame: int
    fps: float


def _array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    arr = np.asarray(value)
    if ndim is not None and arr.ndim != ndim:
        raise WhamFormatError(f"{name} must be {ndim}D, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise WhamFormatError(f"{name} contains non-finite values")
    return arr


def _normalize_track(track_id: Any, raw: dict[str, Any]) -> WhamTrack:
    required = ("pose", "trans", "betas", "frame_ids")
    missing = [k for k in required if k not in raw]
    if missing:
        raise WhamFormatError(f"WHAM track {track_id!r} missing fields: {missing}")

    pose = _array(raw["pose"], name="pose", ndim=2).astype(np.float32)
    trans = _array(raw["trans"], name="trans", ndim=2).astype(np.float32)
    frame_ids = np.asarray(raw["frame_ids"], dtype=np.int64).reshape(-1)
    betas = _array(raw["betas"], name="betas").astype(np.float32)

    if pose.shape[1] != 72:
        raise WhamFormatError(f"WHAM pose must be [T,72], got {pose.shape}")
    if trans.shape != (pose.shape[0], 3):
        raise WhamFormatError(f"WHAM trans must be [T,3], got {trans.shape}")
    if frame_ids.shape != (pose.shape[0],):
        raise WhamFormatError(
            f"frame_ids length {len(frame_ids)} != pose frames {pose.shape[0]}"
        )

    order = np.argsort(frame_ids, kind="stable")
    frame_ids = frame_ids[order]
    if np.any(np.diff(frame_ids) <= 0):
        raise WhamFormatError(f"frame_ids must be unique for track {track_id!r}")

    pose = pose[order]
    trans = trans[order]

    pose_world = None
    if raw.get("pose_world") is not None:
        pose_world = _array(raw["pose_world"], name="pose_world", ndim=2).astype(np.float32)
        if pose_world.shape != pose.shape:
            raise WhamFormatError(
                f"pose_world shape {pose_world.shape} != pose shape {pose.shape}"
            )
        pose_world = pose_world[order]

    trans_world = None
    if raw.get("trans_world") is not None:
        trans_world = _array(raw["trans_world"], name="trans_world", ndim=2).astype(np.float32)
        if trans_world.shape != trans.shape:
            raise WhamFormatError(
                f"trans_world shape {trans_world.shape} != trans shape {trans.shape}"
            )
        trans_world = trans_world[order]

    if betas.ndim == 2 and betas.shape[0] == len(order):
        betas = betas[order]

    return WhamTrack(
        track_id=str(track_id),
        frame_ids=frame_ids,
        pose_camera=pose,
        trans_camera=trans,
        betas=betas,
        pose_world=pose_world,
        trans_world=trans_world,
    )


def load_wham_output(path: str | Path) -> list[WhamTrack]:
    payload = joblib.load(Path(path))
    if not isinstance(payload, dict) or not payload:
        raise WhamFormatError("WHAM output must be a non-empty dict of tracks")

    tracks: list[WhamTrack] = []
    for track_id, raw in payload.items():
        if not isinstance(raw, dict):
            raise WhamFormatError(f"WHAM track {track_id!r} is not a dict")
        tracks.append(_normalize_track(track_id, raw))
    return tracks


def longest_contiguous_indices(frame_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        return np.empty(0, dtype=np.int64)

    breaks = np.flatnonzero(np.diff(ids) != 1) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(ids)]))
    lengths = ends - starts
    best = int(np.argmax(lengths))
    return np.arange(starts[best], ends[best], dtype=np.int64)


def select_wham_track(
    tracks: list[WhamTrack],
    track_id: str | int | None = None,
) -> WhamTrack:
    if not tracks:
        raise WhamFormatError("no WHAM tracks")

    if track_id is not None:
        wanted = str(track_id)
        for track in tracks:
            if track.track_id == wanted:
                return track
        raise KeyError(f"WHAM track {track_id!r} not found")

    def score(track: WhamTrack) -> tuple[int, int]:
        return (len(longest_contiguous_indices(track.frame_ids)), track.frames)

    return max(tracks, key=score)


def read_video_fps(path: str | Path) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid FPS for video {path}: {fps}")
    return fps


def _stable_betas(betas: np.ndarray, indices: np.ndarray) -> np.ndarray:
    x = np.asarray(betas, dtype=np.float32)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        if x.shape[0] >= int(indices.max()) + 1:
            return np.median(x[indices], axis=0).astype(np.float32)
        return np.median(x, axis=0).astype(np.float32)
    return x.reshape(-1).astype(np.float32)


def wham_track_to_amass(
    track: WhamTrack,
    *,
    fps: float,
    coordinate_space: str = "world",
    contiguous_only: bool = True,
    gender: str = "neutral",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if coordinate_space not in {"world", "camera"}:
        raise ValueError("coordinate_space must be 'world' or 'camera'")

    use_world = (
        coordinate_space == "world"
        and track.pose_world is not None
        and track.trans_world is not None
    )

    pose = track.pose_world if use_world else track.pose_camera
    trans = track.trans_world if use_world else track.trans_camera
    used_space = "world" if use_world else "camera"

    indices = (
        longest_contiguous_indices(track.frame_ids)
        if contiguous_only
        else np.arange(track.frames, dtype=np.int64)
    )
    if indices.size == 0:
        raise WhamFormatError("selected WHAM track has no frames")

    pose = np.asarray(pose[indices], dtype=np.float32)
    trans = np.asarray(trans[indices], dtype=np.float32)
    frame_ids = track.frame_ids[indices]

    # WHAM stores SMPL root + 23 body joints = 72 axis-angle dims.
    # MuscleMimic's own AMASS exporter treats 66 dims as the body portion,
    # explicitly zeros the terminal two SMPL hand/wrist slots, then pads to
    # the 156D SMPL-H representation.
    pose_aa_72 = np.zeros((pose.shape[0], 72), dtype=np.float32)
    pose_aa_72[:, :66] = pose[:, :66]

    poses = np.zeros((pose.shape[0], 156), dtype=np.float32)
    poses[:, :72] = pose_aa_72

    stable_betas = _stable_betas(track.betas, indices)

    motion = {
        "poses": poses,
        "trans": trans,
        "betas": stable_betas,
        "gender": np.asarray(str(gender)),
        "mocap_framerate": np.asarray(float(fps), dtype=np.float32),
    }

    metadata = {
        "source": "WHAM",
        "track_id": track.track_id,
        "coordinate_space_requested": coordinate_space,
        "coordinate_space_used": used_space,
        "source_frames": track.frames,
        "exported_frames": int(len(indices)),
        "start_frame": int(frame_ids[0]),
        "end_frame": int(frame_ids[-1]),
        "fps": float(fps),
        "pose_conversion": (
            "WHAM SMPL72 -> keep [:66] -> zero [66:72] -> pad SMPL-H156"
        ),
        "contiguous_only": bool(contiguous_only),
    }
    return motion, metadata


def save_amass_motion(
    motion: dict[str, np.ndarray],
    output_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **motion)

    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata or {}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output, metadata_path


def convert_wham_to_amass(
    wham_pkl: str | Path,
    output_path: str | Path,
    *,
    video_path: str | Path | None = None,
    fps: float | None = None,
    track_id: str | int | None = None,
    coordinate_space: str = "world",
    contiguous_only: bool = True,
    gender: str = "neutral",
) -> WhamConversionResult:
    if fps is None:
        if video_path is None:
            raise ValueError("provide fps=... or video_path=...")
        fps = read_video_fps(video_path)

    track = select_wham_track(load_wham_output(wham_pkl), track_id=track_id)

    motion, metadata = wham_track_to_amass(
        track,
        fps=float(fps),
        coordinate_space=coordinate_space,
        contiguous_only=contiguous_only,
        gender=gender,
    )
    output, metadata_path = save_amass_motion(
        motion,
        output_path,
        metadata=metadata,
    )

    return WhamConversionResult(
        output_path=output,
        metadata_path=metadata_path,
        track_id=track.track_id,
        coordinate_space=metadata["coordinate_space_used"],
        source_frames=int(metadata["source_frames"]),
        exported_frames=int(metadata["exported_frames"]),
        start_frame=int(metadata["start_frame"]),
        end_frame=int(metadata["end_frame"]),
        fps=float(metadata["fps"]),
    )
