from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


FRAME_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    timestamp_us: int
    width: int
    height: int
    mime_type: str
    image_path: str
    ground_truth_path: str

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id must be >= 0")
        if self.timestamp_us < 0:
            raise ValueError("timestamp_us must be >= 0")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("frame dimensions must be positive")
        if self.mime_type not in {"image/jpeg", "image/png"}:
            raise ValueError(f"unsupported frame mime_type: {self.mime_type}")
        _validate_relative_path(self.image_path)
        _validate_relative_path(self.ground_truth_path)


@dataclass(frozen=True)
class FrameSessionManifest:
    session_id: str
    exercise_id: str
    fps: float
    width: int
    height: int
    frames: tuple[FrameRecord, ...]
    schema_version: int = FRAME_SCHEMA_VERSION
    source: str = "mujoco"

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id is required")
        if not self.exercise_id:
            raise ValueError("exercise_id is required")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("session dimensions must be positive")
        if self.schema_version != FRAME_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported frame schema {self.schema_version}; "
                f"expected {FRAME_SCHEMA_VERSION}"
            )
        _validate_frame_sequence(self.frames, self.width, self.height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "exercise_id": self.exercise_id,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "source": self.source,
            "frame_count": len(self.frames),
            "frames": [asdict(frame) for frame in self.frames],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "FrameSessionManifest":
        frames = tuple(FrameRecord(**item) for item in payload.get("frames", ()))
        return FrameSessionManifest(
            schema_version=int(payload.get("schema_version", FRAME_SCHEMA_VERSION)),
            session_id=str(payload["session_id"]),
            exercise_id=str(payload["exercise_id"]),
            fps=float(payload["fps"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            source=str(payload.get("source", "mujoco")),
            frames=frames,
        )


@dataclass(frozen=True)
class AppFrameResult:
    session_id: str
    frame_id: int
    timestamp_us: int
    analysis: dict[str, Any]
    schema_version: int = FRAME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id is required")
        if self.frame_id < 0:
            raise ValueError("frame_id must be >= 0")
        if self.timestamp_us < 0:
            raise ValueError("timestamp_us must be >= 0")
        if self.schema_version != FRAME_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported result schema {self.schema_version}; "
                f"expected {FRAME_SCHEMA_VERSION}"
            )
        if not isinstance(self.analysis, dict):
            raise ValueError("analysis must be a JSON object")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_manifest(path: str | Path) -> FrameSessionManifest:
    p = Path(path)
    return FrameSessionManifest.from_dict(json.loads(p.read_text(encoding="utf-8")))


def save_manifest(path: str | Path, manifest: FrameSessionManifest) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(manifest.to_json() + "\n", encoding="utf-8")


def frame_by_id(manifest: FrameSessionManifest, frame_id: int) -> FrameRecord:
    if frame_id < 0 or frame_id >= len(manifest.frames):
        raise KeyError(f"frame_id {frame_id} not found")
    frame = manifest.frames[frame_id]
    if frame.frame_id != frame_id:
        raise KeyError(f"manifest frame index mismatch at {frame_id}")
    return frame


def resolve_session_path(session_root: str | Path, relative: str) -> Path:
    _validate_relative_path(relative)
    root = Path(session_root).resolve()
    candidate = (root / relative).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"path escapes session root: {relative}")
    return candidate


def validate_result_against_manifest(
    manifest: FrameSessionManifest,
    result: AppFrameResult,
) -> FrameRecord:
    if result.session_id != manifest.session_id:
        raise ValueError(
            f"result session_id={result.session_id!r} does not match "
            f"{manifest.session_id!r}"
        )
    frame = frame_by_id(manifest, result.frame_id)
    if result.timestamp_us != frame.timestamp_us:
        raise ValueError(
            f"result timestamp {result.timestamp_us} does not match "
            f"frame timestamp {frame.timestamp_us}"
        )
    return frame


def _validate_relative_path(value: str) -> None:
    p = Path(value)
    if p.is_absolute():
        raise ValueError(f"session path must be relative: {value}")
    if any(part == ".." for part in p.parts):
        raise ValueError(f"session path cannot contain '..': {value}")


def _validate_frame_sequence(
    frames: tuple[FrameRecord, ...],
    width: int,
    height: int,
) -> None:
    previous_ts = -1
    for index, frame in enumerate(frames):
        if frame.frame_id != index:
            raise ValueError(
                f"frame ids must be dense and ordered; index={index}, "
                f"frame_id={frame.frame_id}"
            )
        if frame.timestamp_us <= previous_ts:
            raise ValueError("frame timestamps must be strictly increasing")
        if frame.width != width or frame.height != height:
            raise ValueError("frame dimensions must match session dimensions")
        previous_ts = frame.timestamp_us
