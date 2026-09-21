from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MotionTrajectory:
    qpos: np.ndarray
    fps: float
    source: str

    def __post_init__(self) -> None:
        if self.qpos.ndim != 2:
            raise ValueError("qpos must be [frames, nq]")
        if self.qpos.shape[0] < 1:
            raise ValueError("trajectory must contain at least one frame")
        if self.fps <= 0:
            raise ValueError("fps must be positive")

    @property
    def frames(self) -> int:
        return int(self.qpos.shape[0])

    def frame(self, index: int) -> np.ndarray:
        return self.qpos[max(0, min(self.frames - 1, int(index)))]

    def phase(self, phase: float) -> np.ndarray:
        p = float(np.clip(phase, 0.0, 1.0))
        u = p * (self.frames - 1)
        i = int(np.floor(u))
        j = min(self.frames - 1, i + 1)
        a = u - i
        return (1.0 - a) * self.qpos[i] + a * self.qpos[j]


def load_npz_motion(path: str | Path) -> MotionTrajectory:
    p = Path(path)
    with np.load(p, allow_pickle=False) as data:
        if "qpos" not in data:
            raise ValueError(f"{p} does not contain qpos")
        qpos = np.asarray(data["qpos"], dtype=float)
        fps = float(data["fps"]) if "fps" in data else 30.0
    return MotionTrajectory(qpos=qpos, fps=fps, source=str(p))


def save_npz_motion(path: str | Path, trajectory: MotionTrajectory) -> None:
    np.savez_compressed(path, qpos=trajectory.qpos, fps=np.asarray(trajectory.fps))


class MuscleMimicTrajectoryAdapter:
    """Adapter for already-retargeted MyoFullBody trajectories.

    MuscleMimic's retargeting pipeline produces qpos trajectories for MyoFullBody.
    This adapter intentionally consumes the retargeted result rather than
    reimplementing SMPL-H/GMR inside this repository.
    """

    @staticmethod
    def from_arrays(qpos, fps: float = 30.0, source: str = "musclemimic") -> MotionTrajectory:
        return MotionTrajectory(qpos=np.asarray(qpos, dtype=float), fps=float(fps), source=source)
