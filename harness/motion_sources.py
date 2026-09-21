from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MotionTrajectory:
    qpos: np.ndarray
    fps: float
    source: str
    qvel: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.qpos.ndim != 2:
            raise ValueError("qpos must be [frames, nq]")
        if self.qpos.shape[0] < 1:
            raise ValueError("trajectory must contain at least one frame")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.qvel is not None and self.qvel.shape[0] != self.qpos.shape[0]:
            raise ValueError("qvel and qpos must contain the same number of frames")

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
        qvel = np.asarray(data["qvel"], dtype=float) if "qvel" in data else None
    return MotionTrajectory(qpos=qpos, qvel=qvel, fps=fps, source=str(p))


def load_musclemimic_npz(
    path: str | Path,
    *,
    trajectory_index: int = 0,
) -> MotionTrajectory:
    """Read MuscleMimic/LocoMuJoCo Trajectory.save() output without importing it.

    Upstream serializes TrajectoryData fields directly into NPZ. We only consume
    qpos/qvel/split_points/frequency, keeping the heavyweight JAX/Flax stack
    outside the Gym Buddy runtime.
    """
    p = Path(path)
    with np.load(p, allow_pickle=True) as data:
        if "qpos" not in data:
            raise ValueError(f"{p} is not a MuscleMimic trajectory: missing qpos")

        qpos = np.asarray(data["qpos"], dtype=float)
        qvel = np.asarray(data["qvel"], dtype=float) if "qvel" in data else None

        if "frequency" in data:
            frequency = float(np.asarray(data["frequency"]).reshape(-1)[0])
        elif "fps" in data:
            frequency = float(np.asarray(data["fps"]).reshape(-1)[0])
        else:
            frequency = 30.0

        if "split_points" in data:
            split = np.asarray(data["split_points"], dtype=int).reshape(-1)
            if split.size < 2:
                raise ValueError("split_points must contain at least [start, end]")
            n = split.size - 1
            idx = int(trajectory_index)
            if idx < 0:
                idx += n
            if not 0 <= idx < n:
                raise IndexError(f"trajectory_index={trajectory_index} outside 0..{n-1}")
            lo, hi = int(split[idx]), int(split[idx + 1])
            qpos = qpos[lo:hi]
            if qvel is not None and qvel.size:
                qvel = qvel[lo:hi]

    return MotionTrajectory(
        qpos=qpos,
        qvel=qvel,
        fps=frequency,
        source=f"{p}#trajectory={trajectory_index}",
    )


def save_npz_motion(path: str | Path, trajectory: MotionTrajectory) -> None:
    kwargs = {"qpos": trajectory.qpos, "fps": np.asarray(trajectory.fps)}
    if trajectory.qvel is not None:
        kwargs["qvel"] = trajectory.qvel
    np.savez_compressed(path, **kwargs)


class MuscleMimicTrajectoryAdapter:
    """Consume upstream retargeted MyoFullBody trajectories; do not reimplement GMR."""

    @staticmethod
    def from_arrays(
        qpos,
        fps: float = 30.0,
        source: str = "musclemimic",
        qvel=None,
    ) -> MotionTrajectory:
        return MotionTrajectory(
            qpos=np.asarray(qpos, dtype=float),
            qvel=None if qvel is None else np.asarray(qvel, dtype=float),
            fps=float(fps),
            source=source,
        )
