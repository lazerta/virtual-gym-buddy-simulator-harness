from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .motion_sources import MotionTrajectory
from .sim_state import MuJoCoState, capture_state


class MuJoCoBackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class GymEquipmentConfig:
    barbell_mass_kg: float = 20.0
    dumbbell_mass_kg: float = 10.0
    add_floor: bool = True


def _imports():
    try:
        import mujoco
        import myo_sim
    except Exception as exc:  # pragma: no cover - exercised by optional-dep environments
        raise MuJoCoBackendUnavailable(
            "MuJoCo backend requires optional dependency group: "
            "pip install -e '.[mujoco]'"
        ) from exc
    return mujoco, myo_sim


def build_myofullbody_gym_model(config: GymEquipmentConfig | None = None):
    """Build MyoFullBody and attach minimal physical gym proxies.

    Visual geometry stays in the Three.js renderer. These MuJoCo geoms are
    collision/inertia proxies only, which keeps physics stable and testable.
    """

    config = config or GymEquipmentConfig()
    mujoco, myo_sim = _imports()

    spec = myo_sim.load_spec("myofullbody")

    # Avoid duplicate floor if MyoSim's scene already contains one.
    if config.add_floor and spec.geom("gym_floor") is None:
        spec.worldbody.add_geom(
            name="gym_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[8.0, 8.0, 0.1],
            rgba=[0.18, 0.18, 0.18, 1.0],
            friction=[1.0, 0.01, 0.001],
        )

    # A standard 20 kg Olympic bar. Free joint is useful for bench / row cases.
    bar = spec.worldbody.add_body(name="gym_barbell", pos=[0.0, 0.0, 1.2])
    bar.add_freejoint(name="gym_barbell_free")
    bar.add_geom(
        name="gym_barbell_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[0.028, 1.1, 0.0],
        quat=[0.70710678, 0.0, 0.70710678, 0.0],
        mass=config.barbell_mass_kg,
        friction=[0.8, 0.01, 0.001],
        rgba=[0.35, 0.35, 0.38, 1.0],
    )

    for side, x in (("left", -0.45), ("right", 0.45)):
        db = spec.worldbody.add_body(name=f"gym_dumbbell_{side}", pos=[x, 0.0, 1.0])
        db.add_freejoint(name=f"gym_dumbbell_{side}_free")
        db.add_geom(
            name=f"gym_dumbbell_{side}_geom",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[0.065, 0.16, 0.0],
            quat=[0.70710678, 0.0, 0.70710678, 0.0],
            mass=config.dumbbell_mass_kg,
            friction=[0.8, 0.01, 0.001],
            rgba=[0.08, 0.08, 0.09, 1.0],
        )

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class MyoFullBodySimulator:
    def __init__(self, config: GymEquipmentConfig | None = None):
        self.mujoco, _ = _imports()
        self.model, self.data = build_myofullbody_gym_model(config)

    @property
    def nq(self) -> int:
        return int(self.model.nq)

    @property
    def nv(self) -> int:
        return int(self.model.nv)

    @property
    def nu(self) -> int:
        return int(self.model.nu)

    def reset(self) -> MuJoCoState:
        self.mujoco.mj_resetData(self.model, self.data)
        self.mujoco.mj_forward(self.model, self.data)
        return capture_state(self.model, self.data)

    def step(self, ctrl: Iterable[float] | None = None, nstep: int = 1) -> MuJoCoState:
        if ctrl is not None:
            values = np.asarray(list(ctrl), dtype=float)
            if values.shape != (self.model.nu,):
                raise ValueError(f"ctrl must have shape ({self.model.nu},), got {values.shape}")
            self.data.ctrl[:] = values
        self.mujoco.mj_step(self.model, self.data, nstep=max(1, int(nstep)))
        return capture_state(self.model, self.data)

    def set_qpos(self, qpos: Iterable[float], zero_velocity: bool = True) -> MuJoCoState:
        values = np.asarray(list(qpos), dtype=float)
        if values.shape != (self.model.nq,):
            raise ValueError(f"qpos must have shape ({self.model.nq},), got {values.shape}")
        self.data.qpos[:] = values
        if zero_velocity:
            self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        return capture_state(self.model, self.data)

    def replay_motion(self, motion: MotionTrajectory, phase: float) -> MuJoCoState:
        if motion.qpos.shape[1] != self.model.nq:
            raise ValueError(
                f"trajectory nq={motion.qpos.shape[1]} does not match model nq={self.model.nq}"
            )
        return self.set_qpos(motion.phase(phase), zero_velocity=True)

    def state(self) -> MuJoCoState:
        return capture_state(self.model, self.data)
