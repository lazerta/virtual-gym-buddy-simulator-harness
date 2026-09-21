from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .motion_sources import MotionTrajectory
from .sim_state import MuJoCoState, capture_state


class MuJoCoBackendUnavailable(RuntimeError):
    pass


GYM_EQUIPMENT_JOINTS = (
    "gym_barbell_free",
    "gym_dumbbell_left_free",
    "gym_dumbbell_right_free",
)


@dataclass(frozen=True)
class GymEquipmentConfig:
    barbell_mass_kg: float = 20.0
    dumbbell_mass_kg: float = 10.0
    add_floor: bool = False


def _imports():
    try:
        import mujoco
        import myo_sim
    except Exception as exc:  # pragma: no cover
        raise MuJoCoBackendUnavailable(
            "MuJoCo backend requires: pip install -e '.[mujoco]'"
        ) from exc
    return mujoco, myo_sim


def _joint_qpos_width(mujoco, joint_type: int) -> int:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1


def build_myofullbody_gym_model(config: GymEquipmentConfig | None = None):
    """Compose upstream MyoFullBody with minimal physical gym proxies.

    Human anatomy/dynamics come from MyoSim. Gym objects are appended as MuJoCo
    rigid bodies. The appended joints are intentionally named so retargeted
    MyoFullBody-only qpos trajectories remain separable from equipment state.
    """
    config = config or GymEquipmentConfig()
    mujoco, myo_sim = _imports()
    spec = myo_sim.load_spec("myofullbody")

    if config.add_floor and spec.geom("gym_floor") is None:
        spec.worldbody.add_geom(
            name="gym_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[8.0, 8.0, 0.1],
            rgba=[0.18, 0.18, 0.18, 1.0],
            friction=[1.0, 0.01, 0.001],
        )

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
        self._equipment_qpos_indices = self._find_equipment_qpos_indices()
        equipment = set(self._equipment_qpos_indices.tolist())
        self._human_qpos_indices = np.asarray(
            [i for i in range(self.model.nq) if i not in equipment], dtype=np.int64
        )

    def _find_equipment_qpos_indices(self) -> np.ndarray:
        ids: list[int] = []
        for name in GYM_EQUIPMENT_JOINTS:
            jid = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if jid < 0:
                raise RuntimeError(f"missing appended gym joint: {name}")
            start = int(self.model.jnt_qposadr[jid])
            width = _joint_qpos_width(self.mujoco, int(self.model.jnt_type[jid]))
            ids.extend(range(start, start + width))
        return np.asarray(sorted(ids), dtype=np.int64)

    @property
    def nq(self) -> int:
        return int(self.model.nq)

    @property
    def human_nq(self) -> int:
        return int(self._human_qpos_indices.size)

    @property
    def equipment_nq(self) -> int:
        return int(self._equipment_qpos_indices.size)

    @property
    def nv(self) -> int:
        return int(self.model.nv)

    @property
    def nu(self) -> int:
        return int(self.model.nu)

    @property
    def human_qpos_indices(self) -> np.ndarray:
        return self._human_qpos_indices.copy()

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

    def set_human_qpos(
        self,
        qpos: Iterable[float],
        *,
        reset_equipment: bool = False,
        zero_velocity: bool = True,
    ) -> MuJoCoState:
        values = np.asarray(list(qpos), dtype=float)
        if values.shape != (self.human_nq,):
            raise ValueError(
                f"human qpos must have shape ({self.human_nq},), got {values.shape}"
            )
        if reset_equipment:
            equipment_values = self.model.qpos0[self._equipment_qpos_indices].copy()
        else:
            equipment_values = self.data.qpos[self._equipment_qpos_indices].copy()

        self.data.qpos[self._human_qpos_indices] = values
        self.data.qpos[self._equipment_qpos_indices] = equipment_values
        if zero_velocity:
            self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        return capture_state(self.model, self.data)

    def replay_motion(
        self,
        motion: MotionTrajectory,
        phase: float,
        *,
        reset_equipment: bool = False,
    ) -> MuJoCoState:
        frame = motion.phase(phase)
        if frame.shape == (self.human_nq,):
            return self.set_human_qpos(
                frame, reset_equipment=reset_equipment, zero_velocity=True
            )
        if frame.shape == (self.nq,):
            return self.set_qpos(frame, zero_velocity=True)
        raise ValueError(
            f"trajectory nq={frame.shape[0]} matches neither "
            f"human_nq={self.human_nq} nor full nq={self.nq}"
        )

    def state(self) -> MuJoCoState:
        return capture_state(self.model, self.data)
