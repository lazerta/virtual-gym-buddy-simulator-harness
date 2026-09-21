from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ContactState:
    geom1: str
    geom2: str
    distance: float


@dataclass(frozen=True)
class MuJoCoState:
    time: float
    qpos: list[float]
    qvel: list[float]
    body_positions: dict[str, list[float]]
    body_quaternions: dict[str, list[float]]
    site_positions: dict[str, list[float]]
    contacts: list[ContactState]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["contacts"] = [asdict(c) for c in self.contacts]
        return out


def _name(model, obj_type: int, idx: int) -> str:
    import mujoco

    value = mujoco.mj_id2name(model, obj_type, idx)
    return value or f"unnamed_{idx}"


def capture_state(model, data) -> MuJoCoState:
    import mujoco

    body_positions: dict[str, list[float]] = {}
    body_quaternions: dict[str, list[float]] = {}
    site_positions: dict[str, list[float]] = {}

    for i in range(model.nbody):
        name = _name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        body_positions[name] = np.asarray(data.xpos[i], dtype=float).tolist()
        body_quaternions[name] = np.asarray(data.xquat[i], dtype=float).tolist()

    for i in range(model.nsite):
        name = _name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        site_positions[name] = np.asarray(data.site_xpos[i], dtype=float).tolist()

    contacts: list[ContactState] = []
    for i in range(data.ncon):
        c = data.contact[i]
        contacts.append(
            ContactState(
                geom1=_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)),
                geom2=_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2)),
                distance=float(c.dist),
            )
        )

    return MuJoCoState(
        time=float(data.time),
        qpos=np.asarray(data.qpos, dtype=float).tolist(),
        qvel=np.asarray(data.qvel, dtype=float).tolist(),
        body_positions=body_positions,
        body_quaternions=body_quaternions,
        site_positions=site_positions,
        contacts=contacts,
    )
