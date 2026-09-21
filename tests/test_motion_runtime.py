from __future__ import annotations

import numpy as np
import pytest

from harness.motion_registry import CANONICAL_EXERCISES, motion_status
from harness.motion_sources import load_musclemimic_npz


def test_registry_covers_all_ten_exercises(tmp_path):
    status = motion_status(tmp_path)
    assert len(status) == 10
    assert {x["exercise_id"] for x in status} == set(CANONICAL_EXERCISES)
    assert not any(x["available"] for x in status)


def test_load_musclemimic_serialized_trajectory_segment(tmp_path):
    path = tmp_path / "motion.npz"
    qpos = np.arange(30, dtype=float).reshape(5, 6)
    qvel = np.arange(25, dtype=float).reshape(5, 5)
    np.savez(
        path,
        qpos=qpos,
        qvel=qvel,
        split_points=np.asarray([0, 2, 5]),
        frequency=np.asarray(50.0),
    )

    motion = load_musclemimic_npz(path, trajectory_index=1)
    assert motion.frames == 3
    assert motion.fps == 50.0
    np.testing.assert_allclose(motion.qpos, qpos[2:5])
    np.testing.assert_allclose(motion.qvel, qvel[2:5])


def test_backend_accepts_human_only_qpos_when_optional_deps_exist():
    pytest.importorskip("mujoco")
    pytest.importorskip("myo_sim")

    from harness.mujoco_backend import MyoFullBodySimulator
    from harness.motion_sources import MotionTrajectory

    sim = MyoFullBodySimulator()
    assert sim.equipment_nq == 21
    assert sim.human_nq + sim.equipment_nq == sim.nq

    qpos = sim.data.qpos[sim.human_qpos_indices].copy()
    motion = MotionTrajectory(qpos=np.stack([qpos, qpos]), fps=30.0, source="test")
    state = sim.replay_motion(motion, 0.5, reset_equipment=True)
    assert len(state.qpos) == sim.nq
