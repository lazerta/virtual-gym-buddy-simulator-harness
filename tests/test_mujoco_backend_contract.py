from __future__ import annotations

import numpy as np
import pytest

from harness.motion_sources import MotionTrajectory, MuscleMimicTrajectoryAdapter


def test_motion_trajectory_interpolates():
    qpos = np.array([[0.0, 2.0], [2.0, 4.0]])
    motion = MotionTrajectory(qpos=qpos, fps=30.0, source="test")
    np.testing.assert_allclose(motion.phase(0.5), [1.0, 3.0])


def test_musclemimic_adapter_preserves_qpos():
    arr = np.arange(12, dtype=float).reshape(3, 4)
    motion = MuscleMimicTrajectoryAdapter.from_arrays(arr, fps=60)
    assert motion.frames == 3
    assert motion.fps == 60


def test_optional_mujoco_backend_builds_when_installed():
    pytest.importorskip("mujoco")
    pytest.importorskip("myo_sim")

    from harness.mujoco_backend import MyoFullBodySimulator

    sim = MyoFullBodySimulator()
    assert sim.nq > 100
    assert sim.nu >= 400

    state = sim.reset()
    assert len(state.qpos) == sim.nq
    assert "gym_barbell" in state.body_positions

    stepped = sim.step(np.zeros(sim.nu), nstep=2)
    assert stepped.time > 0
