from __future__ import annotations

import json

import numpy as np

from harness.motion_sources import MotionTrajectory
from harness.mujoco_frame_export import MujocoFrameExporter, RenderCameraConfig


class FakeSimulator:
    human_nq = 2
    nq = 4

    def __init__(self):
        self.frames = []

    def set_human_qpos(self, qpos, *, reset_equipment=False, zero_velocity=True):
        self.frames.append(("human", list(qpos), reset_equipment))

    def set_qpos(self, qpos, zero_velocity=True):
        self.frames.append(("full", list(qpos), False))


class FakeExporter(MujocoFrameExporter):
    def __init__(self, simulator, camera):
        self.sim = simulator
        self.camera_config = camera
        self._renderer = None
        self._camera = None
        self._gt_index = 0

    def render_rgb(self):
        cfg = self.camera_config
        value = min(255, self._gt_index * 30)
        return np.full((cfg.height, cfg.width, 3), value, dtype=np.uint8)

    def ground_truth_state(self):
        out = {
            "time": self._gt_index / 30.0,
            "qpos": [self._gt_index],
            "qvel": [],
            "body_positions": {},
            "body_quaternions": {},
            "site_positions": {},
            "contacts": [],
        }
        self._gt_index += 1
        return out

    def close(self):
        pass


def test_exporter_writes_dense_monotonic_frame_session(tmp_path):
    sim = FakeSimulator()
    motion = MotionTrajectory(
        qpos=np.asarray([[0.0, 1.0], [0.1, 1.1], [0.2, 1.2]]),
        fps=30.0,
        source="unit-test",
    )
    camera = RenderCameraConfig(width=32, height=24, jpeg_quality=80)
    exporter = FakeExporter(sim, camera)

    manifest = exporter.export_motion(
        motion,
        tmp_path / "session",
        session_id="sim-1",
        exercise_id="smith_squat",
    )

    assert len(manifest.frames) == 3
    assert [x.frame_id for x in manifest.frames] == [0, 1, 2]
    assert [x.timestamp_us for x in manifest.frames] == [0, 33333, 66667]
    assert sim.frames[0][2] is True
    assert sim.frames[1][2] is False

    for frame in manifest.frames:
        assert (tmp_path / "session" / frame.image_path).exists()
        assert (tmp_path / "session" / frame.ground_truth_path).exists()

    gt = json.loads(
        (tmp_path / "session" / "ground_truth" / "000001.json").read_text()
    )
    assert gt["frame_id"] == 1
    assert gt["timestamp_us"] == 33333
    assert gt["motion_source"] == "unit-test"
