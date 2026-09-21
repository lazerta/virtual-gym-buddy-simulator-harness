from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np

from .frame_contract import FrameRecord, FrameSessionManifest, save_manifest
from .motion_sources import MotionTrajectory, load_musclemimic_npz
from .mujoco_backend import MyoFullBodySimulator
from .sim_state import capture_state


@dataclass(frozen=True)
class RenderCameraConfig:
    width: int = 640
    height: int = 480
    azimuth_deg: float = 90.0
    elevation_deg: float = -8.0
    distance_m: float = 3.2
    lookat_x: float = 0.0
    lookat_y: float = 0.0
    lookat_z: float = 1.0
    jpeg_quality: int = 92

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("render dimensions must be positive")
        if self.distance_m <= 0:
            raise ValueError("camera distance must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in 1..100")


class MujocoFrameExporter:
    def __init__(
        self,
        simulator: MyoFullBodySimulator | None = None,
        camera: RenderCameraConfig | None = None,
    ):
        self.sim = simulator or MyoFullBodySimulator()
        self.camera_config = camera or RenderCameraConfig()
        self._renderer = None
        self._camera = None

    def __enter__(self) -> "MujocoFrameExporter":
        self._ensure_renderer()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._camera = None

    def _ensure_renderer(self) -> None:
        if self._renderer is not None:
            return

        mujoco = self.sim.mujoco
        cfg = self.camera_config

        self._renderer = mujoco.Renderer(
            self.sim.model,
            height=cfg.height,
            width=cfg.width,
        )

        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.sim.model, camera)
        camera.azimuth = float(cfg.azimuth_deg)
        camera.elevation = float(cfg.elevation_deg)
        camera.distance = float(cfg.distance_m)
        camera.lookat[:] = [
            float(cfg.lookat_x),
            float(cfg.lookat_y),
            float(cfg.lookat_z),
        ]
        self._camera = camera

    def _set_motion_frame(
        self,
        motion: MotionTrajectory,
        frame_index: int,
    ) -> None:
        qpos = motion.frame(frame_index)
        if qpos.shape == (self.sim.human_nq,):
            self.sim.set_human_qpos(
                qpos,
                reset_equipment=(frame_index == 0),
                zero_velocity=True,
            )
        elif qpos.shape == (self.sim.nq,):
            self.sim.set_qpos(qpos, zero_velocity=True)
        else:
            raise ValueError(
                f"trajectory qpos width {qpos.shape[0]} matches neither "
                f"human_nq={self.sim.human_nq} nor full nq={self.sim.nq}"
            )

    def ground_truth_state(self) -> dict[str, Any]:
        return capture_state(self.sim.model, self.sim.data).to_dict()

    def render_rgb(self) -> np.ndarray:
        self._ensure_renderer()
        assert self._renderer is not None
        assert self._camera is not None
        self._renderer.update_scene(self.sim.data, self._camera)
        rgb = np.asarray(self._renderer.render())
        expected = (
            self.camera_config.height,
            self.camera_config.width,
            3,
        )
        if rgb.shape != expected or rgb.dtype != np.uint8:
            raise RuntimeError(
                f"unexpected MuJoCo RGB frame: shape={rgb.shape}, "
                f"dtype={rgb.dtype}; expected {expected}/uint8"
            )
        return rgb

    def export_motion(
        self,
        motion: MotionTrajectory,
        output_dir: str | Path,
        *,
        session_id: str,
        exercise_id: str,
        overwrite: bool = False,
    ) -> FrameSessionManifest:
        root = Path(output_dir).expanduser().resolve()

        if root.exists():
            if not overwrite and any(root.iterdir()):
                raise FileExistsError(
                    f"output session already exists and is non-empty: {root}"
                )
            if overwrite:
                shutil.rmtree(root)

        frames_dir = root / "frames"
        gt_dir = root / "ground_truth"
        frames_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        records: list[FrameRecord] = []
        cfg = self.camera_config

        for i in range(motion.frames):
            self._set_motion_frame(motion, i)
            rgb = self.render_rgb()

            image_rel = f"frames/{i:06d}.jpg"
            gt_rel = f"ground_truth/{i:06d}.json"
            image_path = root / image_rel
            gt_path = root / gt_rel

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(
                ".jpg",
                bgr,
                [cv2.IMWRITE_JPEG_QUALITY, int(cfg.jpeg_quality)],
            )
            if not ok:
                raise RuntimeError(f"JPEG encode failed at frame {i}")
            image_path.write_bytes(encoded.tobytes())

            timestamp_us = int(round(i * 1_000_000.0 / motion.fps))
            state = self.ground_truth_state()

            gt_payload: dict[str, Any] = {
                "schema_version": 1,
                "session_id": session_id,
                "exercise_id": exercise_id,
                "frame_id": i,
                "timestamp_us": timestamp_us,
                "motion_source": motion.source,
                "camera": {
                    **asdict(cfg),
                    "lookat": [
                        cfg.lookat_x,
                        cfg.lookat_y,
                        cfg.lookat_z,
                    ],
                },
                "mujoco": state,
            }
            gt_path.write_text(
                json.dumps(gt_payload, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

            records.append(
                FrameRecord(
                    frame_id=i,
                    timestamp_us=timestamp_us,
                    width=cfg.width,
                    height=cfg.height,
                    mime_type="image/jpeg",
                    image_path=image_rel,
                    ground_truth_path=gt_rel,
                )
            )

        manifest = FrameSessionManifest(
            session_id=session_id,
            exercise_id=exercise_id,
            fps=float(motion.fps),
            width=cfg.width,
            height=cfg.height,
            frames=tuple(records),
            source="mujoco-myofullbody",
        )
        save_manifest(root / "manifest.json", manifest)
        return manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gym-buddy-export-frames")
    p.add_argument("--motion", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--session-id", required=True)
    p.add_argument("--exercise", required=True)
    p.add_argument("--trajectory-index", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--azimuth", type=float, default=90.0)
    p.add_argument("--elevation", type=float, default=-8.0)
    p.add_argument("--distance", type=float, default=3.2)
    p.add_argument("--lookat-z", type=float, default=1.0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    motion = load_musclemimic_npz(
        args.motion,
        trajectory_index=args.trajectory_index,
    )
    camera = RenderCameraConfig(
        width=args.width,
        height=args.height,
        azimuth_deg=args.azimuth,
        elevation_deg=args.elevation,
        distance_m=args.distance,
        lookat_z=args.lookat_z,
    )

    with MujocoFrameExporter(camera=camera) as exporter:
        manifest = exporter.export_motion(
            motion,
            args.output,
            session_id=args.session_id,
            exercise_id=args.exercise,
            overwrite=args.overwrite,
        )

    print(
        json.dumps(
            {
                "session_id": manifest.session_id,
                "exercise_id": manifest.exercise_id,
                "frame_count": len(manifest.frames),
                "fps": manifest.fps,
                "output": str(Path(args.output).resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
