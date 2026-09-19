from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import cv2
import numpy as np


@dataclass(frozen=True)
class PoseFrame:
    frame_index: int
    timestamp_ms: float
    landmarks: np.ndarray
    confidence: np.ndarray


class PoseProvider(Protocol):
    name: str
    def infer(self, frame_bgr: np.ndarray, frame_index: int, timestamp_ms: float) -> PoseFrame | None: ...


class MediaPipePoseProvider:
    name = "mediapipe"

    def __init__(self, model_path: str, num_poses: int = 2):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("mediapipe is not installed; install the harness 'vision' extra") from exc
        self.mp = mp
        BaseOptions = mp.tasks.BaseOptions
        vision = mp.tasks.vision
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)

    def infer(self, frame_bgr: np.ndarray, frame_index: int, timestamp_ms: float) -> PoseFrame | None:
        rgb=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
        img=self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=rgb)
        result=self.landmarker.detect_for_video(img,int(timestamp_ms))
        if not result.pose_landmarks:
            return None
        lm=result.pose_landmarks[0]
        arr=np.array([[p.x,p.y,p.z] for p in lm],dtype=np.float32)
        conf=np.array([getattr(p,"visibility",1.0) or 0.0 for p in lm],dtype=np.float32)
        return PoseFrame(frame_index,timestamp_ms,arr,conf)


class RgbVideoRunner:
    def __init__(self, pose_provider: PoseProvider):
        self.pose_provider=pose_provider

    def run(self, video_path: str | Path, max_frames: int | None = None) -> list[PoseFrame]:
        cap=cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {video_path}")
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
        out=[]; idx=0
        try:
            while True:
                ok,frame=cap.read()
                if not ok: break
                ts=1000.0*idx/fps
                pose=self.pose_provider.infer(frame,idx,ts)
                if pose is not None: out.append(pose)
                idx+=1
                if max_frames is not None and idx>=max_frames: break
        finally:
            cap.release()
        return out
