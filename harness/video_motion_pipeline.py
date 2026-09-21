from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .motion_registry import CANONICAL_EXERCISES, motion_root
from .wham_adapter import convert_wham_to_amass


@dataclass(frozen=True)
class VideoPipelinePaths:
    video: Path
    exercise_id: str
    work_dir: Path
    wham_output_dir: Path
    wham_pkl: Path
    amass_npz: Path
    myofullbody_npz: Path


def _require_dir(value: str | os.PathLike[str] | None, env_name: str) -> Path:
    raw = value or os.environ.get(env_name)
    if not raw:
        raise ValueError(f"set {env_name} or pass its CLI override")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{env_name} path does not exist: {path}")
    return path


def resolve_paths(
    video: str | os.PathLike[str],
    exercise_id: str,
    *,
    work_root: str | os.PathLike[str] = "artifacts/video_motion",
    output_root: str | os.PathLike[str] | None = None,
    require_video: bool = True,
) -> VideoPipelinePaths:
    if exercise_id not in CANONICAL_EXERCISES:
        raise KeyError(f"unknown canonical exercise: {exercise_id}")

    video_path = Path(video).expanduser().resolve()
    if require_video and not video_path.exists():
        raise FileNotFoundError(video_path)

    work_dir = Path(work_root).expanduser().resolve() / exercise_id / video_path.stem
    wham_output_dir = work_dir / "wham"
    sequence_dir = wham_output_dir / video_path.stem
    output_base = motion_root(output_root).resolve()

    return VideoPipelinePaths(
        video=video_path,
        exercise_id=exercise_id,
        work_dir=work_dir,
        wham_output_dir=wham_output_dir,
        wham_pkl=sequence_dir / "wham_output.pkl",
        amass_npz=work_dir / f"{video_path.stem}_smplh_amass.npz",
        myofullbody_npz=output_base / f"{exercise_id}.npz",
    )


def public_video_target(
    url: str,
    exercise_id: str,
    *,
    work_root: str | os.PathLike[str] = "artifacts/video_motion",
) -> Path:
    if exercise_id not in CANONICAL_EXERCISES:
        raise KeyError(f"unknown canonical exercise: {exercise_id}")
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    root = Path(work_root).expanduser().resolve() / "_downloads" / exercise_id
    return root / f"url_{digest}.mp4"


def yt_dlp_command(url: str, target: str | os.PathLike[str]) -> list[str]:
    target = Path(target).expanduser().resolve()
    template = target.with_suffix(".%(ext)s")
    return [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--no-part",
        "--merge-output-format",
        "mp4",
        "-f",
        "bv*+ba/b",
        "-o",
        str(template),
        "--print",
        "after_move:filepath",
        str(url),
    ]


def download_public_video(
    url: str,
    exercise_id: str,
    *,
    work_root: str | os.PathLike[str] = "artifacts/video_motion",
    dry_run: bool = False,
) -> Path:
    target = public_video_target(url, exercise_id, work_root=work_root)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and target.stat().st_size > 0:
        print(f"reusing downloaded video: {target}")
        return target

    cmd = yt_dlp_command(url, target)
    print("$", " ".join(cmd))
    if dry_run:
        return target

    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
    )

    reported = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if reported:
        candidate = Path(reported[-1]).expanduser()
        if candidate.exists():
            if candidate.resolve() != target.resolve():
                target = candidate.resolve()
            return target

    if target.exists():
        return target

    fallback = sorted(target.parent.glob(f"{target.stem}.*"))
    fallback = [p for p in fallback if p.is_file() and p.suffix not in {".part", ".ytdl"}]
    if fallback:
        return fallback[0].resolve()

    raise FileNotFoundError(f"yt-dlp completed without an output file for {url}")


def wham_command(
    paths: VideoPipelinePaths,
    *,
    wham_root: str | os.PathLike[str],
    wham_python: str | None = None,
    local_only: bool = False,
) -> list[str]:
    root = Path(wham_root).expanduser().resolve()
    python = wham_python or os.environ.get("WHAM_PYTHON") or sys.executable

    cmd = [
        str(python),
        str(root / "demo.py"),
        "--video",
        str(paths.video),
        "--output_pth",
        str(paths.wham_output_dir),
        "--save_pkl",
    ]
    if local_only:
        cmd.append("--estimate_local_only")
    return cmd


def musclemimic_command(
    paths: VideoPipelinePaths,
    *,
    musclemimic_root: str | os.PathLike[str],
    target_fps: float = 30.0,
) -> list[str]:
    root = Path(musclemimic_root).expanduser().resolve()
    helper = Path(__file__).resolve().parent.parent / "scripts" / "musclemimic_gmr_retarget.py"

    return [
        "uv",
        "run",
        "python",
        str(helper),
        "--input",
        str(paths.amass_npz),
        "--output",
        str(paths.myofullbody_npz),
        "--target-fps",
        str(float(target_fps)),
    ]


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_video_pipeline(
    video: str | os.PathLike[str],
    exercise_id: str,
    *,
    wham_root: str | os.PathLike[str] | None = None,
    musclemimic_root: str | os.PathLike[str] | None = None,
    wham_python: str | None = None,
    work_root: str | os.PathLike[str] = "artifacts/video_motion",
    output_root: str | os.PathLike[str] | None = None,
    coordinate_space: str = "world",
    target_fps: float = 30.0,
    track_id: str | int | None = None,
    local_only: bool = False,
    dry_run: bool = False,
) -> VideoPipelinePaths:
    paths = resolve_paths(
        video,
        exercise_id,
        work_root=work_root,
        output_root=output_root,
        require_video=not dry_run,
    )

    paths.work_dir.mkdir(parents=True, exist_ok=True)
    paths.wham_output_dir.mkdir(parents=True, exist_ok=True)
    paths.myofullbody_npz.parent.mkdir(parents=True, exist_ok=True)

    wham = _require_dir(wham_root, "WHAM_ROOT")
    musclemimic = _require_dir(musclemimic_root, "MUSCLEMIMIC_ROOT")

    _run(
        wham_command(
            paths,
            wham_root=wham,
            wham_python=wham_python,
            local_only=local_only,
        ),
        cwd=wham,
        dry_run=dry_run,
    )

    if dry_run:
        print(
            json.dumps(
                {
                    "expected_wham_pkl": str(paths.wham_pkl),
                    "amass_npz": str(paths.amass_npz),
                    "myofullbody_npz": str(paths.myofullbody_npz),
                },
                indent=2,
            )
        )
    else:
        if not paths.wham_pkl.exists():
            raise FileNotFoundError(
                f"WHAM finished but output was not found: {paths.wham_pkl}"
            )

        result = convert_wham_to_amass(
            paths.wham_pkl,
            paths.amass_npz,
            video_path=paths.video,
            track_id=track_id,
            coordinate_space=coordinate_space,
        )

        print(
            json.dumps(
                {
                    "wham_track": result.track_id,
                    "coordinate_space": result.coordinate_space,
                    "frames": result.exported_frames,
                    "fps": result.fps,
                    "amass_npz": str(result.output_path),
                },
                indent=2,
            )
        )

    _run(
        musclemimic_command(
            paths,
            musclemimic_root=musclemimic,
            target_fps=target_fps,
        ),
        cwd=musclemimic,
        dry_run=dry_run,
    )

    if not dry_run and not paths.myofullbody_npz.exists():
        raise FileNotFoundError(
            f"MuscleMimic retargeting completed without output: {paths.myofullbody_npz}"
        )

    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gym-buddy-video-motion")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video")
    source.add_argument("--url")

    parser.add_argument("--exercise", required=True, choices=CANONICAL_EXERCISES)
    parser.add_argument("--wham-root", default=None)
    parser.add_argument("--musclemimic-root", default=None)
    parser.add_argument("--wham-python", default=None)
    parser.add_argument("--work-root", default="artifacts/video_motion")
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--coordinate-space",
        choices=["world", "camera"],
        default="world",
    )
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--track-id", default=None)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    video = args.video
    if args.url:
        video = str(
            download_public_video(
                args.url,
                args.exercise,
                work_root=args.work_root,
                dry_run=args.dry_run,
            )
        )

    paths = run_video_pipeline(
        video,
        args.exercise,
        wham_root=args.wham_root,
        musclemimic_root=args.musclemimic_root,
        wham_python=args.wham_python,
        work_root=args.work_root,
        output_root=args.output_root,
        coordinate_space=args.coordinate_space,
        target_fps=args.target_fps,
        track_id=args.track_id,
        local_only=args.local_only,
        dry_run=args.dry_run,
    )

    print(json.dumps({"output": str(paths.myofullbody_npz)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
