from __future__ import annotations

import json

import joblib
import numpy as np

from harness.video_motion_pipeline import (
    musclemimic_command,
    resolve_paths,
    wham_command,
)
from harness.wham_adapter import (
    convert_wham_to_amass,
    load_wham_output,
    longest_contiguous_indices,
    select_wham_track,
    wham_track_to_amass,
)


def _track(frame_ids, offset=0.0):
    t = len(frame_ids)
    pose = np.zeros((t, 72), dtype=np.float32)
    pose[:, 0] = np.arange(t) + offset
    pose[:, 65] = 3.0
    pose[:, 66:72] = 9.0

    trans = np.stack(
        [np.arange(t), np.zeros(t), np.ones(t)],
        axis=1,
    ).astype(np.float32)

    return {
        "pose": pose,
        "trans": trans,
        "pose_world": pose + 0.25,
        "trans_world": trans + 2.0,
        "betas": np.tile(np.arange(10, dtype=np.float32), (t, 1)),
        "frame_ids": np.asarray(frame_ids),
    }


def test_longest_contiguous_run_and_auto_track(tmp_path):
    path = tmp_path / "wham_output.pkl"
    joblib.dump(
        {
            1: _track([0, 1, 2, 8]),
            2: _track([10, 11, 12, 13, 14]),
        },
        path,
    )

    tracks = load_wham_output(path)
    selected = select_wham_track(tracks)

    assert selected.track_id == "2"
    np.testing.assert_array_equal(
        longest_contiguous_indices(tracks[0].frame_ids),
        [0, 1, 2],
    )


def test_wham_to_amass_matches_musclemimic_body_end_convention(tmp_path):
    path = tmp_path / "wham_output.pkl"
    joblib.dump({7: _track([4, 5, 6, 20, 21])}, path)

    track = select_wham_track(load_wham_output(path), 7)
    motion, meta = wham_track_to_amass(
        track,
        fps=30.0,
        coordinate_space="world",
    )

    assert motion["poses"].shape == (3, 156)
    np.testing.assert_allclose(
        motion["poses"][:, :66],
        track.pose_world[:3, :66],
    )
    np.testing.assert_allclose(motion["poses"][:, 66:], 0.0)
    np.testing.assert_allclose(motion["trans"], track.trans_world[:3])

    assert meta["coordinate_space_used"] == "world"
    assert meta["start_frame"] == 4
    assert meta["end_frame"] == 6


def test_convert_writes_amass_fields_and_metadata(tmp_path):
    pkl = tmp_path / "wham_output.pkl"
    joblib.dump({"person": _track([0, 1, 2])}, pkl)

    out = tmp_path / "motion.npz"
    result = convert_wham_to_amass(pkl, out, fps=29.97)

    with np.load(out, allow_pickle=False) as data:
        assert set(data.files) == {
            "poses",
            "trans",
            "betas",
            "gender",
            "mocap_framerate",
        }
        assert data["poses"].shape == (3, 156)
        assert abs(float(data["mocap_framerate"]) - 29.97) < 1e-3

    meta = json.loads(result.metadata_path.read_text())
    assert meta["track_id"] == "person"


def test_pipeline_commands_are_external_open_source_glue(tmp_path):
    video = tmp_path / "incline.mp4"
    video.write_bytes(b"placeholder")

    paths = resolve_paths(
        video,
        "incline_smith_press",
        work_root=tmp_path / "work",
        output_root=tmp_path / "motions",
    )

    wham = wham_command(
        paths,
        wham_root=tmp_path / "WHAM",
        wham_python="python",
    )
    assert "demo.py" in wham[1]
    assert "--save_pkl" in wham

    mm = musclemimic_command(
        paths,
        musclemimic_root=tmp_path / "musclemimic",
    )
    assert mm[:3] == ["uv", "run", "python"]
    assert str(paths.amass_npz) in mm
    assert str(paths.myofullbody_npz) in mm
