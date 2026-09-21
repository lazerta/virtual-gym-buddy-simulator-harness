from __future__ import annotations

import json

import pytest

from harness.frame_contract import (
    AppFrameResult,
    FrameRecord,
    FrameSessionManifest,
    frame_by_id,
    load_manifest,
    save_manifest,
    validate_result_against_manifest,
)


def _manifest():
    return FrameSessionManifest(
        session_id="s1",
        exercise_id="smith_squat",
        fps=30.0,
        width=640,
        height=480,
        frames=(
            FrameRecord(
                frame_id=0,
                timestamp_us=0,
                width=640,
                height=480,
                mime_type="image/jpeg",
                image_path="frames/000000.jpg",
                ground_truth_path="ground_truth/000000.json",
            ),
            FrameRecord(
                frame_id=1,
                timestamp_us=33333,
                width=640,
                height=480,
                mime_type="image/jpeg",
                image_path="frames/000001.jpg",
                ground_truth_path="ground_truth/000001.json",
            ),
        ),
    )


def test_manifest_round_trip(tmp_path):
    manifest = _manifest()
    path = tmp_path / "manifest.json"
    save_manifest(path, manifest)

    loaded = load_manifest(path)
    assert loaded == manifest
    payload = json.loads(path.read_text())
    assert payload["frame_count"] == 2
    assert payload["frames"][1]["timestamp_us"] == 33333


def test_result_must_echo_exact_frame_timestamp():
    manifest = _manifest()
    ok = AppFrameResult(
        session_id="s1",
        frame_id=1,
        timestamp_us=33333,
        analysis={"rep_count": 1},
    )
    assert validate_result_against_manifest(manifest, ok) == frame_by_id(manifest, 1)

    bad = AppFrameResult(
        session_id="s1",
        frame_id=1,
        timestamp_us=33000,
        analysis={},
    )
    with pytest.raises(ValueError, match="timestamp"):
        validate_result_against_manifest(manifest, bad)


def test_manifest_rejects_non_monotonic_timestamps():
    with pytest.raises(ValueError, match="strictly increasing"):
        FrameSessionManifest(
            session_id="s1",
            exercise_id="smith_squat",
            fps=30.0,
            width=640,
            height=480,
            frames=(
                FrameRecord(
                    frame_id=0,
                    timestamp_us=1000,
                    width=640,
                    height=480,
                    mime_type="image/jpeg",
                    image_path="frames/0.jpg",
                    ground_truth_path="ground_truth/0.json",
                ),
                FrameRecord(
                    frame_id=1,
                    timestamp_us=1000,
                    width=640,
                    height=480,
                    mime_type="image/jpeg",
                    image_path="frames/1.jpg",
                    ground_truth_path="ground_truth/1.json",
                ),
            ),
        )


def test_session_paths_cannot_escape_root():
    with pytest.raises(ValueError, match="cannot contain"):
        FrameRecord(
            frame_id=0,
            timestamp_us=0,
            width=1,
            height=1,
            mime_type="image/jpeg",
            image_path="../secret.jpg",
            ground_truth_path="ground_truth/0.json",
        )
