from __future__ import annotations

import json
from threading import Thread
import urllib.error
import urllib.request

from harness.frame_contract import FrameRecord, FrameSessionManifest, save_manifest
from harness.frame_server import create_server


def _session(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "ground_truth").mkdir()

    (tmp_path / "frames" / "000000.jpg").write_bytes(b"jpeg-bytes")
    (tmp_path / "ground_truth" / "000000.json").write_text(
        json.dumps({"frame_id": 0, "truth": "hidden-from-app"}),
        encoding="utf-8",
    )

    manifest = FrameSessionManifest(
        session_id="session-1",
        exercise_id="incline_db_press",
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
        ),
    )
    save_manifest(tmp_path / "manifest.json", manifest)
    return manifest


def _read_json(url):
    with urllib.request.urlopen(url, timeout=2) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_frame_server_serves_frames_and_accepts_results(tmp_path):
    _session(tmp_path)
    server = create_server(tmp_path, "127.0.0.1", 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address[:2]
        base = f"http://{host}:{port}"

        session = _read_json(base + "/v1/session")
        assert session["session_id"] == "session-1"
        assert session["frames"][0]["timestamp_us"] == 0
        assert "ground_truth_path" not in session["frames"][0]
        assert "image_path" not in session["frames"][0]

        with urllib.request.urlopen(base + "/v1/frames/0", timeout=2) as resp:
            assert resp.read() == b"jpeg-bytes"
            assert resp.headers["X-GymBuddy-Frame-Id"] == "0"
            assert resp.headers["X-GymBuddy-Timestamp-Us"] == "0"

        body = json.dumps(
            {
                "schema_version": 1,
                "session_id": "session-1",
                "frame_id": 0,
                "timestamp_us": 0,
                "analysis": {
                    "pose_landmarks": 33,
                    "rep_count": 1,
                },
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            base + "/v1/results",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            assert resp.status == 201

        results = _read_json(base + "/v1/results")
        assert results[0]["analysis"]["rep_count"] == 1

        # Ground truth is a harness-only endpoint. It is intentionally separate
        # from the frame payload consumed by the app.
        gt = _read_json(base + "/v1/ground-truth/0")
        assert gt["truth"] == "hidden-from-app"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_frame_server_rejects_timestamp_mismatch(tmp_path):
    _session(tmp_path)
    server = create_server(tmp_path, "127.0.0.1", 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address[:2]
        body = json.dumps(
            {
                "schema_version": 1,
                "session_id": "session-1",
                "frame_id": 0,
                "timestamp_us": 999,
                "analysis": {},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"http://{host}:{port}/v1/results",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=2)
            raise AssertionError("expected HTTP 400")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
