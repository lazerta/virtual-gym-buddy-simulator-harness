from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import subprocess
from threading import Thread

import pytest

from harness.video_motion_pipeline import (
    download_public_video,
    public_video_target,
)


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def _make_fixture(path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required for live URL ingestion smoke")

    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x120:r=15:d=1",
            "-an",
            "-c:v",
            "mpeg4",
            str(path),
        ],
        check=True,
    )

    assert path.exists()
    assert path.stat().st_size > 1_000


def test_real_yt_dlp_download_from_local_http(tmp_path):
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    fixture = fixture_dir / "fixture.mp4"
    _make_fixture(fixture)

    handler = partial(_QuietHandler, directory=str(fixture_dir))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address[:2]
        url = f"http://{host}:{port}/fixture.mp4"
        work_root = tmp_path / "work"

        expected = public_video_target(
            url,
            "smith_squat",
            work_root=work_root,
        ).resolve()

        downloaded = download_public_video(
            url,
            "smith_squat",
            work_root=work_root,
        ).resolve()

        assert downloaded == expected
        assert downloaded.suffix == ".mp4"
        assert downloaded.exists()
        assert downloaded.stat().st_size > 1_000

        first_mtime_ns = downloaded.stat().st_mtime_ns
        first_size = downloaded.stat().st_size

        downloaded_again = download_public_video(
            url,
            "smith_squat",
            work_root=work_root,
        ).resolve()

        assert downloaded_again == downloaded
        assert downloaded_again.stat().st_size == first_size
        assert downloaded_again.stat().st_mtime_ns == first_mtime_ns
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
