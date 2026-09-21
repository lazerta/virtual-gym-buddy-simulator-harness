from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

from .frame_contract import (
    AppFrameResult,
    FrameSessionManifest,
    frame_by_id,
    load_manifest,
    resolve_session_path,
    validate_result_against_manifest,
)


class FrameSessionStore:
    def __init__(self, session_root: str | Path):
        self.root = Path(session_root).expanduser().resolve()
        self.manifest_path = self.root / "manifest.json"
        self.manifest = load_manifest(self.manifest_path)
        self.results_path = self.root / "results.jsonl"
        self._result_lock = Lock()

    def frame_bytes(self, frame_id: int) -> tuple[bytes, str]:
        frame = frame_by_id(self.manifest, frame_id)
        path = resolve_session_path(self.root, frame.image_path)
        return path.read_bytes(), frame.mime_type

    def ground_truth(self, frame_id: int) -> dict:
        frame = frame_by_id(self.manifest, frame_id)
        path = resolve_session_path(self.root, frame.ground_truth_path)
        return json.loads(path.read_text(encoding="utf-8"))

    def append_result(self, payload: dict) -> AppFrameResult:
        result = AppFrameResult(
            schema_version=int(payload.get("schema_version", 1)),
            session_id=str(payload["session_id"]),
            frame_id=int(payload["frame_id"]),
            timestamp_us=int(payload["timestamp_us"]),
            analysis=dict(payload.get("analysis") or {}),
        )
        validate_result_against_manifest(self.manifest, result)
        line = json.dumps(result.to_dict(), separators=(",", ":")) + "\n"
        with self._result_lock:
            with self.results_path.open("a", encoding="utf-8") as f:
                f.write(line)
        return result

    def results(self) -> list[dict]:
        if not self.results_path.exists():
            return []
        out = []
        for line in self.results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out


def make_handler(store: FrameSessionStore):
    class Handler(BaseHTTPRequestHandler):
        server_version = "GymBuddyFrameServer/1"

        def log_message(self, format, *args):  # pragma: no cover
            return

        def _json(self, status: int, payload: dict | list) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message})

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == "/v1/session":
                    payload = store.manifest.to_dict()
                    for frame in payload["frames"]:
                        frame.pop("ground_truth_path", None)
                        frame.pop("image_path", None)
                    payload["protocol"] = {
                        "frame": "/v1/frames/{frame_id}",
                        "ground_truth": "/v1/ground-truth/{frame_id}",
                        "results": "/v1/results",
                    }
                    self._json(HTTPStatus.OK, payload)
                    return

                if path == "/v1/results":
                    self._json(HTTPStatus.OK, store.results())
                    return

                if path.startswith("/v1/frames/"):
                    frame_id = int(path.rsplit("/", 1)[-1])
                    frame = frame_by_id(store.manifest, frame_id)
                    body, mime_type = store.frame_bytes(frame_id)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", mime_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-GymBuddy-Session-Id", store.manifest.session_id)
                    self.send_header("X-GymBuddy-Frame-Id", str(frame.frame_id))
                    self.send_header("X-GymBuddy-Timestamp-Us", str(frame.timestamp_us))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if path.startswith("/v1/ground-truth/"):
                    frame_id = int(path.rsplit("/", 1)[-1])
                    self._json(HTTPStatus.OK, store.ground_truth(frame_id))
                    return

                self._error(HTTPStatus.NOT_FOUND, "not found")
            except (KeyError, ValueError, FileNotFoundError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))

        def do_POST(self):
            path = urlparse(self.path).path
            if path != "/v1/results":
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 4_000_000:
                    raise ValueError("invalid request body size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                result = store.append_result(payload)
                self._json(HTTPStatus.CREATED, result.to_dict())
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))

    return Handler


def create_server(
    session_root: str | Path,
    host: str = "127.0.0.1",
    port: int = 8788,
) -> ThreadingHTTPServer:
    store = FrameSessionStore(session_root)
    return ThreadingHTTPServer((host, int(port)), make_handler(store))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gym-buddy-frame-server")
    p.add_argument("--session", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8788)
    args = p.parse_args(argv)

    server = create_server(args.session, args.host, args.port)
    host, port = server.server_address[:2]
    print(
        f"Gym Buddy frame server: http://{host}:{port}/v1/session "
        f"(session={Path(args.session).resolve()})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
