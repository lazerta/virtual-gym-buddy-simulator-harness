from __future__ import annotations

import json
import subprocess
import urllib.request
from dataclasses import asdict
from typing import Protocol, runtime_checkable

from .models import AnalyzerOutput, ExerciseProfile, Observation


@runtime_checkable
class ProductionAdapter(Protocol):
    """Black-box boundary between the Python harness and Gym Buddy production logic."""

    name: str

    def analyze(self, obs: Observation, ex: ExerciseProfile, calibration=None) -> AnalyzerOutput:
        ...


def request_payload(obs: Observation, ex: ExerciseProfile, calibration=None) -> dict:
    return {
        "schema_version": 1,
        "observation": asdict(obs),
        "exercise_profile": asdict(ex),
        "personal_calibration": asdict(calibration) if calibration is not None else None,
    }


def parse_output(payload: dict) -> AnalyzerOutput:
    required = {"ready", "pause", "reset_count", "rep_count", "cue_count", "reason"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"production adapter output missing fields: {sorted(missing)}")
    return AnalyzerOutput(
        ready=bool(payload["ready"]),
        pause=bool(payload["pause"]),
        reset_count=int(payload["reset_count"]),
        rep_count=int(payload["rep_count"]),
        cue_count=int(payload["cue_count"]),
        reason=str(payload["reason"]),
    )


class JsonlSubprocessAdapter:
    name = "jsonl-subprocess"

    def __init__(self, command: list[str], timeout_s: float = 10.0):
        self.command = command
        self.timeout_s = timeout_s

    def analyze(self, obs: Observation, ex: ExerciseProfile, calibration=None) -> AnalyzerOutput:
        payload = json.dumps(request_payload(obs, ex, calibration), separators=(",", ":")) + "\n"
        proc = subprocess.run(
            self.command,
            input=payload,
            text=True,
            capture_output=True,
            timeout=self.timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"production analyzer failed ({proc.returncode}): {proc.stderr.strip()}")
        line = next((ln for ln in proc.stdout.splitlines() if ln.strip()), None)
        if line is None:
            raise RuntimeError("production analyzer returned no JSON")
        return parse_output(json.loads(line))


class HttpProductionAdapter:
    name = "http-local"

    def __init__(self, url: str = "http://127.0.0.1:8787/analyze", timeout_s: float = 5.0):
        self.url = url
        self.timeout_s = timeout_s

    def analyze(self, obs: Observation, ex: ExerciseProfile, calibration=None) -> AnalyzerOutput:
        body = json.dumps(request_payload(obs, ex, calibration)).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return parse_output(json.loads(resp.read().decode("utf-8")))
