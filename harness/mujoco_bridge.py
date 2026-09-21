from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from .mujoco_backend import MyoFullBodySimulator


def _ok(payload: Any) -> None:
    sys.stdout.write(json.dumps({"ok": True, "result": payload}) + "\n")
    sys.stdout.flush()


def _err(exc: Exception) -> None:
    sys.stdout.write(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
    sys.stdout.flush()


def serve() -> int:
    sim = MyoFullBodySimulator()
    _ok({"event": "ready", "nq": sim.nq, "nv": sim.nv, "nu": sim.nu})

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
            op = req.get("op")
            if op == "reset":
                _ok(sim.reset().to_dict())
            elif op == "state":
                _ok(sim.state().to_dict())
            elif op == "step":
                ctrl = req.get("ctrl")
                nstep = int(req.get("nstep", 1))
                _ok(sim.step(ctrl, nstep=nstep).to_dict())
            elif op == "set_qpos":
                _ok(sim.set_qpos(req["qpos"], bool(req.get("zero_velocity", True))).to_dict())
            elif op == "zero_ctrl":
                _ok(sim.step(np.zeros(sim.nu), nstep=int(req.get("nstep", 1))).to_dict())
            elif op == "close":
                _ok({"event": "closed"})
                return 0
            else:
                raise ValueError(f"unknown op: {op!r}")
        except Exception as exc:
            _err(exc)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gym-buddy-mujoco-bridge")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    if args.smoke:
        sim = MyoFullBodySimulator()
        state = sim.reset()
        print(json.dumps({
            "ok": True,
            "backend": "mujoco+myo-sim",
            "nq": sim.nq,
            "nv": sim.nv,
            "nu": sim.nu,
            "bodies": len(state.body_positions),
            "sites": len(state.site_positions),
            "contacts": len(state.contacts),
        }, indent=2))
        return 0

    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
