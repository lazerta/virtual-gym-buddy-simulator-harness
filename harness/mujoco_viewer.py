from __future__ import annotations

import argparse
import time
from pathlib import Path

from .motion_registry import load_exercise_motion
from .motion_sources import load_musclemimic_npz
from .mujoco_backend import MyoFullBodySimulator


def _load_motion(args):
    if args.motion:
        return load_musclemimic_npz(args.motion, trajectory_index=args.trajectory_index)
    if args.exercise:
        return load_exercise_motion(
            args.exercise,
            root=args.motion_root,
            trajectory_index=args.trajectory_index,
        )
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gym-buddy-sim")
    p.add_argument("--motion", default=None, help="MuscleMimic/LocoMuJoCo trajectory NPZ")
    p.add_argument("--exercise", default=None, help="canonical exercise id from motion registry")
    p.add_argument("--motion-root", default=None, help="override GYM_BUDDY_MOTION_ROOT")
    p.add_argument("--trajectory-index", type=int, default=0)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--dynamic", action="store_true", help="step zero-control dynamics when no motion is supplied")
    p.add_argument("--once", action="store_true", help="play a motion once instead of looping")
    args = p.parse_args(argv)

    if args.speed <= 0:
        p.error("--speed must be > 0")

    import mujoco
    import mujoco.viewer

    sim = MyoFullBodySimulator()
    motion = _load_motion(args)

    if motion is not None and motion.qpos.shape[1] not in (sim.human_nq, sim.nq):
        raise SystemExit(
            f"Motion qpos width {motion.qpos.shape[1]} does not match "
            f"MyoFullBody human_nq={sim.human_nq} or full nq={sim.nq}"
        )

    print(
        f"MuJoCo MyoFullBody viewer: nq={sim.nq} human_nq={sim.human_nq} "
        f"nu={sim.nu} motion={motion.source if motion else 'none'}"
    )

    sim.reset()
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        if motion is None:
            while viewer.is_running():
                start = time.perf_counter()
                if args.dynamic:
                    sim.step([0.0] * sim.nu)
                else:
                    mujoco.mj_forward(sim.model, sim.data)
                viewer.sync()
                remaining = sim.model.opt.timestep - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)
            return 0

        frame_dt = 1.0 / (motion.fps * args.speed)
        while viewer.is_running():
            for frame in range(motion.frames):
                if not viewer.is_running():
                    return 0
                started = time.perf_counter()
                qpos = motion.frame(frame)
                if qpos.shape == (sim.human_nq,):
                    sim.set_human_qpos(qpos, reset_equipment=(frame == 0))
                else:
                    sim.set_qpos(qpos)
                viewer.sync()
                remaining = frame_dt - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
            if args.once:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)
                return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
