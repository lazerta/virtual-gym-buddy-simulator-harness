from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run MuscleMimic GMR retargeting for one AMASS-compatible SMPL-H file."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-fps", type=float, default=30.0)
    args = parser.parse_args(argv)

    import numpy as np
    from loco_mujoco.smpl import retargeting as r

    input_path = str(Path(args.input).resolve())
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    robot_conf = r.load_robot_conf_file("MyoFullBody")
    logger = r.setup_logger(
        "gym_buddy_video",
        identifier="[Gym Buddy WHAM -> MuscleMimic GMR]",
    )

    trajectory, _analysis = r.fit_gmr_motion(
        env_name="MyoFullBody",
        robot_conf=robot_conf,
        motion_data=input_path,
        logger=logger,
        gmr_config={
            "src_human": "smplh",
            "target_fps": float(args.target_fps),
            "offset_to_ground": True,
            "use_fitted_shape": True,
        },
    )
    trajectory.save(str(output_path))

    summary = {
        "output": str(output_path),
        "frames": int(np.asarray(trajectory.data.qpos).shape[0]),
        "qpos_width": int(np.asarray(trajectory.data.qpos).shape[1]),
        "frequency": float(trajectory.info.frequency),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
