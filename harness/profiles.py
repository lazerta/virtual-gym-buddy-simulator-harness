from __future__ import annotations

import json
import os
from pathlib import Path

from .models import SubjectProfile, ExerciseProfile

EXERCISES = [
    ExerciseProfile("incline_db_press", "press_db", 32, 20, 0.42, 0.86, True),
    ExerciseProfile("flat_db_press", "press_db", 35, 22, 0.42, 0.86, True),
    ExerciseProfile("barbell_bench", "press_bar", 35, 20, 0.43, 0.88, False),
    ExerciseProfile("incline_smith_press", "press_smith", 30, 18, 0.42, 0.86, False),
    ExerciseProfile("seated_ohp", "press_machine", 25, 20, 0.44, 0.88, True),
    ExerciseProfile("smith_squat", "squat", 78, 16, 0.55, 0.94, False),
    ExerciseProfile("leg_press", "leg_press", 55, 25, 0.48, 0.92, False),
    ExerciseProfile("lateral_raise", "raise", 5, 18, 0.55, 0.93, True),
    ExerciseProfile("lat_pulldown", "vertical_pull", 8, 22, 0.52, 0.92, True),
    ExerciseProfile("chest_supported_t_row", "row", 45, 25, 0.48, 0.90, True),
]


def external_subjects() -> list[SubjectProfile]:
    """Anonymous morphology stress fixtures around published anthropometric ranges."""
    anchors = [
        ("small", .92, .305, .180, .142, .238, .242, .238, .181),
        ("lower_mid", .96, .308, .183, .144, .241, .244, .241, .183),
        ("median", 1.00, .310, .186, .146, .245, .246, .245, .185),
        ("upper_mid", 1.04, .312, .189, .148, .248, .248, .249, .188),
        ("large", 1.08, .314, .192, .150, .252, .250, .253, .191),
        ("very_large", 1.12, .316, .195, .152, .255, .252, .257, .194),
    ]
    out=[]
    for name,scale,torso,ua,fa,th,sh,sw,hw in anchors:
        out.append(SubjectProfile(f"ansur_{name}_central","ANSUR-II-derived-stress",scale,torso,ua,fa,th,sh,sw,hw,scale))
        out.append(SubjectProfile(f"ansur_{name}_long_limb","ANSUR-II-derived-stress",scale,torso*.94,ua*1.045,fa*1.045,th*1.035,sh*1.035,sw,hw,scale))
        out.append(SubjectProfile(f"ansur_{name}_long_torso","ANSUR-II-derived-stress",scale,torso*1.06,ua*.965,fa*.965,th*.975,sh*.975,sw,hw,scale))
    return out


# Public repository fallback. These are generic synthetic values, not Shawn's data.
SYNTHETIC_PERSONAL_REFERENCE = SubjectProfile(
    id="personal_synthetic_reference",
    source="nonpersonal-public-fixture",
    stature_scale=1.0,
    torso_ratio=.310,
    upper_arm_ratio=.186,
    forearm_ratio=.146,
    thigh_ratio=.245,
    shin_ratio=.246,
    shoulder_width_ratio=.245,
    hip_width_ratio=.185,
    silhouette_scale=1.0,
    visual_variant="neutral",
)


def load_local_personal_profile(path: str | os.PathLike[str] | None = None) -> SubjectProfile | None:
    """Load a private local profile. The referenced file is intentionally gitignored."""
    value = path or os.environ.get("GYM_BUDDY_PERSONAL_PROFILE")
    if not value:
        return None
    p=Path(value).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"GYM_BUDDY_PERSONAL_PROFILE does not exist: {p}")
    data=json.loads(p.read_text(encoding="utf-8"))
    fields={
        "id": str(data.get("id","local_personal")),
        "source": "local-private-profile",
        "stature_scale": float(data.get("stature_scale",1.0)),
        "torso_ratio": float(data.get("torso_ratio",.310)),
        "upper_arm_ratio": float(data.get("upper_arm_ratio",.186)),
        "forearm_ratio": float(data.get("forearm_ratio",.146)),
        "thigh_ratio": float(data.get("thigh_ratio",.245)),
        "shin_ratio": float(data.get("shin_ratio",.246)),
        "shoulder_width_ratio": float(data.get("shoulder_width_ratio",.245)),
        "hip_width_ratio": float(data.get("hip_width_ratio",.185)),
        "silhouette_scale": float(data.get("silhouette_scale",1.0)),
        "visual_variant": str(data.get("visual_variant","current_like")),
    }
    return SubjectProfile(**fields)


def personal_test_profiles() -> list[SubjectProfile]:
    base=load_local_personal_profile() or SYNTHETIC_PERSONAL_REFERENCE
    return [
        base,
        SubjectProfile(**{**base.__dict__, "id":f"{base.id}_loose_clothes", "silhouette_scale":base.silhouette_scale*1.06, "visual_variant":"loose_clothes"}),
        SubjectProfile(**{**base.__dict__, "id":f"{base.id}_fitted", "silhouette_scale":base.silhouette_scale*.98, "visual_variant":"fitted"}),
        SubjectProfile(**{**base.__dict__, "id":f"{base.id}_leaner_stress", "silhouette_scale":base.silhouette_scale*.94, "visual_variant":"leaner_stress"}),
        SubjectProfile(**{**base.__dict__, "id":f"{base.id}_heavier_stress", "silhouette_scale":base.silhouette_scale*1.08, "visual_variant":"heavier_stress"}),
        SubjectProfile(**{**base.__dict__, "id":f"{base.id}_appearance_variant", "visual_variant":"appearance_variant"}),
    ]


PERSONAL_VARIANTS = personal_test_profiles()
PERSONAL_REFERENCE = PERSONAL_VARIANTS[0]

