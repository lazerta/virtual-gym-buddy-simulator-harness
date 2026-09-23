from __future__ import annotations

import json
import os
from pathlib import Path

from .models import SubjectProfile, ExerciseProfile
from .synthetic_subjects import (
    BodyShapeEnvelope,
    NormalizedSkeletonProportions,
    SyntheticSubjectProfile,
    SyntheticSubjectVariant,
)

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


def _single_variant_profile(
    profile_id: str,
    *,
    source: str,
    scale: float,
    torso: float,
    upper_arm: float,
    forearm: float,
    thigh: float,
    shin: float,
    shoulder_width: float,
    hip_width: float,
) -> SyntheticSubjectProfile:
    return SyntheticSubjectProfile(
        synthetic_subject_profile_id=profile_id,
        version=1,
        normalized_skeleton_proportions=NormalizedSkeletonProportions(
            body_scale_factor=scale,
            torso_ratio=torso,
            upper_arm_ratio=upper_arm,
            forearm_ratio=forearm,
            thigh_ratio=thigh,
            shin_ratio=shin,
            shoulder_width_ratio=shoulder_width,
            hip_width_ratio=hip_width,
        ),
        body_shape_envelope=BodyShapeEnvelope(scale, scale),
        variants=(
            SyntheticSubjectVariant(
                "reference",
                silhouette_scale=scale,
                visual_variant="neutral",
            ),
        ),
        source=source,
        source_confidence=.60,
        source_notes=(
            "anonymous normalized anthropometric stress fixture",
            "not a production user record",
        ),
    )


def external_synthetic_profiles() -> list[SyntheticSubjectProfile]:
    """Anonymous morphology stress fixtures around published anthropometric ranges."""
    anchors = [
        ("small", .92, .305, .180, .142, .238, .242, .238, .181),
        ("lower_mid", .96, .308, .183, .144, .241, .244, .241, .183),
        ("median", 1.00, .310, .186, .146, .245, .246, .245, .185),
        ("upper_mid", 1.04, .312, .189, .148, .248, .248, .249, .188),
        ("large", 1.08, .314, .192, .150, .252, .250, .253, .191),
        ("very_large", 1.12, .316, .195, .152, .255, .252, .257, .194),
    ]
    out: list[SyntheticSubjectProfile] = []
    for name, scale, torso, ua, fa, th, sh, sw, hw in anchors:
        out.append(
            _single_variant_profile(
                f"ansur_{name}_central",
                source="ANSUR-II-derived-stress",
                scale=scale,
                torso=torso,
                upper_arm=ua,
                forearm=fa,
                thigh=th,
                shin=sh,
                shoulder_width=sw,
                hip_width=hw,
            )
        )
        out.append(
            _single_variant_profile(
                f"ansur_{name}_long_limb",
                source="ANSUR-II-derived-stress",
                scale=scale,
                torso=torso * .94,
                upper_arm=ua * 1.045,
                forearm=fa * 1.045,
                thigh=th * 1.035,
                shin=sh * 1.035,
                shoulder_width=sw,
                hip_width=hw,
            )
        )
        out.append(
            _single_variant_profile(
                f"ansur_{name}_long_torso",
                source="ANSUR-II-derived-stress",
                scale=scale,
                torso=torso * 1.06,
                upper_arm=ua * .965,
                forearm=fa * .965,
                thigh=th * .975,
                shin=sh * .975,
                shoulder_width=sw,
                hip_width=hw,
            )
        )
    return out


def external_subjects() -> list[SubjectProfile]:
    return [profile.resolve() for profile in external_synthetic_profiles()]


def _appearance_variants(
    base_silhouette_scale: float = 1.0,
    reference_visual: str = "neutral",
) -> tuple[SyntheticSubjectVariant, ...]:
    return (
        SyntheticSubjectVariant(
            "reference",
            silhouette_scale=base_silhouette_scale,
            visual_variant=reference_visual,
        ),
        SyntheticSubjectVariant(
            "loose_clothes",
            silhouette_scale=base_silhouette_scale * 1.06,
            visual_variant="loose_clothes",
            clothing_variant="loose",
        ),
        SyntheticSubjectVariant(
            "fitted",
            silhouette_scale=base_silhouette_scale * .98,
            visual_variant="fitted",
            clothing_variant="fitted",
        ),
        SyntheticSubjectVariant(
            "leaner_stress",
            silhouette_scale=base_silhouette_scale * .94,
            visual_variant="leaner_stress",
            body_state_variant="shape_envelope_low",
        ),
        SyntheticSubjectVariant(
            "heavier_stress",
            silhouette_scale=base_silhouette_scale * 1.08,
            visual_variant="heavier_stress",
            body_state_variant="shape_envelope_high",
        ),
        SyntheticSubjectVariant(
            "appearance_variant",
            silhouette_scale=base_silhouette_scale,
            visual_variant="appearance_variant",
            accessory_variant="generic_accessory",
            rendering_variant="appearance_stress",
        ),
    )


def _personal_synthetic_profile(
    *,
    profile_id: str,
    source: str,
    version: int,
    source_confidence: float,
    source_notes: tuple[str, ...],
    stature_scale: float,
    torso_ratio: float,
    upper_arm_ratio: float,
    forearm_ratio: float,
    thigh_ratio: float,
    shin_ratio: float,
    shoulder_width_ratio: float,
    hip_width_ratio: float,
    silhouette_scale: float,
    reference_visual: str,
) -> SyntheticSubjectProfile:
    variants = _appearance_variants(silhouette_scale, reference_visual)
    scales = [x.silhouette_scale for x in variants]
    return SyntheticSubjectProfile(
        synthetic_subject_profile_id=profile_id,
        version=version,
        normalized_skeleton_proportions=NormalizedSkeletonProportions(
            body_scale_factor=stature_scale,
            torso_ratio=torso_ratio,
            upper_arm_ratio=upper_arm_ratio,
            forearm_ratio=forearm_ratio,
            thigh_ratio=thigh_ratio,
            shin_ratio=shin_ratio,
            shoulder_width_ratio=shoulder_width_ratio,
            hip_width_ratio=hip_width_ratio,
        ),
        body_shape_envelope=BodyShapeEnvelope(min(scales), max(scales)),
        variants=variants,
        source=source,
        source_confidence=source_confidence,
        source_notes=source_notes,
    )


SYNTHETIC_PERSONAL_PROFILE = _personal_synthetic_profile(
    profile_id="personal_synthetic_reference",
    source="nonpersonal-public-fixture",
    version=1,
    source_confidence=.50,
    source_notes=(
        "public non-personal fallback",
        "normalized approximation only",
    ),
    stature_scale=1.0,
    torso_ratio=.310,
    upper_arm_ratio=.186,
    forearm_ratio=.146,
    thigh_ratio=.245,
    shin_ratio=.246,
    shoulder_width_ratio=.245,
    hip_width_ratio=.185,
    silhouette_scale=1.0,
    reference_visual="neutral",
)
SYNTHETIC_PERSONAL_REFERENCE = SYNTHETIC_PERSONAL_PROFILE.resolve()


def _source_notes(value) -> tuple[str, ...]:
    if value is None:
        return ("local normalized approximation",)
    if isinstance(value, str):
        return (value,)
    return tuple(str(x) for x in value)


def load_local_synthetic_subject_profile(
    path: str | os.PathLike[str] | None = None,
) -> SyntheticSubjectProfile | None:
    """Load a private local normalized profile from an ignored JSON file."""
    value = path or os.environ.get("GYM_BUDDY_PERSONAL_PROFILE")
    if not value:
        return None
    p = Path(value).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"GYM_BUDDY_PERSONAL_PROFILE does not exist: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return _personal_synthetic_profile(
        profile_id=str(data.get("id", "local_personal")),
        source="local-private-profile",
        version=int(data.get("version", 1)),
        source_confidence=float(data.get("source_confidence", .50)),
        source_notes=_source_notes(data.get("source_notes")),
        stature_scale=float(data.get("stature_scale", 1.0)),
        torso_ratio=float(data.get("torso_ratio", .310)),
        upper_arm_ratio=float(data.get("upper_arm_ratio", .186)),
        forearm_ratio=float(data.get("forearm_ratio", .146)),
        thigh_ratio=float(data.get("thigh_ratio", .245)),
        shin_ratio=float(data.get("shin_ratio", .246)),
        shoulder_width_ratio=float(data.get("shoulder_width_ratio", .245)),
        hip_width_ratio=float(data.get("hip_width_ratio", .185)),
        silhouette_scale=float(data.get("silhouette_scale", 1.0)),
        reference_visual=str(data.get("visual_variant", "current_like")),
    )


def load_local_personal_profile(
    path: str | os.PathLike[str] | None = None,
) -> SubjectProfile | None:
    """Backward-compatible resolved view of the private local synthetic profile."""
    profile = load_local_synthetic_subject_profile(path)
    return profile.resolve() if profile is not None else None


def personal_test_profiles() -> list[SubjectProfile]:
    profile = load_local_synthetic_subject_profile() or SYNTHETIC_PERSONAL_PROFILE
    return profile.resolve_all()


PERSONAL_VARIANTS = personal_test_profiles()
PERSONAL_REFERENCE = PERSONAL_VARIANTS[0]
