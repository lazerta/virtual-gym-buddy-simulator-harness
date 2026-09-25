from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from .models import SubjectProfile


def _finite_between(name: str, value: float, low: float, high: float) -> None:
    if not isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}], got {value}")


@dataclass(frozen=True)
class NormalizedSkeletonProportions:
    """Normalized movement-relevant geometry; no absolute body dimensions."""

    body_scale_factor: float = 1.0
    torso_ratio: float = 0.310
    upper_arm_ratio: float = 0.186
    forearm_ratio: float = 0.146
    thigh_ratio: float = 0.245
    shin_ratio: float = 0.246
    shoulder_width_ratio: float = 0.245
    hip_width_ratio: float = 0.185

    def __post_init__(self) -> None:
        _finite_between("body_scale_factor", self.body_scale_factor, 0.5, 1.5)
        for name in (
            "torso_ratio",
            "upper_arm_ratio",
            "forearm_ratio",
            "thigh_ratio",
            "shin_ratio",
            "shoulder_width_ratio",
            "hip_width_ratio",
        ):
            _finite_between(name, getattr(self, name), 0.05, 0.60)


@dataclass(frozen=True)
class BodyShapeEnvelope:
    """Allowed synthetic silhouette stress range, not a body-fat estimate."""

    min_silhouette_scale: float = 0.90
    max_silhouette_scale: float = 1.10

    def __post_init__(self) -> None:
        _finite_between("min_silhouette_scale", self.min_silhouette_scale, 0.5, 1.5)
        _finite_between("max_silhouette_scale", self.max_silhouette_scale, 0.5, 1.5)
        if self.min_silhouette_scale > self.max_silhouette_scale:
            raise ValueError("min_silhouette_scale must be <= max_silhouette_scale")

    def contains(self, value: float) -> bool:
        return self.min_silhouette_scale <= value <= self.max_silhouette_scale


@dataclass(frozen=True)
class SyntheticSubjectVariant:
    variant_id: str
    silhouette_scale: float = 1.0
    visual_variant: str = "neutral"
    clothing_variant: str = "neutral"
    accessory_variant: str = "none"
    body_state_variant: str = "baseline"
    rendering_variant: str = "neutral"

    def __post_init__(self) -> None:
        if not self.variant_id.strip():
            raise ValueError("variant_id must be non-empty")
        _finite_between("silhouette_scale", self.silhouette_scale, 0.5, 1.5)


@dataclass(frozen=True)
class SyntheticSubjectProfile:
    """Versioned harness-only subject contract.

    It deliberately contains normalized geometry and synthetic appearance labels
    only. It is not a production user profile and does not encode an assumed exact
    height, body-fat percentage, or circumference.
    """

    synthetic_subject_profile_id: str
    version: int
    normalized_skeleton_proportions: NormalizedSkeletonProportions
    body_shape_envelope: BodyShapeEnvelope = field(default_factory=BodyShapeEnvelope)
    variants: tuple[SyntheticSubjectVariant, ...] = field(
        default_factory=lambda: (SyntheticSubjectVariant("reference"),)
    )
    source: str = "synthetic"
    source_confidence: float = 0.5
    source_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.synthetic_subject_profile_id.strip():
            raise ValueError("synthetic_subject_profile_id must be non-empty")
        if self.version < 1:
            raise ValueError("version must be >= 1")
        _finite_between("source_confidence", self.source_confidence, 0.0, 1.0)
        if not self.variants:
            raise ValueError("at least one synthetic subject variant is required")
        ids = [x.variant_id for x in self.variants]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate synthetic subject variant ids: {ids}")
        for variant in self.variants:
            if not self.body_shape_envelope.contains(variant.silhouette_scale):
                raise ValueError(
                    f"{variant.variant_id} silhouette_scale={variant.silhouette_scale} "
                    f"outside body shape envelope {self.body_shape_envelope}"
                )

    @property
    def visual_variants(self) -> tuple[str, ...]:
        return tuple(x.visual_variant for x in self.variants)

    @property
    def clothing_variants(self) -> tuple[str, ...]:
        return tuple(x.clothing_variant for x in self.variants)

    @property
    def accessory_variants(self) -> tuple[str, ...]:
        return tuple(x.accessory_variant for x in self.variants)

    @property
    def body_state_variants(self) -> tuple[str, ...]:
        return tuple(x.body_state_variant for x in self.variants)

    @property
    def rendering_variants(self) -> tuple[str, ...]:
        return tuple(x.rendering_variant for x in self.variants)

    def resolve(self, variant_id: str = "reference") -> SubjectProfile:
        variant = next((x for x in self.variants if x.variant_id == variant_id), None)
        if variant is None:
            raise KeyError(
                f"unknown variant {variant_id!r} for {self.synthetic_subject_profile_id}"
            )
        p = self.normalized_skeleton_proportions
        subject_id = (
            self.synthetic_subject_profile_id
            if variant.variant_id == "reference"
            else f"{self.synthetic_subject_profile_id}_{variant.variant_id}"
        )
        return SubjectProfile(
            id=subject_id,
            source=self.source,
            stature_scale=p.body_scale_factor,
            torso_ratio=p.torso_ratio,
            upper_arm_ratio=p.upper_arm_ratio,
            forearm_ratio=p.forearm_ratio,
            thigh_ratio=p.thigh_ratio,
            shin_ratio=p.shin_ratio,
            shoulder_width_ratio=p.shoulder_width_ratio,
            hip_width_ratio=p.hip_width_ratio,
            silhouette_scale=variant.silhouette_scale,
            visual_variant=variant.visual_variant,
            clothing_variant=variant.clothing_variant,
            accessory_variant=variant.accessory_variant,
            body_state_variant=variant.body_state_variant,
            rendering_variant=variant.rendering_variant,
            source_confidence=self.source_confidence,
            synthetic_profile_version=self.version,
            simulation_identity=self.synthetic_subject_profile_id,
        )

    def resolve_all(self) -> list[SubjectProfile]:
        return [self.resolve(x.variant_id) for x in self.variants]
