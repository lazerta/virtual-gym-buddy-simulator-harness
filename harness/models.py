from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any

@dataclass(frozen=True)
class SubjectProfile:
    """Resolved subject fixture consumed by the simulator.

    SyntheticSubjectProfile is the versioned authoring contract. SubjectProfile is
    intentionally the small, resolved runtime view so older harness code does not
    need to understand clothing/rendering metadata unless a test needs it.
    """
    id: str
    source: str
    stature_scale: float = 1.0
    torso_ratio: float = 0.31
    upper_arm_ratio: float = 0.186
    forearm_ratio: float = 0.146
    thigh_ratio: float = 0.245
    shin_ratio: float = 0.246
    shoulder_width_ratio: float = 0.245
    hip_width_ratio: float = 0.185
    silhouette_scale: float = 1.0
    visual_variant: str = "neutral"
    clothing_variant: str = "neutral"
    accessory_variant: str = "none"
    body_state_variant: str = "baseline"
    rendering_variant: str = "neutral"
    source_confidence: float = 0.5
    synthetic_profile_version: int = 1
    # Visual-only variants share this identity so rendering/clothing changes do not
    # accidentally change the generator's underlying stochastic movement sample.
    simulation_identity: str | None = None

@dataclass(frozen=True)
class ExerciseProfile:
    id: str
    family: str
    preferred_yaw_deg: float
    yaw_tolerance_deg: float
    min_frame_fill: float
    max_frame_fill: float
    bilateral: bool = False

@dataclass(frozen=True)
class Scenario:
    id: str
    family: str
    seed: int
    issue_strength: float = 0.0
    camera_distance_m: float = 4.3
    camera_yaw_deg: float = 30.0
    camera_height_m: float = 1.45
    bystander_count: int = 0
    identity_jump: bool = False
    mirror_ambiguity: bool = False
    mirror_known_region: bool = False
    competitor_similarity: float = 0.10
    target_occlusion_fraction: float = 0.0
    occlusion_fraction: float = 0.0
    tracking_quality: float = 0.95
    tracking_gap_ms: int = 0
    camera_bump_deg: float = 0.0
    setup_motion: bool = False
    rep_count_gt: int = 10
    repeated_issue_reps: tuple[int, ...] = ()
    components: tuple[str, ...] = ()
    # Commercial-gym environment modifiers. These represent visual/sensor stress,
    # not RGB rendering. They are intentionally independent from form ground truth.
    visual_clutter: float = 0.0
    background_motion: float = 0.0
    equipment_occlusion_fraction: float = 0.0
    camera_micro_jitter: float = 0.0
    lighting_instability: float = 0.0

@dataclass(frozen=True)
class Observation:
    scenario_id: str
    frame_fill: float
    yaw_error_deg: float
    tracking_quality: float
    visible_required_fraction: float
    detected_people: int
    primary_target_score: float
    best_competitor_score: float
    identity_continuity: float
    mirror_ambiguity_score: float
    camera_motion_score: float
    rep_candidates: int
    issue_evidence_by_rep: tuple[float, ...]
    setup_motion_score: float
    camera_vertical_error_m: float
    tracking_gap_ms: int

    @property
    def identity_margin(self) -> float:
        return self.primary_target_score - self.best_competitor_score

@dataclass(frozen=True)
class ExpectedOutcome:
    ready: bool
    pause: bool
    reset_once: bool
    expected_reps: int
    cue_expected: bool
    cue_forbidden: bool
    reason: str

@dataclass
class AnalyzerOutput:
    ready: bool
    pause: bool
    reset_count: int
    rep_count: int
    cue_count: int
    reason: str

@dataclass
class CaseResult:
    case_id: str
    passed: bool
    expected: ExpectedOutcome
    observed: AnalyzerOutput
    metadata: dict[str, Any] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        d = {"case_id": self.case_id, "passed": self.passed, **self.metadata}
        d.update({f"exp_{k}": v for k, v in asdict(self.expected).items()})
        d.update({f"obs_{k}": v for k, v in asdict(self.observed).items()})
        return d
