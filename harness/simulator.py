from __future__ import annotations
import hashlib
import numpy as np
from .models import SubjectProfile, ExerciseProfile, Scenario, Observation

def _stable_id_seed(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=4).digest(), "little")


def simulate(subject: SubjectProfile, exercise: ExerciseProfile, s: Scenario) -> Observation:
    simulation_identity=subject.simulation_identity or subject.id
    rng=np.random.default_rng(s.seed ^ _stable_id_seed(simulation_identity) ^ _stable_id_seed(exercise.id))
    physical_scale=subject.stature_scale * (0.97 + 0.06*subject.silhouette_scale)
    family_factor={"squat":1.08,"raise":1.04,"vertical_pull":1.06,"leg_press":.82,"press_db":.76,"press_bar":.75,"press_smith":.76,"press_machine":.82,"row":.80}.get(exercise.family,.9)
    comps=set(s.components or (s.family,))
    if comps & {"too_close","too_far"}:
        effective_distance=s.camera_distance_m
    else:
        target_fill=(exercise.min_frame_fill+exercise.max_frame_fill)/2
        effective_distance=(3.55*physical_scale*family_factor)/target_fill
    frame_fill=float(np.clip((3.55/effective_distance)*physical_scale*family_factor,0.05,1.35))
    height_penalty=max(0,abs(s.camera_height_m-1.45)-.65)*.22
    frame_fill=max(.02, frame_fill-height_penalty)
    effective_yaw = s.camera_yaw_deg if "wrong_view" in comps else exercise.preferred_yaw_deg
    yaw_error=abs(((effective_yaw-exercise.preferred_yaw_deg+180)%360)-180)

    visible=max(0.,1.0-s.occlusion_fraction-s.target_occlusion_fraction-s.equipment_occlusion_fraction-height_penalty-rng.normal(0,.015))
    detected_people=1+s.bystander_count
    primary=float(np.clip(.965-rng.normal(0,.018),0,1))
    continuity=float(np.clip(.975-rng.normal(0,.018),0,1))
    competitor=float(np.clip(s.competitor_similarity+.045*s.background_motion+.025*s.visual_clutter+rng.normal(0,.018),0,1))
    if s.identity_jump:
        primary=float(np.clip(.42+rng.normal(0,.035),0,1))
        continuity=float(np.clip(.20+rng.normal(0,.03),0,1))
        competitor=float(np.clip(max(competitor,.86)+rng.normal(0,.02),0,1))

    if s.mirror_ambiguity:
        mirror=float(np.clip(.90+rng.normal(0,.02),0,1)); primary=min(primary,.76); competitor=max(competitor,.82)
    elif s.mirror_known_region:
        mirror=float(np.clip(.16+rng.normal(0,.025),0,1)); competitor=min(competitor,.34)
    else:
        mirror=float(np.clip(rng.normal(.05,.025),0,1))

    camera_motion=float(np.clip(s.camera_bump_deg/15+s.camera_micro_jitter+abs(rng.normal(0,.02)),0,1))
    tracking=float(np.clip(s.tracking_quality-.24*(s.occlusion_fraction+s.target_occlusion_fraction)-.18*s.equipment_occlusion_fraction-.055*s.visual_clutter-.050*s.background_motion-.080*s.lighting_instability+rng.normal(0,.015),0,1))
    reps=s.rep_count_gt

    evid=[]
    repeated=set(s.repeated_issue_reps)
    for i in range(1,max(10,s.rep_count_gt)+1):
        base=s.issue_strength if i in repeated else rng.normal(.08,.025)
        morph_bias=(subject.torso_ratio-.31)*2.8 + (subject.upper_arm_ratio-.186)*3.2
        evid.append(float(np.clip(base+morph_bias+rng.normal(0,.025),0,1)))

    return Observation(
        scenario_id=s.id, frame_fill=frame_fill, yaw_error_deg=yaw_error,
        tracking_quality=tracking, visible_required_fraction=visible,
        detected_people=detected_people, primary_target_score=primary,
        best_competitor_score=competitor, identity_continuity=continuity,
        mirror_ambiguity_score=mirror, camera_motion_score=camera_motion,
        rep_candidates=reps, issue_evidence_by_rep=tuple(evid),
        setup_motion_score=.9 if s.setup_motion else .05,
        camera_vertical_error_m=abs(s.camera_height_m-1.45),
        tracking_gap_ms=s.tracking_gap_ms
    )
