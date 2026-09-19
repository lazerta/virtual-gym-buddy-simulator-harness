from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable

import numpy as np
import pandas as pd

from .models import ExerciseProfile, SubjectProfile
from .profiles import EXERCISES
from .rep_detection import MovementFrame, REP_CASES, detect_reps, generate_rep_case


def _stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little") & 0x7FFFFFFF


@dataclass(frozen=True)
class KinematicFrame:
    frame_index: int
    t_s: float
    valid: bool
    assistance_evidence: float
    truth_depth: float
    observed_depth: float
    channels: dict[str, float]
    variant: str


@dataclass(frozen=True)
class KinematicVariant:
    id: str
    params: dict[str, float | str]
    provenance: str


class ExerciseKinematicModel:
    exercise_id: str
    variants: tuple[KinematicVariant, ...]

    def render(
        self,
        frames: list[MovementFrame],
        *,
        subject: SubjectProfile,
        variant: KinematicVariant,
        seed: int,
    ) -> list[KinematicFrame]:
        raise NotImplementedError


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _phase_curve(depth: float) -> float:
    d = _clip01(depth)
    return float(d * d * (3.0 - 2.0 * d))


def _inverse_phase_curve(value: float) -> float:
    y=_clip01(value)
    lo,hi=0.0,1.0
    for _ in range(22):
        mid=(lo+hi)/2
        if _phase_curve(mid) < y:
            lo=mid
        else:
            hi=mid
    return (lo+hi)/2


class InclineDBPressModel(ExerciseKinematicModel):
    exercise_id = "incline_db_press"
    variants = (
        KinematicVariant("bench20_neutral", {"bench_angle": 20.0, "elbow_path": 0.0, "arm_lag": 0.00}, "SOURCE_ANCHORED"),
        KinematicVariant("bench30_neutral", {"bench_angle": 30.0, "elbow_path": 0.0, "arm_lag": 0.00}, "SOURCE_ANCHORED"),
        KinematicVariant("bench45_neutral", {"bench_angle": 45.0, "elbow_path": 0.0, "arm_lag": 0.00}, "SOURCE_ANCHORED"),
        KinematicVariant("mild_arm_lag", {"bench_angle": 30.0, "elbow_path": 0.0, "arm_lag": 0.055}, "SYNTHETIC_STRESS"),
        KinematicVariant("narrower_elbow_path", {"bench_angle": 30.0, "elbow_path": -10.0, "arm_lag": 0.02}, "SYNTHETIC_STRESS"),
        KinematicVariant("wider_elbow_path", {"bench_angle": 30.0, "elbow_path": 10.0, "arm_lag": 0.02}, "SYNTHETIC_STRESS"),
    )

    def render(self, frames, *, subject, variant, seed):
        rng = np.random.default_rng(_stable_seed("idbp", subject.id, variant.id, seed))
        bench = float(variant.params["bench_angle"])
        path = float(variant.params["elbow_path"])
        lag = float(variant.params["arm_lag"])
        top_elbow = float(rng.uniform(8, 20))
        bottom_elbow = float(rng.uniform(88, 112))
        top_sh_abd = float(rng.uniform(18, 34) + 0.10 * (bench - 30))
        bottom_sh_abd = float(np.clip(rng.uniform(42, 62) + path, 25, 75))
        top_sh_flex = float(rng.uniform(18, 35) + 0.65 * bench)
        bottom_sh_flex = float(rng.uniform(30, 50) + 0.55 * bench)
        lateral_spread = float(rng.uniform(.10, .18) * subject.shoulder_width_ratio / .245)

        out=[]
        for f in frames:
            d=_clip01(f.depth)
            dl=_clip01(d + lag/2)
            dr=_clip01(d - lag/2)
            cl=_phase_curve(dl); cr=_phase_curve(dr)
            elbow_l=top_elbow+(bottom_elbow-top_elbow)*cl
            elbow_r=top_elbow+(bottom_elbow-top_elbow)*cr
            sh_abd_l=top_sh_abd+(bottom_sh_abd-top_sh_abd)*cl
            sh_abd_r=top_sh_abd+(bottom_sh_abd-top_sh_abd)*cr
            sh_flex_l=top_sh_flex+(bottom_sh_flex-top_sh_flex)*cl
            sh_flex_r=top_sh_flex+(bottom_sh_flex-top_sh_flex)*cr
            dbz_l=1.0-dl; dbz_r=1.0-dr
            spread_l=lateral_spread*dl; spread_r=lateral_spread*dr
            norm_el_l=(elbow_l-top_elbow)/max(1e-6,bottom_elbow-top_elbow)
            norm_el_r=(elbow_r-top_elbow)/max(1e-6,bottom_elbow-top_elbow)
            observed=float(np.median([_inverse_phase_curve(norm_el_l),_inverse_phase_curve(norm_el_r),1-dbz_l,1-dbz_r]))
            if f.valid:
                observed=float(np.clip(observed+rng.normal(0,.008),0,1.06))
            out.append(KinematicFrame(f.frame_index,f.t_s,f.valid,f.assistance_evidence,d,observed,{
                "bench_angle_deg":bench,
                "left_elbow_flex_deg":elbow_l,"right_elbow_flex_deg":elbow_r,
                "left_shoulder_abduction_deg":sh_abd_l,"right_shoulder_abduction_deg":sh_abd_r,
                "left_shoulder_flexion_deg":sh_flex_l,"right_shoulder_flexion_deg":sh_flex_r,
                "left_db_height_norm":dbz_l,"right_db_height_norm":dbz_r,
                "left_db_lateral_norm":spread_l,"right_db_lateral_norm":spread_r,
            },variant.id))
        return out


class SmithSquatModel(ExerciseKinematicModel):
    exercise_id = "smith_squat"
    variants = (
        KinematicVariant("foot0_guide0", {"foot_offset_h":0.00,"guide_slope_deg":0.0}, "SOURCE_ANCHORED"),
        KinematicVariant("foot014_guide0", {"foot_offset_h":0.14,"guide_slope_deg":0.0}, "SOURCE_ANCHORED"),
        KinematicVariant("foot028_guide0", {"foot_offset_h":0.28,"guide_slope_deg":0.0}, "SOURCE_ANCHORED"),
        KinematicVariant("foot0_guide10", {"foot_offset_h":0.00,"guide_slope_deg":10.0}, "SOURCE_ANCHORED"),
        KinematicVariant("foot014_guide10", {"foot_offset_h":0.14,"guide_slope_deg":10.0}, "SOURCE_ANCHORED"),
        KinematicVariant("foot028_guide20", {"foot_offset_h":0.28,"guide_slope_deg":20.0}, "SOURCE_ANCHORED"),
    )

    def render(self, frames, *, subject, variant, seed):
        rng=np.random.default_rng(_stable_seed("smith",subject.id,variant.id,seed))
        foot=float(variant.params["foot_offset_h"]); slope=float(variant.params["guide_slope_deg"])
        knee_bottom=float(np.clip(rng.normal(124,5) - 18*foot + .15*slope,108,136))
        hip_bottom=float(np.clip(rng.normal(105,8) + 42*foot - .35*slope,78,132))
        ankle_bottom=float(np.clip(rng.normal(22,4) - 20*foot + .18*slope,7,34))
        trunk_bottom=float(np.clip(rng.normal(18,5) + 42*foot - .40*slope,4,42))
        asym=float(rng.normal(0,1.7))
        out=[]
        for f in frames:
            d=_clip01(f.depth); c=_phase_curve(d)
            knee_l=knee_bottom*c+asym*c; knee_r=knee_bottom*c-asym*c
            hip_l=hip_bottom*c+0.8*asym*c; hip_r=hip_bottom*c-0.8*asym*c
            ankle_l=ankle_bottom*c+0.3*asym*c; ankle_r=ankle_bottom*c-0.3*asym*c
            trunk=trunk_bottom*c
            bar=1.0-d
            vals=[_inverse_phase_curve(knee_l/knee_bottom),_inverse_phase_curve(knee_r/knee_bottom),_inverse_phase_curve(hip_l/hip_bottom),_inverse_phase_curve(hip_r/hip_bottom),1-bar]
            observed=float(np.median(vals))
            if f.valid: observed=float(np.clip(observed+rng.normal(0,.007),0,1.06))
            out.append(KinematicFrame(f.frame_index,f.t_s,f.valid,f.assistance_evidence,d,observed,{
                "guide_slope_deg":slope,"foot_offset_height_ratio":foot,
                "left_knee_flex_deg":knee_l,"right_knee_flex_deg":knee_r,
                "left_hip_flex_deg":hip_l,"right_hip_flex_deg":hip_r,
                "left_ankle_dorsi_deg":ankle_l,"right_ankle_dorsi_deg":ankle_r,
                "trunk_forward_lean_deg":trunk,"bar_height_norm":bar,
            },variant.id))
        return out


class LateralRaiseModel(ExerciseKinematicModel):
    exercise_id = "lateral_raise"
    variants = (
        KinematicVariant("neutral_90", {"target_elev":90.0,"plane_deg":0.0,"rotation_deg":0.0,"elbow_flex":10.0}, "SOURCE_ANCHORED"),
        KinematicVariant("external_90", {"target_elev":90.0,"plane_deg":0.0,"rotation_deg":25.0,"elbow_flex":10.0}, "SOURCE_ANCHORED"),
        KinematicVariant("internal_90", {"target_elev":90.0,"plane_deg":0.0,"rotation_deg":-25.0,"elbow_flex":10.0}, "SOURCE_ANCHORED"),
        KinematicVariant("scaption_90", {"target_elev":90.0,"plane_deg":30.0,"rotation_deg":0.0,"elbow_flex":10.0}, "SOURCE_ANCHORED"),
        KinematicVariant("flexed_elbow_90", {"target_elev":90.0,"plane_deg":0.0,"rotation_deg":0.0,"elbow_flex":90.0}, "SOURCE_ANCHORED"),
        KinematicVariant("natural_target_80", {"target_elev":80.0,"plane_deg":15.0,"rotation_deg":0.0,"elbow_flex":15.0}, "SYNTHETIC_STRESS"),
        KinematicVariant("natural_target_100", {"target_elev":100.0,"plane_deg":15.0,"rotation_deg":0.0,"elbow_flex":15.0}, "SYNTHETIC_STRESS"),
    )

    def render(self, frames, *, subject, variant, seed):
        rng=np.random.default_rng(_stable_seed("raise",subject.id,variant.id,seed))
        target=float(variant.params["target_elev"])
        plane=float(variant.params["plane_deg"])
        rotation=float(variant.params["rotation_deg"])
        elbow=float(variant.params["elbow_flex"])
        center=2.0 if plane <= 10 else 1.65
        shr=float(np.clip(rng.normal(center,.30),1.1,2.7))
        lag=float(rng.normal(0,.025))
        out=[]
        for f in frames:
            d=_clip01(f.depth); dl=_clip01(d+lag/2); dr=_clip01(d-lag/2)
            cl=_phase_curve(dl); cr=_phase_curve(dr)
            elev_l=target*cl; elev_r=target*cr
            scap_l=(elev_l/(shr+1.0))*_phase_curve(min(1,elev_l/max(1,target)))
            scap_r=(elev_r/(shr+1.0))*_phase_curve(min(1,elev_r/max(1,target)))
            elbow_l=float(np.clip(elbow+rng.normal(0,1.2),0,100)); elbow_r=float(np.clip(elbow+rng.normal(0,1.2),0,100))
            observed=float(np.median([_inverse_phase_curve(elev_l/target),_inverse_phase_curve(elev_r/target)]))
            if f.valid: observed=float(np.clip(observed+rng.normal(0,.008),0,1.06))
            out.append(KinematicFrame(f.frame_index,f.t_s,f.valid,f.assistance_evidence,d,observed,{
                "left_humeral_elevation_deg":elev_l,"right_humeral_elevation_deg":elev_r,
                "left_scapular_uprot_deg":scap_l,"right_scapular_uprot_deg":scap_r,
                "left_elbow_flex_deg":elbow_l,"right_elbow_flex_deg":elbow_r,
                "elevation_plane_deg":plane,"humeral_rotation_deg":rotation,"scapulohumeral_ratio":shr,
            },variant.id))
        return out


MODELS: dict[str, ExerciseKinematicModel] = {
    "incline_db_press": InclineDBPressModel(),
    "smith_squat": SmithSquatModel(),
    "lateral_raise": LateralRaiseModel(),
}


def as_rep_frames(frames: list[KinematicFrame]) -> list[MovementFrame]:
    return [MovementFrame(f.frame_index,f.t_s,f.observed_depth,f.valid,f.assistance_evidence) for f in frames]


def run_kinematic_model_benchmark(
    subjects: list[SubjectProfile],
    *,
    fps: int = 20,
    base_seed: int = 880000,
) -> pd.DataFrame:
    ex_by_id={x.id:x for x in EXERCISES}
    rows=[]
    for si,subj in enumerate(subjects):
        for mi,(exercise_id,model) in enumerate(MODELS.items()):
            ex=ex_by_id[exercise_id]
            for vi,variant in enumerate(model.variants):
                for ci,truth in enumerate(REP_CASES):
                    seed=base_seed+si*100000+mi*10000+vi*500+ci*13
                    base=generate_rep_case(truth.case_id,seed=seed,subject=subj,exercise=ex,fps=fps)
                    kin=model.render(base,subject=subj,variant=variant,seed=seed)
                    events=detect_reps(as_rep_frames(kin))
                    assisted=sum(x.classification=="ASSISTED" for x in events)
                    uncertain=sum(x.classification=="UNCERTAIN" for x in events)
                    passed=(truth.total_range.contains(len(events)) and truth.assisted_range.contains(assisted) and truth.uncertain_range.contains(uncertain))
                    rows.append({
                        "subject":subj.id,"exercise":exercise_id,"variant":variant.id,"provenance":variant.provenance,
                        "case":truth.case_id,"seed":seed,"passed":passed,"detected_total":len(events),
                        "expected_total_min":truth.total_range.low,"expected_total_max":truth.total_range.high,
                        "detected_assisted":assisted,"detected_uncertain":uncertain,
                    })
    return pd.DataFrame(rows)


def sample_kinematic_summary(subject: SubjectProfile, *, seed: int = 900001) -> pd.DataFrame:
    ex_by_id={x.id:x for x in EXERCISES}; rows=[]
    template=[MovementFrame(0,0,0,True,0),MovementFrame(1,.5,1,True,0)]
    for eid,model in MODELS.items():
        for v in model.variants:
            kin=model.render(template,subject=subject,variant=v,seed=seed)
            bottom=kin[-1]
            rows.append({"exercise":eid,"variant":v.id,"provenance":v.provenance,**bottom.channels})
    return pd.DataFrame(rows)
