from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from .adapter import PersonalCalibration, ReferenceProductionAdapter
from .models import Observation
from .profiles import EXERCISES, external_subjects, PERSONAL_VARIANTS


def _stable_uniform(key: str, low: float, high: float) -> float:
    seed=int.from_bytes(hashlib.blake2b(key.encode('utf-8'),digest_size=8).digest(),'little')
    return float(np.random.default_rng(seed).uniform(low,high))


def run_false_cue_benchmark(
    n: int = 100_000,
    seed: int = 20260919,
    calibrated: bool = False,
    adapter=None,
    mode: str = "natural",
    subjects=None,
) -> pd.DataFrame:
    """Measure spoken false corrections on movements labeled valid by the generator.

    Modes are deliberately separated:

    natural
        Normal person/exercise style variation with moderate within-person noise. This
        is the release-style false-cue KPI and should be very quiet.

    boundary
        Valid-but-unusual styles deliberately placed near the edge of the synthetic
        evidence domain. This is an adversarial diagnostic, not a release gate. It is
        expected to produce more false cues and is useful for comparing calibration.

    Body morphology is NOT directly converted into form-error evidence. Morphology
    belongs in geometry/camera simulation. A tiny capped projection residual remains
    only to represent imperfect normalization in a monocular observation pipeline.
    """
    if mode not in {"natural","boundary"}:
        raise ValueError("mode must be 'natural' or 'boundary'")
    adapter = adapter or ReferenceProductionAdapter()
    subjects = list(subjects) if subjects is not None else external_subjects() + PERSONAL_VARIANTS
    rng=np.random.default_rng(seed)
    rows=[]
    for i in range(n):
        subj=subjects[int(rng.integers(0,len(subjects)))]
        ex=EXERCISES[int(rng.integers(0,len(EXERCISES)))]

        # Stable valid movement style is keyed by subject+exercise, not morphology.
        key=f"{subj.id}:{ex.id}:{mode}"
        if mode=="natural":
            style=_stable_uniform(key,-.04,.32)
            center=.12+style
            rep_sd=.065
            excursion_prob=.04
            excursion_lo,excursion_hi=.08,.20
        else:
            style=_stable_uniform(key,.08,.32)
            center=.20+style
            rep_sd=.060
            excursion_prob=.05
            excursion_lo,excursion_hi=.05,.13

        # Small residual from imperfect generic normalization; capped so body shape
        # cannot magically become a form fault in the generator.
        raw_residual=(subj.torso_ratio-.31)*1.0+(subj.upper_arm_ratio-.186)*1.0
        projection_residual=float(np.clip(raw_residual,-.03,.03))
        center=float(np.clip(center+projection_residual,0,1))

        evid=[]
        for _ in range(10):
            v=center+rng.normal(0,rep_sd)
            if rng.random()<excursion_prob:
                v += rng.uniform(excursion_lo,excursion_hi)
            evid.append(float(np.clip(v,0,1)))

        obs=Observation(
            scenario_id=f"valid-{mode}-{i}",
            frame_fill=float(rng.uniform(ex.min_frame_fill+.03, ex.max_frame_fill-.03)),
            yaw_error_deg=float(rng.uniform(0,max(1.0,ex.yaw_tolerance_deg*.55))),
            tracking_quality=float(rng.uniform(.82,.99)),
            visible_required_fraction=float(rng.uniform(.88,1.0)),
            detected_people=1,
            primary_target_score=float(rng.uniform(.86,.99)),
            best_competitor_score=float(rng.uniform(.02,.30)),
            identity_continuity=float(rng.uniform(.88,.99)),
            mirror_ambiguity_score=float(rng.uniform(0,.15)),
            camera_motion_score=float(rng.uniform(0,.08)),
            rep_candidates=10,
            issue_evidence_by_rep=tuple(evid),
            setup_motion_score=.03,
            camera_vertical_error_m=float(rng.uniform(0,.25)),
            tracking_gap_ms=0,
        )
        cal=None
        if calibrated:
            # Calibration learns this person's stable valid style for this exercise.
            cal=PersonalCalibration(issue_baseline=center,confidence=.85)
        out=adapter.analyze(obs,ex,cal)
        rows.append({
            "case":i,"subject":subj.id,"exercise":ex.id,"mode":mode,
            "calibrated":calibrated,"style_offset":style,
            "projection_residual":projection_residual,
            "false_cue":out.cue_count>0,"cue_count":out.cue_count,
            "pause":out.pause,"ready":out.ready,"reason":out.reason,
            "max_evidence":max(evid),"mean_evidence":float(np.mean(evid)),
        })
    return pd.DataFrame(rows)
