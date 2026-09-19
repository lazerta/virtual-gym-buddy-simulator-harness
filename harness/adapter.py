from __future__ import annotations
from dataclasses import dataclass
from .models import Observation, ExerciseProfile, AnalyzerOutput

@dataclass(frozen=True)
class PersonalCalibration:
    issue_baseline: float = .10
    confidence: float = .0

class ReferenceProductionAdapter:
    name = "reference-python"
    """Reference stand-in for the future real Gym Buddy ProductionAdapter.
    It consumes observations only; it cannot see Scenario or oracle truth.
    Subject selection follows primary-target-lock semantics: extra people are ignored
    while the locked target remains confidently separable from competitors.
    """
    def analyze(self, obs: Observation, ex: ExerciseProfile, calibration: PersonalCalibration|None=None) -> AnalyzerOutput:
        frame_ok=ex.min_frame_fill <= obs.frame_fill <= ex.max_frame_fill
        view_ok=obs.yaw_error_deg <= ex.yaw_tolerance_deg
        vertical_ok=obs.camera_vertical_error_m <= .75
        ready=frame_ok and view_ok and vertical_ok
        if not ready:
            return AnalyzerOutput(False,False,0,0,0,"camera_guidance")

        camera_changed=obs.camera_motion_score>.55
        if camera_changed:
            return AnalyzerOutput(False,True,1,0,0,"camera_changed")

        # Long discontinuity invalidates temporal biomechanics state even if identity can later be re-identified.
        if obs.tracking_gap_ms>300:
            return AnalyzerOutput(True,True,1,obs.rep_candidates,0,"target_temporarily_lost")

        identity_margin=obs.identity_margin
        target_ambiguous=(
            obs.primary_target_score < .58 or
            obs.identity_continuity < .52 or
            identity_margin < .12 or
            obs.mirror_ambiguity_score > .72
        )
        if target_ambiguous:
            return AnalyzerOutput(True,True,1,0,0,"target_ambiguous")

        # Other people are irrelevant here unless they actually prevent observing the locked target.
        low_quality=obs.tracking_quality<.60 or obs.visible_required_fraction<.72
        if low_quality:
            return AnalyzerOutput(True,True,0,obs.rep_candidates,0,"target_observation_low")

        if obs.setup_motion_score>.65:
            return AnalyzerOutput(True,False,0,0,0,"setup_motion")

        baseline=.10
        if calibration:
            w=max(0,min(.65, calibration.confidence*.65))
            baseline=(1-w)*.10+w*calibration.issue_baseline
        threshold=.56+baseline*.18
        flags=[v>threshold for v in obs.issue_evidence_by_rep]
        cue=0
        for i in range(2,len(flags)):
            if sum(flags[i-2:i+1])>=2:
                cue=1; break
        return AnalyzerOutput(True,False,0,obs.rep_candidates,cue,"active")
