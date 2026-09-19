from __future__ import annotations
from .models import Scenario, ExpectedOutcome


def _components(s: Scenario) -> set[str]:
    return set(s.components or (s.family,))

# Oracle is specification-driven and does not inspect AnalyzerOutput or production thresholds.
def expected_for(s: Scenario) -> ExpectedOutcome:
    c=_components(s)
    # Setup/camera validity has highest precedence because active biomechanics never starts.
    if c & {"wrong_view","too_close","too_far","camera_low","camera_high"}:
        return ExpectedOutcome(False,False,False,0,False,True,"camera_guidance")
    if "camera_bump" in c:
        return ExpectedOutcome(False,True,True,0,False,True,"camera_changed")

    # Identity uncertainty takes precedence over form evidence and generic observation quality.
    if c & {"subject_swap","lookalike_competitor","mirror_ambiguous"}:
        return ExpectedOutcome(True,True,True,0,False,True,"target_ambiguous")

    # Long target discontinuity invalidates the active temporal candidate.
    if c & {"tracking_gap","target_exit_reenter"}:
        return ExpectedOutcome(True,True,True,s.rep_count_gt,False,True,"target_temporarily_lost")

    # Identity may remain known while biomechanics is not observable.
    if c & {"foreground_occlusion","motion_blur","low_light"}:
        return ExpectedOutcome(True,True,False,s.rep_count_gt if "foreground_occlusion" in c else 0,False,True,"target_observation_low")

    if "setup_motion" in c:
        return ExpectedOutcome(True,False,False,0,False,True,"setup_motion")

    # Background people/spotters/known mirrors are intentionally no-ops under a strong lock.
    persistent=bool(c & {"repeated_issue","short_rom","asymmetry"})
    isolated=("single_issue" in c) and not persistent
    if persistent:
        return ExpectedOutcome(True,False,False,s.rep_count_gt,True,False,"persistent_issue")
    if isolated:
        return ExpectedOutcome(True,False,False,s.rep_count_gt,False,True,"isolated_issue")
    return ExpectedOutcome(True,False,False,s.rep_count_gt,False,True,"clean")
