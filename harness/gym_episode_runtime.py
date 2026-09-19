from __future__ import annotations

import numpy as np
import pandas as pd

from .models import SubjectProfile, ExerciseProfile
from .profiles import EXERCISES
from .rep_detection import MovementFrame, TemporalRepDetector
from .tolerance import DEFAULT_REALITY_TOLERANCE, IntegerRange
from .subject_lock import CandidateEvidence, STRATEGIES, TemporalPrimarySubjectLock
from .gym_episode_model import (
    EpisodeFrame, EpisodeResult, RepExpectation, RepTruth, CommercialGymEpisode,
    COMMERCIAL_GYM_EPISODES, _active, _rep_truths, _target_depth, _trainer_demo_depth,
    _background_depth, _crowd_count, _stable_seed,
)

def _candidate(track_id: str,rng: np.random.Generator,*,motion: float,pose: float,appearance: float,spatial: float,
               face: float|None=.80,mirror: bool=False,observability: float=.96) -> CandidateEvidence:
    def j(v: float, sd: float=.014) -> float: return float(np.clip(v+rng.normal(0,sd),0,1))
    return CandidateEvidence(track_id,j(motion),j(pose),j(appearance),j(spatial),None if face is None else j(face),mirror,float(np.clip(observability,0,1)))


def generate_episode_frames(subject: SubjectProfile,exercise: ExerciseProfile,ep: CommercialGymEpisode,seed: int,*,fps: int=5) -> list[EpisodeFrame]:
    rng=np.random.default_rng(_stable_seed(subject.id,exercise.id,ep.id,seed))
    reps=_rep_truths(ep,seed=seed); changed_track=False; frames=[]
    morph_delta=(subject.torso_ratio-.31)*.7+(subject.upper_arm_ratio-.186)*.8
    for fi in range(int(round(ep.duration_s*fps))):
        t=fi/fps; events=_active(ep,t); kinds={e.kind for e in events}
        intens={k:max(e.intensity for e in events if e.kind==k) for k in kinds}
        if "track_id_change" in kinds: changed_track=True
        target_id="target_reid" if changed_track else "target"
        target_depth,active_rep_idx=_target_depth(t,reps)
        visual=ep.base_visual_clutter+.035*_crowd_count(ep.crowd_level); bg_motion=.03*_crowd_count(ep.crowd_level)
        if "background_traffic" in kinds:
            bg_motion+=.22*intens["background_traffic"]; visual+=.08*intens["background_traffic"]
        tracking=.965-.10*visual-.06*bg_motion; visible=.985-.05*visual
        camera_valid=True; camera_changed=False; truth_identity_ambiguous=False; truth_blocks=False
        if "variable_lighting" in kinds: tracking-=.08*intens["variable_lighting"]
        if "low_light" in kinds: tracking-=.52*intens["low_light"]; visible-=.08*intens["low_light"]; truth_blocks=True
        if "motion_blur" in kinds: tracking-=.50*intens["motion_blur"]; truth_blocks=True
        if "equipment_occlusion" in kinds:
            visible-=.46*intens["equipment_occlusion"]; tracking-=.16*intens["equipment_occlusion"]
            if visible<.72: truth_blocks=True
        if "foreground_cross" in kinds: visible-=.12*intens["foreground_cross"]; tracking-=.08*intens["foreground_cross"]
        if "camera_bump" in kinds: camera_valid=False; camera_changed=True; truth_blocks=True
        if "floor_vibration" in kinds: tracking-=.045*intens["floor_vibration"]
        if "trainer_contact" in kinds:
            visible-=.08*intens["trainer_contact"]; tracking-=.03*intens["trainer_contact"]
        tracking=float(np.clip(tracking+rng.normal(0,.012),0,1)); visible=float(np.clip(visible+rng.normal(0,.010),0,1))

        cands=[]; actor_depths=[]
        if "full_occlusion" not in kinds:
            cands.append(_candidate(target_id,rng,motion=.945,pose=.945+morph_delta,appearance=.925,spatial=.95,face=.86,observability=visible))
            actor_depths.append((target_id,float(np.clip(target_depth+rng.normal(0,.006),0,1.05))))
        else: truth_blocks=True
        for pi in range(_crowd_count(ep.crowd_level)):
            tid=f"bg{pi}"; cands.append(_candidate(tid,rng,motion=.20+.05*(pi%3),pose=.40,appearance=.28,spatial=.26+.04*(pi%2),face=.25,observability=.95))
            actor_depths.append((tid,_background_depth(t,ep,pi)))
        trainer_present=bool({"trainer_presence","trainer_gesture","trainer_walkaround","trainer_demo","trainer_contact","trainer_assist"}&kinds)
        if trainer_present:
            motion=.56
            if "trainer_gesture" in kinds: motion=.78
            if "trainer_demo" in kinds: motion=.91
            spatial=.57 if "trainer_walkaround" not in kinds else .62+.12*np.sin(t*.7)
            cands.append(_candidate("trainer",rng,motion=motion,pose=.62 if "trainer_demo" not in kinds else .82,appearance=.36,spatial=float(spatial),face=.34,observability=.97))
            actor_depths.append(("trainer",_trainer_demo_depth(t,ep)))
        if "spotter" in kinds and not trainer_present:
            cands.append(_candidate("spotter",rng,motion=.50,pose=.54,appearance=.34,spatial=.58,face=.30,observability=.95)); actor_depths.append(("spotter",.15+.12*np.sin(t*2)))
        if "foreground_cross" in kinds:
            cands.append(_candidate("crossing",rng,motion=.68,pose=.48,appearance=.34,spatial=.67,face=.24,observability=.97)); actor_depths.append(("crossing",.38))
        if "known_mirror" in kinds or ep.mirror_wall:
            cands.append(_candidate("reflection",rng,motion=.92,pose=.94,appearance=.92,spatial=.83,face=.82,mirror=True,observability=.98)); actor_depths.append(("reflection",target_depth))
        if "lookalike" in kinds:
            cands.append(_candidate("lookalike",rng,motion=.92,pose=.93,appearance=.925,spatial=.90,face=.80,observability=.98)); actor_depths.append(("lookalike",_background_depth(t,ep,7)))
            truth_identity_ambiguous=True; truth_blocks=True
        assist=0.0
        if "trainer_assist" in kinds and active_rep_idx is not None: assist=max(assist,intens["trainer_assist"])
        if "trainer_contact" in kinds and active_rep_idx is not None: assist=max(assist,.35*intens["trainer_contact"])
        completion_idx=None
        for r in reps:
            if abs(t-r.complete_s) <= .5/fps: completion_idx=r.index; break
        movement_valid=True
        for r in reps:
            if r.style=="failed" and r.complete_s <= t < r.complete_s + .8:
                movement_valid=False; break
        issue=.08+float(rng.normal(0,.018))
        if active_rep_idx in ep.issue_reps or ("repeated_issue" in kinds and active_rep_idx is not None): issue=.78+float(rng.normal(0,.025))
        if "fatigue" in kinds and active_rep_idx is not None: issue=max(issue,.26+float(rng.normal(0,.025)))
        frames.append(EpisodeFrame(fi,t,tuple(cands),tracking,visible,camera_valid,camera_changed,("target","target_reid"),tuple(actor_depths),
                                   float(np.clip(assist,0,1)),movement_valid,completion_idx,float(np.clip(issue,0,1)),truth_blocks,truth_identity_ambiguous,tuple(sorted(kinds))))
    return frames


def _truth_reset_episodes(frames):
    count=0; active=False
    for f in frames:
        hard=f.truth_identity_ambiguous or f.camera_changed or ("full_occlusion" in f.active_events)
        if hard and not active: count+=1
        active=hard
    return count


def _truth_rep_expectations(frames: list[EpisodeFrame], reps: list[RepTruth], fps: int) -> dict[int,RepExpectation]:
    out={}
    for r in reps:
        if not r.should_count:
            out[r.index]=RepExpectation(r.index,"MUST_NOT",frozenset(),1.0); continue
        relevant=[f for f in frames if r.start_s - .5/fps <= f.t_s <= r.complete_s + .5/fps]
        if not relevant:
            out[r.index]=RepExpectation(r.index,"MAY",frozenset({"NORMAL","ASSISTED","UNCERTAIN"}),0.0); continue
        observable=[f for f in relevant if not f.truth_blocks_biomechanics]
        frac=len(observable)/len(relevant)
        phase_window=max(.25,1.0/fps)
        phase_critical=[f for f in relevant if min(abs(f.t_s-r.start_s),abs(f.t_s-r.bottom_s),abs(f.t_s-r.complete_s))<=phase_window]
        phase_blocked=any(f.truth_blocks_biomechanics for f in phase_critical)
        policy="MUST" if frac>=.90 and not phase_blocked else ("MUST_NOT" if frac<=.45 else "MAY")
        concentric=[f.target_assistance_evidence for f in observable if f.t_s >= r.bottom_s + .5/fps]
        if policy=="MUST_NOT": classes=frozenset()
        elif not concentric: classes=frozenset({"NORMAL"})
        else:
            arr=np.asarray(concentric); af=float(np.mean(arr>=.28)); sf=float(np.mean(arr>=.78))
            if sf>=.50: classes=frozenset({"UNCERTAIN"})
            elif sf>0: classes=frozenset({"ASSISTED","UNCERTAIN"}) if af>=.50 else frozenset({"NORMAL","ASSISTED","UNCERTAIN"})
            elif af>=.50: classes=frozenset({"ASSISTED"})
            elif af>0: classes=frozenset({"NORMAL","ASSISTED"})
            else: classes=frozenset({"NORMAL"})
        out[r.index]=RepExpectation(r.index,policy,classes,float(frac))
    return out


def run_episode(subject: SubjectProfile,exercise: ExerciseProfile,ep: CommercialGymEpisode,seed: int,*,fps: int=5) -> EpisodeResult:
    frames=generate_episode_frames(subject,exercise,ep,seed,fps=fps)
    reps=_rep_truths(ep,seed=seed); tol=DEFAULT_REALITY_TOLERANCE
    tracker=TemporalPrimarySubjectLock(STRATEGIES[-1],occlusion_grace_frames=max(2,int(.8*fps)),reacquire_frames=2)
    repdet=TemporalRepDetector(top_dwell_frames=max(1,int(.25*fps)),min_rep_duration_s=.40)
    expectations=_truth_rep_expectations(frames,reps,fps)
    must={i for i,e in expectations.items() if e.count_policy=="MUST"}; may={i for i,e in expectations.items() if e.count_policy=="MAY"}; allowed=must|may
    count_range=IntegerRange(len(must),len(allowed))
    assisted_range=IntegerRange(sum(e.count_policy=="MUST" and e.allowed_classes==frozenset({"ASSISTED"}) for e in expectations.values()),sum(e.count_policy!="MUST_NOT" and "ASSISTED" in e.allowed_classes for e in expectations.values()))
    uncertain_range=IntegerRange(sum(e.count_policy=="MUST" and e.allowed_classes==frozenset({"UNCERTAIN"}) for e in expectations.values()),sum(e.count_policy!="MUST_NOT" and "UNCERTAIN" in e.allowed_classes for e in expectations.values()))
    cue_expected=sum(i in ep.issue_reps for i in must)>=2
    counted=[]; detected_class_by_rep={}; assisted=uncertain=false_reps=cue_count=0; issue_history=[]
    wrong_locks=unsafe_bio=unexpected_pause=expected_pause=observed_pause=reset_count=0; in_interrupt=False
    max_reacquire_s=0.0; block_ended_at=None
    for f in frames:
        d=tracker.step(list(f.candidates))
        if d.state=="TARGET_LOCKED" and d.target_track_id not in f.target_track_ids: wrong_locks+=1
        identity_ok=d.state=="TARGET_LOCKED" and d.target_track_id in f.target_track_ids
        tracking_ok=f.tracking_quality>=.60 and f.visible_required_fraction>=.72
        analysis_active=d.state=="TARGET_LOCKED" and tracking_ok and f.camera_valid and not f.camera_changed
        expected_ok=not f.truth_blocks_biomechanics
        if not expected_ok: expected_pause+=1
        if not analysis_active or not identity_ok: observed_pause+=1
        if analysis_active and (not expected_ok or not identity_ok): unsafe_bio+=1
        if (not analysis_active or not identity_ok) and expected_ok and (block_ended_at is None or f.t_s-block_ended_at>.65): unexpected_pause+=1
        hard_interrupt=d.state in {"TARGET_AMBIGUOUS","TARGET_LOST"} or f.camera_changed
        if hard_interrupt and not in_interrupt: reset_count+=1
        in_interrupt=hard_interrupt
        if f.truth_blocks_biomechanics: block_ended_at=None
        elif block_ended_at is None:
            prev=frames[f.frame_index-1] if f.frame_index>0 else None
            if prev is not None and prev.truth_blocks_biomechanics: block_ended_at=f.t_s
        if block_ended_at is not None and analysis_active and identity_ok:
            max_reacquire_s=max(max_reacquire_s,f.t_s-block_ended_at); block_ended_at=None
        depths=dict(f.actor_depths); selected_depth=depths.get(d.target_track_id or "",0.0)
        ev=repdet.step(MovementFrame(f.frame_index,f.t_s,selected_depth,valid=(analysis_active and identity_ok and f.movement_valid),assistance_evidence=f.target_assistance_evidence if identity_ok else 0.0))
        if ev is not None:
            if not identity_ok: false_reps+=1; continue
            nearest=min(reps,key=lambda r:abs(r.complete_s-ev.completed_t_s)); exp=expectations[nearest.index]
            concentric_s=max(0.0,nearest.complete_s-nearest.bottom_s); early=max(tol.rep_completion_early_s,tol.rep_completion_early_fraction_of_concentric*concentric_s)
            if nearest.complete_s-early <= ev.completed_t_s <= nearest.complete_s+tol.rep_completion_late_s and exp.count_policy!="MUST_NOT" and nearest.index not in detected_class_by_rep:
                counted.append(nearest.index); detected_class_by_rep[nearest.index]=ev.classification
                assisted+=ev.classification=="ASSISTED"; uncertain+=ev.classification=="UNCERTAIN"
                issue_history.append((nearest.index in ep.issue_reps) or f.issue_strength>.58)
                if len(issue_history)>=3 and sum(issue_history[-3:])>=2 and cue_count==0: cue_count=1
            else: false_reps+=1
    expected_reset=_truth_reset_episodes(frames)
    detected=set(counted)
    passed=(wrong_locks==0 and unsafe_bio==0 and false_reps==0 and must.issubset(detected) and detected.issubset(allowed) and count_range.contains(len(detected))
            and assisted_range.contains(int(assisted)) and uncertain_range.contains(int(uncertain))
            and all(detected_class_by_rep[i] in expectations[i].allowed_classes for i in detected)
            and (cue_count>0)==cue_expected and reset_count<=expected_reset and (expected_reset==0 or reset_count>=1)
            and unexpected_pause<=max(2,int(tol.post_interrupt_pause_grace_s*fps)) and max_reacquire_s<=tol.reacquire_max_s+1e-9)
    return EpisodeResult(subject.id,exercise.id,ep.id,seed,ep.duration_s,passed,len(reps),count_range.low,count_range.high,len(counted),
                         assisted_range.low,assisted_range.high,int(assisted),uncertain_range.low,uncertain_range.high,int(uncertain),
                         false_reps,cue_count,cue_expected,wrong_locks,unsafe_bio,unexpected_pause,expected_pause,observed_pause,
                         reset_count,expected_reset,max_reacquire_s,ep.crowd_level,ep.ambient_noise_db)


def run_commercial_gym_episode_benchmark(subjects,*,seeds_per_episode=1,base_seed=910000,exercises=None,fps=5):
    exercises=exercises or EXERCISES; rows=[]
    for si,subj in enumerate(subjects):
        for ei,ex in enumerate(exercises):
            for pi,ep in enumerate(COMMERCIAL_GYM_EPISODES):
                for k in range(seeds_per_episode):
                    seed=base_seed+si*100000+ei*1000+pi*20+k
                    rows.append(run_episode(subj,ex,ep,seed,fps=fps).row())
    return pd.DataFrame(rows)
