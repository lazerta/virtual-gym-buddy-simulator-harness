from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np
import pandas as pd

from .models import ExerciseProfile, SubjectProfile
from .profiles import EXERCISES
from .tolerance import IntegerRange


@dataclass(frozen=True)
class MovementFrame:
    frame_index: int
    t_s: float
    depth: float
    valid: bool = True
    assistance_evidence: float = 0.0


@dataclass(frozen=True)
class RepEvent:
    rep_index: int
    completed_t_s: float
    duration_s: float
    max_depth: float
    classification: str
    assistance_evidence: float


class TemporalRepDetector:
    def __init__(self, *, top_threshold=.20,start_threshold=.30,bottom_threshold=.78,top_dwell_frames=2,
                 min_rep_duration_s=.45,max_rep_duration_s=8.0,assisted_threshold=.28,uncertain_threshold=.78):
        self.top_threshold=top_threshold; self.start_threshold=start_threshold; self.bottom_threshold=bottom_threshold
        self.top_dwell_frames=top_dwell_frames; self.min_rep_duration_s=min_rep_duration_s; self.max_rep_duration_s=max_rep_duration_s
        self.assisted_threshold=assisted_threshold; self.uncertain_threshold=uncertain_threshold; self._rep_index=0
        self.reset(require_top=True)

    def reset(self, *, require_top=True):
        self.state="SEEK_TOP" if require_top else "ARMED"; self.top_dwell=0; self.start_t=None
        self.max_depth=0.0; self.bottom_reached=False; self.prev_depth=None; self.concentric_assistance_max=0.0

    @property
    def rep_count(self): return self._rep_index

    def step(self,f):
        if not f.valid or not np.isfinite(f.depth): self.reset(require_top=True); return None
        depth=float(np.clip(f.depth,0,1.2))
        if self.state=="SEEK_TOP":
            if depth<=self.top_threshold:
                self.top_dwell+=1
                if self.top_dwell>=self.top_dwell_frames: self.state="ARMED"; self.prev_depth=depth
            else: self.top_dwell=0
            return None
        if self.state=="ARMED":
            if depth<=self.top_threshold: self.prev_depth=depth; return None
            if depth>=self.start_threshold:
                self.state="ECCENTRIC"; self.start_t=f.t_s; self.max_depth=depth
                self.bottom_reached=depth>=self.bottom_threshold; self.prev_depth=depth
            return None
        assert self.start_t is not None
        duration=f.t_s-self.start_t
        if duration>self.max_rep_duration_s: self.reset(require_top=(depth>self.top_threshold)); return None
        self.max_depth=max(self.max_depth,depth)
        if depth>=self.bottom_threshold: self.bottom_reached=True
        if self.state=="ECCENTRIC":
            if self.bottom_reached and self.prev_depth is not None and depth<self.prev_depth-.012:
                self.state="CONCENTRIC"; self.concentric_assistance_max=max(self.concentric_assistance_max,f.assistance_evidence)
            elif not self.bottom_reached and depth<=self.top_threshold:
                self.reset(require_top=False); self.prev_depth=depth; return None
        elif self.state=="CONCENTRIC":
            self.concentric_assistance_max=max(self.concentric_assistance_max,f.assistance_evidence)
            if depth<=self.top_threshold:
                if duration<self.min_rep_duration_s or not self.bottom_reached:
                    self.reset(require_top=False); self.prev_depth=depth; return None
                self._rep_index+=1; a=float(np.clip(self.concentric_assistance_max,0,1))
                cls="UNCERTAIN" if a>=self.uncertain_threshold else ("ASSISTED" if a>=self.assisted_threshold else "NORMAL")
                ev=RepEvent(self._rep_index,f.t_s,duration,self.max_depth,cls,a); self.reset(require_top=False); self.prev_depth=depth; return ev
        self.prev_depth=depth; return None


def _stable_seed(*parts):
    return int.from_bytes(hashlib.blake2b("|".join(map(str,parts)).encode(),digest_size=8).digest(),"little") & 0x7FFFFFFF

def _append_hold(rows,t,duration,depth,fps,*,valid=True,assistance=0.0):
    for _ in range(max(1,int(round(duration*fps)))): rows.append((t,depth,valid,assistance)); t+=1/fps
    return t

def _append_ramp(rows,t,duration,a,b,fps,*,valid=True,assistance=0.0):
    n=max(2,int(round(duration*fps)))
    for i in range(n):
        x=(i+1)/n; rows.append((t,a+(b-a)*x,valid,assistance)); t+=1/fps
    return t

def _append_complete_rep(rows,t,fps,*,down_s=.85,bottom_pause_s=.08,up_s=.85,top_pause_s=.28,assistance=0.0,bounce=False):
    t=_append_ramp(rows,t,down_s,0,1,fps)
    if bounce:
        for a,b in [(1,.84),(.84,.97),(.97,.90),(.90,1)]: t=_append_ramp(rows,t,.10,a,b,fps)
    t=_append_hold(rows,t,bottom_pause_s,1,fps); t=_append_ramp(rows,t,up_s,1,0,fps,assistance=assistance)
    return _append_hold(rows,t,top_pause_s,0,fps)

@dataclass(frozen=True)
class RepCaseTruth:
    case_id:str
    total_range:IntegerRange
    assisted_range:IntegerRange=IntegerRange.exact(0)
    uncertain_range:IntegerRange=IntegerRange.exact(0)
    @property
    def expected_total(self): return self.total_range.low
    @property
    def expected_assisted(self): return self.assisted_range.low
    @property
    def expected_uncertain(self): return self.uncertain_range.low

REP_CASES=(
 RepCaseTruth("clean_5",IntegerRange.exact(5)),RepCaseTruth("partial_only",IntegerRange.exact(0)),
 RepCaseTruth("half_reverse",IntegerRange.exact(0)),RepCaseTruth("failed_concentric",IntegerRange.exact(0)),
 RepCaseTruth("bottom_pause",IntegerRange.exact(3)),RepCaseTruth("top_pause",IntegerRange.exact(3)),
 RepCaseTruth("slow_grinder",IntegerRange.exact(3)),RepCaseTruth("bounce",IntegerRange.exact(3)),
 RepCaseTruth("pulse_cluster",IntegerRange.exact(0)),RepCaseTruth("rest_pause",IntegerRange.exact(4)),
 RepCaseTruth("setup_then_clean",IntegerRange.exact(2)),RepCaseTruth("tracking_gap_mid_rep_then_clean",IntegerRange.exact(1)),
 RepCaseTruth("assisted_light_last2",IntegerRange.exact(5),assisted_range=IntegerRange.exact(2)),
 RepCaseTruth("assisted_strong_last1",IntegerRange.exact(5),uncertain_range=IntegerRange.exact(1)),
)

def _natural_rep_params(rng,*,slow=False):
    return {"down_s":float(rng.uniform(.62,1.18)),"bottom_pause_s":float(rng.uniform(.04,.24)),
            "up_s":float(rng.uniform(2.5,4.0) if slow else rng.uniform(.62,1.22)),"top_pause_s":float(rng.uniform(.18,.58))}

def generate_rep_case(case_id,*,seed,subject=None,exercise=None,fps=20):
    rng=np.random.default_rng(_stable_seed(case_id,seed,subject.id if subject else "none",exercise.id if exercise else "none"))
    rows=[]; t=_append_hold(rows,0.0,.45,0,fps)
    if case_id=="clean_5":
        for _ in range(5): t=_append_complete_rep(rows,t,fps,**_natural_rep_params(rng))
    elif case_id=="partial_only":
        for _ in range(5): t=_append_ramp(rows,t,.55,0,.60,fps); t=_append_ramp(rows,t,.55,.60,0,fps); t=_append_hold(rows,t,.22,0,fps)
    elif case_id=="half_reverse":
        for _ in range(4):
            t=_append_ramp(rows,t,.45,0,.68,fps); t=_append_ramp(rows,t,.22,.68,.36,fps); t=_append_ramp(rows,t,.20,.36,.66,fps); t=_append_ramp(rows,t,.40,.66,0,fps); t=_append_hold(rows,t,.25,0,fps)
    elif case_id=="failed_concentric":
        t=_append_ramp(rows,t,.8,0,1,fps); t=_append_ramp(rows,t,1.3,1,.43,fps); t=_append_hold(rows,t,.5,.43,fps); t=_append_hold(rows,t,.35,.2,fps,valid=False)
    elif case_id=="bottom_pause":
        for _ in range(3): p=_natural_rep_params(rng); p["bottom_pause_s"]=float(rng.uniform(.75,1.35)); t=_append_complete_rep(rows,t,fps,**p)
    elif case_id=="top_pause":
        for _ in range(3): p=_natural_rep_params(rng); p["top_pause_s"]=float(rng.uniform(1.4,2.2)); t=_append_complete_rep(rows,t,fps,**p)
    elif case_id=="slow_grinder":
        for _ in range(3): t=_append_complete_rep(rows,t,fps,**_natural_rep_params(rng,slow=True))
    elif case_id=="bounce":
        for _ in range(3): t=_append_complete_rep(rows,t,fps,bounce=True,**_natural_rep_params(rng))
    elif case_id=="pulse_cluster":
        for _ in range(7): t=_append_ramp(rows,t,.22,.28,.62,fps); t=_append_ramp(rows,t,.22,.62,.30,fps)
        t=_append_hold(rows,t,.5,0,fps)
    elif case_id=="rest_pause":
        for _ in range(2): t=_append_complete_rep(rows,t,fps,**_natural_rep_params(rng))
        t=_append_hold(rows,t,float(rng.uniform(6.5,10)),0,fps)
        for _ in range(2): t=_append_complete_rep(rows,t,fps,**_natural_rep_params(rng))
    elif case_id=="setup_then_clean":
        rows.clear(); t=0.; t=_append_ramp(rows,t,.35,.45,.15,fps); t=_append_ramp(rows,t,.35,.15,.58,fps); t=_append_hold(rows,t,.25,.42,fps); t=_append_hold(rows,t,.55,0,fps)
        for _ in range(2): t=_append_complete_rep(rows,t,fps,**_natural_rep_params(rng))
    elif case_id=="tracking_gap_mid_rep_then_clean":
        t=_append_ramp(rows,t,.75,0,.95,fps); t=_append_hold(rows,t,.40,.95,fps,valid=False); t=_append_hold(rows,t,.50,0,fps); t=_append_complete_rep(rows,t,fps,**_natural_rep_params(rng))
    elif case_id=="assisted_light_last2":
        for i in range(5): t=_append_complete_rep(rows,t,fps,assistance=float(rng.uniform(.40,.60)) if i>=3 else 0,**_natural_rep_params(rng))
    elif case_id=="assisted_strong_last1":
        for i in range(5): t=_append_complete_rep(rows,t,fps,assistance=float(rng.uniform(.86,.96)) if i==4 else 0,**_natural_rep_params(rng))
    else: raise KeyError(case_id)
    noise_sd=.004
    if subject is not None: noise_sd+=min(.004,abs(subject.torso_ratio-.31)*.08+abs(subject.upper_arm_ratio-.186)*.08)
    if exercise is not None and exercise.bilateral: noise_sd+=.001
    return [MovementFrame(i,tt,float(np.clip(d+(rng.normal(0,noise_sd) if v else 0),0,1.05)),v,float(np.clip(a+(rng.normal(0,.015) if a>0 else 0),0,1))) for i,(tt,d,v,a) in enumerate(rows)]

def detect_reps(frames,detector=None):
    det=detector or TemporalRepDetector(); out=[]
    for f in frames:
        ev=det.step(f)
        if ev is not None: out.append(ev)
    return out

def run_rep_detection_benchmark(subjects,*,exercises=None,fps=20,base_seed=760000):
    exercises=exercises or EXERCISES; rows=[]
    for si,subj in enumerate(subjects):
        for ei,ex in enumerate(exercises):
            for ci,truth in enumerate(REP_CASES):
                seed=base_seed+si*100000+ei*1000+ci*11; events=detect_reps(generate_rep_case(truth.case_id,seed=seed,subject=subj,exercise=ex,fps=fps))
                assisted=sum(e.classification=="ASSISTED" for e in events); uncertain=sum(e.classification=="UNCERTAIN" for e in events); normal=sum(e.classification=="NORMAL" for e in events)
                rows.append({"subject":subj.id,"exercise":ex.id,"case":truth.case_id,"seed":seed,"passed":truth.total_range.contains(len(events)) and truth.assisted_range.contains(assisted) and truth.uncertain_range.contains(uncertain),
                             "expected_total_min":truth.total_range.low,"expected_total_max":truth.total_range.high,"detected_total":len(events),
                             "expected_assisted_min":truth.assisted_range.low,"expected_assisted_max":truth.assisted_range.high,"detected_assisted":assisted,
                             "expected_uncertain_min":truth.uncertain_range.low,"expected_uncertain_max":truth.uncertain_range.high,"detected_uncertain":uncertain,"detected_normal":normal})
    return pd.DataFrame(rows)
