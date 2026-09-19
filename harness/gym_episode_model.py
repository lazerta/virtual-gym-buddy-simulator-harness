from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np
import pandas as pd

from .models import SubjectProfile, ExerciseProfile
from .profiles import EXERCISES
from .rep_detection import MovementFrame, TemporalRepDetector
from .tolerance import DEFAULT_REALITY_TOLERANCE, IntegerRange, allowed_assistance_classes
from .subject_lock import CandidateEvidence, STRATEGIES, TemporalPrimarySubjectLock


@dataclass(frozen=True)
class EpisodeEvent:
    kind: str
    start_s: float
    end_s: float
    intensity: float = 1.0

    def active(self, t_s: float) -> bool:
        return self.start_s <= t_s < self.end_s


@dataclass(frozen=True)
class CommercialGymEpisode:
    id: str
    duration_s: float
    crowd_level: str
    ambient_noise_db: float
    mirror_wall: bool = False
    base_visual_clutter: float = 0.15
    base_camera_jitter: float = 0.015
    events: tuple[EpisodeEvent, ...] = ()
    issue_reps: tuple[int, ...] = ()
    rep_styles: tuple[tuple[int, str], ...] = ()
    rest_pause_after_rep: int | None = None


@dataclass(frozen=True)
class RepTruth:
    index: int
    start_s: float
    bottom_s: float
    complete_s: float
    style: str = "normal"

    @property
    def should_count(self) -> bool:
        return self.style not in {"partial", "failed", "pulse"}


@dataclass(frozen=True)
class RepExpectation:
    index: int
    count_policy: str  # MUST | MAY | MUST_NOT
    allowed_classes: frozenset[str]
    observable_fraction: float


@dataclass(frozen=True)
class EpisodeFrame:
    frame_index: int
    t_s: float
    candidates: tuple[CandidateEvidence, ...]
    tracking_quality: float
    visible_required_fraction: float
    camera_valid: bool
    camera_changed: bool
    target_track_ids: tuple[str, ...]
    actor_depths: tuple[tuple[str, float], ...]
    target_assistance_evidence: float
    movement_valid: bool
    target_truth_rep_index: int | None
    issue_strength: float
    truth_blocks_biomechanics: bool
    truth_identity_ambiguous: bool
    active_events: tuple[str, ...]


@dataclass(frozen=True)
class EpisodeResult:
    subject: str
    exercise: str
    episode: str
    seed: int
    duration_s: float
    passed: bool
    total_target_attempts: int
    expected_counted_reps: int
    expected_counted_max_reps: int
    counted_reps: int
    expected_assisted_reps: int
    expected_assisted_max_reps: int
    assisted_reps: int
    expected_uncertain_reps: int
    expected_uncertain_max_reps: int
    uncertain_reps: int
    false_reps: int
    cue_count: int
    cue_expected: bool
    wrong_identity_locks: int
    unsafe_biomechanics_frames: int
    unexpected_pause_frames: int
    expected_pause_frames: int
    observed_pause_frames: int
    reset_count: int
    expected_reset_episodes: int
    max_reacquire_s: float
    crowd_level: str
    ambient_noise_db: float

    @property
    def total_target_reps(self) -> int:
        return self.total_target_attempts

    def row(self) -> dict:
        d=self.__dict__.copy()
        d["total_target_reps"]=self.total_target_attempts
        return d


def _e(kind: str, start: float, end: float, intensity: float = 1.0) -> EpisodeEvent:
    return EpisodeEvent(kind, start, end, intensity)


COMMERCIAL_GYM_EPISODES: tuple[CommercialGymEpisode, ...] = (
    CommercialGymEpisode("off_peak_control",42,"low",67,False,.08,.008,(_e("background_traffic",8,36,.25),)),
    CommercialGymEpisode("rush_hour_bench_spotter",66,"high",86,False,.30,.020,(
        _e("background_traffic",0,66,.85),_e("spotter",11,57,.75),_e("foreground_cross",22,23.6,.65),
        _e("floor_vibration",31,36,.55),_e("variable_lighting",40,47,.45),_e("fatigue",49,61,.70),)),
    CommercialGymEpisode("mirror_wall_evening",58,"high",84,True,.32,.018,(
        _e("background_traffic",0,58,.70),_e("known_mirror",0,58),_e("foreground_cross",17.5,19,.55),
        _e("variable_lighting",28,34,.50),_e("repeated_issue",42,55,.80)),issue_reps=(8,9,10)),
    CommercialGymEpisode("crowded_smith_rack",72,"high",88,False,.38,.025,(
        _e("background_traffic",0,72,.90),_e("equipment_occlusion",20,23,.70),_e("foreground_cross",33,34.8,.70),
        _e("equipment_occlusion",45,49,.85),_e("floor_vibration",52,59,.65),_e("fatigue",54,68,.75),)),
    CommercialGymEpisode("lookalike_overlap",62,"high",85,False,.30,.018,(
        _e("background_traffic",0,62,.75),_e("lookalike",27,33),_e("foreground_cross",40,41.5,.55),_e("fatigue",47,59,.60),)),
    CommercialGymEpisode("occlusion_reidentify",52,"medium",81,False,.22,.014,(
        _e("background_traffic",0,52,.50),_e("foreground_cross",15.5,17,.65),_e("full_occlusion",27,29.2),
        _e("track_id_change",29.2,52),_e("background_traffic",32,52,.65),)),
    CommercialGymEpisode("low_light_fast_movement",48,"medium",82,False,.24,.018,(
        _e("background_traffic",0,48,.55),_e("low_light",16,19.5,.85),_e("motion_blur",29,31.5,.85),_e("variable_lighting",35,41,.65),)),
    CommercialGymEpisode("camera_bump_recovery",54,"medium",80,False,.20,.015,(
        _e("background_traffic",0,54,.50),_e("camera_bump",25,26.4),_e("floor_vibration",38,44,.55),_e("fatigue",40,51,.65),)),
    CommercialGymEpisode("busy_issue_under_distraction",64,"high",89,True,.36,.022,(
        _e("background_traffic",0,64,.90),_e("known_mirror",0,64),_e("spotter",10,55,.60),_e("foreground_cross",24,25.4,.65),
        _e("variable_lighting",32,38,.55),_e("repeated_issue",44,61,.90)),issue_reps=(7,8,9,10)),
    CommercialGymEpisode("mixed_adversarial_rush",86,"high",91,True,.42,.028,(
        _e("background_traffic",0,86),_e("known_mirror",0,86),_e("spotter",9,68,.65),_e("foreground_cross",18,19.6,.75),
        _e("equipment_occlusion",30,33,.75),_e("lookalike",42,47),_e("variable_lighting",51,57,.60),
        _e("motion_blur",60,62,.70),_e("floor_vibration",66,74,.65),_e("repeated_issue",70,83,.85)),issue_reps=(9,10)),
    CommercialGymEpisode("trainer_close_coaching",60,"medium",83,False,.24,.016,(
        _e("background_traffic",0,60,.55),_e("trainer_presence",7,57,.90),_e("trainer_gesture",14,30,.80),
        _e("trainer_walkaround",31,47,.85),_e("foreground_cross",38.5,39.6,.35),)),
    CommercialGymEpisode("trainer_demonstrates_same_exercise",68,"high",87,True,.34,.020,(
        _e("background_traffic",0,68,.80),_e("known_mirror",0,68),_e("trainer_presence",8,62,.90),
        _e("trainer_demo",18,49,1.0),_e("trainer_gesture",51,58,.65)),),
    CommercialGymEpisode("trainer_assisted_finish",72,"high",88,False,.32,.020,(
        _e("background_traffic",0,72,.75),_e("trainer_presence",10,69,.90),_e("spotter",20,69,.80),
        _e("trainer_assist",49,60,.50),_e("trainer_assist",60,69,.92),_e("fatigue",48,68,.75)),),
    CommercialGymEpisode("trainer_edge_rep_set",88,"high",90,True,.36,.022,(
        _e("background_traffic",0,88,.90),_e("known_mirror",0,88),_e("trainer_presence",8,83,.90),
        _e("trainer_demo",16,36,.85),_e("trainer_contact",44,47,.50),_e("spotter",55,82,.75),
        _e("trainer_assist",70,82,.52),),rep_styles=((4,"partial"),(6,"failed"),(8,"grinder"),(9,"bounce")),rest_pause_after_rep=5),
)


def _stable_seed(*parts: object) -> int:
    raw="|".join(map(str,parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw,digest_size=8).digest(),"little") & 0x7FFFFFFF


def _active(ep: CommercialGymEpisode, t_s: float) -> list[EpisodeEvent]:
    return [e for e in ep.events if e.active(t_s)]


def _rep_times(ep: CommercialGymEpisode, count: int = 10, *, seed: int | None = None) -> list[float]:
    start=max(7.0,ep.duration_s*.16)
    end=ep.duration_s-5.0
    if ep.rest_pause_after_rep is None:
        weights=np.linspace(.85,1.25,count-1); weights=weights/weights.sum()*(end-start)
        times=[start]
        for w in weights: times.append(times[-1]+float(w))
    else:
        k=ep.rest_pause_after_rep
        rest=8.0
        first=np.linspace(start, ep.duration_s*.48, k)
        second=np.linspace(ep.duration_s*.48+rest, end, count-k)
        times=[float(x) for x in np.concatenate([first,second])]
    if seed is not None and len(times)>2:
        rng=np.random.default_rng(_stable_seed(ep.id,seed,"rep_timeline"))
        jitter=rng.normal(0,.16,len(times))
        jitter[0]=0.0; jitter[-1]=0.0
        times=[float(t+j) for t,j in zip(times,jitter)]
        for i in range(1,len(times)):
            times[i]=max(times[i],times[i-1]+2.35)
        if times[-1]>end+.25:
            shift=times[-1]-(end+.25)
            times=[t-shift*(i/(len(times)-1)) for i,t in enumerate(times)]
    return [float(x) for x in times]


def _rep_truths(ep: CommercialGymEpisode, *, seed: int | None = None) -> list[RepTruth]:
    styles=dict(ep.rep_styles)
    rng=np.random.default_rng(_stable_seed(ep.id,seed if seed is not None else 0,"rep_kinematics"))
    out=[]
    for i,complete in enumerate(_rep_times(ep,seed=seed),1):
        style=styles.get(i,"normal")
        down=float(rng.uniform(.68,1.08)); up=float(rng.uniform(.68,1.15))
        if style=="grinder": up=float(rng.uniform(2.2,3.2))
        start=complete-(down+up)
        bottom=complete-up
        out.append(RepTruth(i,start,bottom,complete,style))
    return out


def _depth_for_rep(t: float, r: RepTruth) -> float | None:
    if t < r.start_s or t > r.complete_s: return None
    if r.style=="partial":
        mid=(r.start_s+r.complete_s)/2
        if t<=mid: return .62*(t-r.start_s)/(mid-r.start_s)
        return .62*(r.complete_s-t)/(r.complete_s-mid)
    if r.style=="failed":
        if t<=r.bottom_s: return (t-r.start_s)/(r.bottom_s-r.start_s)
        x=(t-r.bottom_s)/(r.complete_s-r.bottom_s)
        return 1.0-.55*x
    if t<=r.bottom_s:
        x=(t-r.start_s)/(r.bottom_s-r.start_s)
        d=float(np.clip(x,0,1))
        if r.style=="bounce" and x>.84:
            d=float(np.clip(.93+.06*np.sin((x-.84)/.16*np.pi*4),0,1))
        return d
    x=(t-r.bottom_s)/(r.complete_s-r.bottom_s)
    return float(np.clip(1-x,0,1))


def _target_depth(t: float, reps: list[RepTruth]) -> tuple[float,int|None]:
    for r in reps:
        d=_depth_for_rep(t,r)
        if d is not None: return d,r.index
    return 0.0,None


def _trainer_demo_depth(t: float, ep: CommercialGymEpisode) -> float:
    active=[e for e in ep.events if e.kind=="trainer_demo" and e.active(t)]
    if not active: return 0.0
    e=active[0]; period=2.7
    p=((t-e.start_s)%period)/period
    return 2*p if p<.5 else 2*(1-p)


def _background_depth(t: float, ep: CommercialGymEpisode, actor_i: int) -> float:
    period=3.0+.35*actor_i; phase=(actor_i*.23)%1
    p=((t/period)+phase)%1
    return float(.92*(2*p if p<.5 else 2*(1-p)))


def _crowd_count(level: str) -> int:
    return {"low":1,"medium":3,"high":6}[level]
