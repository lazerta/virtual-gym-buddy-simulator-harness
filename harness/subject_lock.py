from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateEvidence:
    track_id: str
    motion: float
    pose_geometry: float
    appearance: float
    spatial: float
    face: float | None = None
    in_known_mirror: bool = False
    observability: float = 1.0


@dataclass(frozen=True)
class LockDecision:
    target_track_id: str | None
    confidence: float
    identity_margin: float
    state: str


@dataclass(frozen=True)
class SubjectLockStrategy:
    name: str
    motion_w: float
    pose_w: float
    appearance_w: float
    spatial_w: float
    face_w: float = 0.0
    mirror_penalty: float = 0.25
    min_score: float = 0.56
    min_margin: float = 0.10
    min_observability: float = 0.45

    def score(self, c: CandidateEvidence) -> float:
        face = 0.5 if c.face is None else c.face
        denom = self.motion_w + self.pose_w + self.appearance_w + self.spatial_w + self.face_w
        raw = (
            self.motion_w * c.motion
            + self.pose_w * c.pose_geometry
            + self.appearance_w * c.appearance
            + self.spatial_w * c.spatial
            + self.face_w * face
        ) / max(denom, 1e-9)
        if c.in_known_mirror:
            raw -= self.mirror_penalty
        # Target observability is a downstream biomechanics gate, not an identity score multiplier.
        # Keeping them separate lets the tracker preserve identity through short occlusions.
        return float(np.clip(raw, 0.0, 1.0))

    def choose(self, candidates: list[CandidateEvidence]) -> LockDecision:
        if not candidates:
            return LockDecision(None, 0.0, 0.0, "TARGET_LOST")
        ranked = sorted(((self.score(c), c.track_id) for c in candidates), reverse=True)
        best_score, best_id = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        margin = best_score - second
        best = next(c for c in candidates if c.track_id == best_id)
        if best_score < self.min_score:
            return LockDecision(None, best_score, margin, "TARGET_LOST")
        # Identity can remain known while the target is too occluded for biomechanics.
        # Preserve the track id but expose a distinct state so downstream analysis pauses.
        if best.observability < self.min_observability:
            return LockDecision(best_id, best_score, margin, "TARGET_OCCLUDED")
        if margin < self.min_margin:
            return LockDecision(None, best_score, margin, "TARGET_AMBIGUOUS")
        return LockDecision(best_id, best_score, margin, "TARGET_LOCKED")


STRATEGIES = [
    SubjectLockStrategy("motion_only", 0.70, 0.0, 0.0, 0.30, min_margin=0.12),
    SubjectLockStrategy("motion_pose", 0.40, 0.35, 0.0, 0.25, min_margin=0.10),
    SubjectLockStrategy("motion_pose_appearance", 0.30, 0.25, 0.30, 0.15, min_margin=0.09),
    SubjectLockStrategy("multimodal_personal", 0.25, 0.25, 0.28, 0.12, 0.10, min_margin=0.08),
]


def _scenario_candidates(name: str, rng: np.random.Generator) -> tuple[list[CandidateEvidence], str | None, str]:
    jitter=lambda s=.025: float(rng.normal(0, s))
    target = CandidateEvidence("target", .94+jitter(), .93+jitter(), .91+jitter(), .95+jitter(), .86+jitter(), False, .96)
    if name == "alone":
        return [target], "target", "TARGET_LOCKED"
    if name == "background_bystander":
        other = CandidateEvidence("other", .18+jitter(), .42+jitter(), .26+jitter(), .20+jitter(), .20+jitter(), False, .95)
        return [target, other], "target", "TARGET_LOCKED"
    if name == "spotter":
        other = CandidateEvidence("spotter", .48+jitter(), .50+jitter(), .31+jitter(), .55+jitter(), .25+jitter(), False, .92)
        return [target, other], "target", "TARGET_LOCKED"
    if name == "foreground_cross":
        other = CandidateEvidence("crossing", .63+jitter(), .43+jitter(), .30+jitter(), .62+jitter(), .20+jitter(), False, .95)
        return [target, other], "target", "TARGET_LOCKED"
    if name == "known_mirror":
        refl = CandidateEvidence("reflection", .90+jitter(), .92+jitter(), .90+jitter(), .80+jitter(), .80+jitter(), True, .95)
        return [target, refl], "target", "TARGET_LOCKED"
    if name == "lookalike":
        look = CandidateEvidence("lookalike", .87+jitter(), .89+jitter(), .88+jitter(), .84+jitter(), .72+jitter(), False, .96)
        return [target, look], None, "TARGET_AMBIGUOUS"
    if name == "full_occlusion":
        hidden = CandidateEvidence("target", .60+jitter(), .75+jitter(), .70+jitter(), .68+jitter(), None, False, .25)
        other = CandidateEvidence("other", .58+jitter(), .55+jitter(), .44+jitter(), .61+jitter(), .30+jitter(), False, .95)
        return [hidden, other], "target", "TARGET_OCCLUDED"
    if name == "subject_swap":
        swapped = CandidateEvidence("wrong_person", .91+jitter(), .62+jitter(), .55+jitter(), .91+jitter(), .35+jitter(), False, .97)
        weak_target = CandidateEvidence("target", .35+jitter(), .88+jitter(), .86+jitter(), .25+jitter(), .75+jitter(), False, .95)
        return [swapped, weak_target], None, "TARGET_AMBIGUOUS"
    if name == "exit":
        return [], None, "TARGET_LOST"
    raise KeyError(name)


def run_subject_lock_benchmark(seeds: int = 500) -> pd.DataFrame:
    scenarios = ["alone", "background_bystander", "spotter", "foreground_cross", "known_mirror", "lookalike", "full_occlusion", "subject_swap", "exit"]
    rows=[]
    for seed in range(seeds):
        rng=np.random.default_rng(100000+seed)
        for scenario in scenarios:
            candidates, expected_id, expected_state = _scenario_candidates(scenario, rng)
            for strategy in STRATEGIES:
                d = strategy.choose(candidates)
                passed = d.state == expected_state and (expected_id is None or d.target_track_id == expected_id)
                rows.append({
                    "seed":seed,"scenario":scenario,"strategy":strategy.name,"passed":passed,
                    "expected_state":expected_state,"observed_state":d.state,
                    "expected_target":expected_id,"observed_target":d.target_track_id,
                    "confidence":d.confidence,"identity_margin":d.identity_margin,
                })
    return pd.DataFrame(rows)

@dataclass
class TemporalLockState:
    locked_track_id: str | None = None
    state: str = "NO_TARGET"
    lost_frames: int = 0
    reacquire_candidate_id: str | None = None
    reacquire_streak: int = 0


class TemporalPrimarySubjectLock:
    """Stateful primary-subject tracker built on target-similarity evidence.

    The tracker never silently jumps from the current target to a competing person.
    If identity evidence becomes close, it emits TARGET_AMBIGUOUS. During short full
    occlusions it preserves the previous identity internally and waits for confident
    re-identification before resuming.
    """

    def __init__(
        self,
        strategy: SubjectLockStrategy,
        *,
        occlusion_grace_frames: int = 8,
        reacquire_frames: int = 2,
        hold_score: float | None = None,
        switch_margin: float = 0.16,
    ):
        self.strategy = strategy
        self.occlusion_grace_frames = occlusion_grace_frames
        self.reacquire_frames = reacquire_frames
        self.hold_score = strategy.min_score - 0.05 if hold_score is None else hold_score
        self.switch_margin = switch_margin
        self.s = TemporalLockState()

    def reset(self) -> None:
        self.s = TemporalLockState()

    def _rank(self, candidates: list[CandidateEvidence]):
        return sorted(((self.strategy.score(c), c) for c in candidates), key=lambda x: x[0], reverse=True)

    def step(self, candidates: list[CandidateEvidence]) -> LockDecision:
        if not candidates:
            self.s.lost_frames += 1
            self.s.reacquire_candidate_id = None
            self.s.reacquire_streak = 0
            if self.s.locked_track_id is not None and self.s.lost_frames <= self.occlusion_grace_frames:
                self.s.state = "TARGET_OCCLUDED"
                return LockDecision(self.s.locked_track_id, 0.0, 0.0, self.s.state)
            self.s.state = "TARGET_LOST"
            return LockDecision(None, 0.0, 0.0, self.s.state)

        ranked = self._rank(candidates)
        best_score, best = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        best_margin = best_score - second_score

        # First acquisition uses the normal per-frame gate, but requires no history.
        if self.s.locked_track_id is None and self.s.state in {"NO_TARGET", "TARGET_LOST"}:
            d = self.strategy.choose(candidates)
            if d.state == "TARGET_LOCKED":
                self.s.locked_track_id = d.target_track_id
                self.s.lost_frames = 0
                self.s.state = d.state
            else:
                self.s.state = d.state
            return d

        current = next((c for c in candidates if c.track_id == self.s.locked_track_id), None)
        if current is not None:
            current_score = self.strategy.score(current)
            competitor_scores = [self.strategy.score(c) for c in candidates if c.track_id != current.track_id]
            competitor = max(competitor_scores, default=0.0)
            margin = current_score - competitor
            self.s.lost_frames = 0
            self.s.reacquire_candidate_id = None
            self.s.reacquire_streak = 0

            if current.observability < self.strategy.min_observability:
                self.s.state = "TARGET_OCCLUDED"
                return LockDecision(current.track_id, current_score, margin, self.s.state)

            # A close competitor is ambiguity, not permission to switch identities.
            if margin < self.strategy.min_margin:
                self.s.state = "TARGET_AMBIGUOUS"
                return LockDecision(None, current_score, margin, self.s.state)

            # Strong evidence that another candidate displaced the current track is also
            # treated conservatively as ambiguity. We wait for re-identification rather
            # than silently following the new person.
            if competitor - current_score > self.switch_margin or current_score < self.hold_score:
                self.s.state = "TARGET_AMBIGUOUS"
                return LockDecision(None, current_score, margin, self.s.state)

            self.s.state = "TARGET_LOCKED"
            return LockDecision(current.track_id, current_score, margin, self.s.state)

        # The previous tracker id disappeared. Keep identity continuity through a short
        # occlusion and require a small stable window before accepting a new tracker id.
        self.s.lost_frames += 1
        if best_score < self.strategy.min_score or best_margin < self.strategy.min_margin:
            self.s.reacquire_candidate_id = None
            self.s.reacquire_streak = 0
            self.s.state = "TARGET_OCCLUDED" if self.s.lost_frames <= self.occlusion_grace_frames else "TARGET_AMBIGUOUS"
            return LockDecision(self.s.locked_track_id if self.s.state == "TARGET_OCCLUDED" else None, best_score, best_margin, self.s.state)

        if self.s.reacquire_candidate_id == best.track_id:
            self.s.reacquire_streak += 1
        else:
            self.s.reacquire_candidate_id = best.track_id
            self.s.reacquire_streak = 1

        if self.s.reacquire_streak >= self.reacquire_frames:
            self.s.locked_track_id = best.track_id
            self.s.lost_frames = 0
            self.s.reacquire_candidate_id = None
            self.s.reacquire_streak = 0
            self.s.state = "TARGET_LOCKED" if best.observability >= self.strategy.min_observability else "TARGET_OCCLUDED"
            return LockDecision(best.track_id, best_score, best_margin, self.s.state)

        self.s.state = "TARGET_OCCLUDED"
        return LockDecision(self.s.locked_track_id, best_score, best_margin, self.s.state)


def _temporal_sequence(name: str, rng: np.random.Generator):
    """Return frames plus hidden truth labels used only by the benchmark oracle."""
    j=lambda s=.018: float(rng.normal(0,s))
    def target(track="target", obs=.97, motion=.94, spatial=.95):
        return CandidateEvidence(track,motion+j(),.94+j(),.92+j(),spatial+j(),.87+j(),False,obs)
    def other(track="other", motion=.30, pose=.45, app=.30, spatial=.35, obs=.95):
        return CandidateEvidence(track,motion+j(),pose+j(),app+j(),spatial+j(),.25+j(),False,obs)

    warm=[[target()] for _ in range(5)]
    if name=="background_bystander":
        frames=warm+[[target(),other()] for _ in range(15)]
        return frames,["target"]*len(frames),{"may_pause":False}
    if name=="spotter":
        spot=lambda: other("spotter",.50,.52,.32,.58,.94)
        frames=warm+[[target(),spot()] for _ in range(15)]
        return frames,["target"]*len(frames),{"may_pause":False}
    if name=="foreground_cross":
        crossing=lambda: other("cross",.72,.48,.34,.72,.96)
        frames=warm+[[target(),crossing()] for _ in range(5)]+[[target()] for _ in range(10)]
        return frames,["target"]*len(frames),{"may_pause":False}
    if name=="full_occlusion":
        frames=warm+[[] for _ in range(4)]+[[target()] for _ in range(8)]
        return frames,["target"]*len(frames),{"occlusion_range":range(5,9),"max_reacquire":2}
    if name=="track_id_change":
        frames=warm+[[] for _ in range(3)]+[[target("target_reid")]]*8
        truth=["target"]*len(frames)
        return frames,truth,{"occlusion_range":range(5,8),"max_reacquire":3,"reid_track":"target_reid"}
    if name=="lookalike":
        look=lambda: CandidateEvidence("lookalike",.90+j(),.92+j(),.91+j(),.89+j(),.78+j(),False,.98)
        frames=warm+[[target(),look()] for _ in range(8)]+[[target()] for _ in range(7)]
        return frames,["target"]*len(frames),{"ambiguous_range":range(5,13)}
    if name=="known_mirror":
        refl=lambda: CandidateEvidence("reflection",.93+j(),.94+j(),.93+j(),.82+j(),.84+j(),True,.98)
        frames=warm+[[target(),refl()] for _ in range(15)]
        return frames,["target"]*len(frames),{"may_pause":False}
    if name=="subject_swap":
        wrong=lambda: CandidateEvidence("wrong",.95+j(),.65+j(),.57+j(),.95+j(),.35+j(),False,.98)
        weak=lambda: CandidateEvidence("target",.40+j(),.90+j(),.88+j(),.28+j(),.78+j(),False,.97)
        frames=warm+[[weak(),wrong()] for _ in range(8)]+[[target()] for _ in range(7)]
        return frames,["target"]*len(frames),{"ambiguous_range":range(5,13)}
    if name=="exit_reenter":
        frames=warm+[[] for _ in range(12)]+[[target("target_reid")]]*8
        return frames,["target"]*len(frames),{"lost_range":range(13,17),"max_reacquire":3,"reid_track":"target_reid"}
    raise KeyError(name)


def run_temporal_subject_lock_benchmark(seeds: int = 250, strategy: SubjectLockStrategy | None = None) -> pd.DataFrame:
    """Sequence benchmark for occlusion/re-identification/identity continuity."""
    strategy = strategy or STRATEGIES[-1]
    scenarios=["background_bystander","spotter","foreground_cross","full_occlusion","track_id_change","lookalike","known_mirror","subject_swap","exit_reenter"]
    rows=[]
    for seed in range(seeds):
        rng=np.random.default_rng(220000+seed)
        for scenario in scenarios:
            frames, truth, rules=_temporal_sequence(scenario,rng)
            tracker=TemporalPrimarySubjectLock(strategy)
            wrong_locks=0; pauses=0; unsafe_locked_in_ambiguous=0; states=[]; ids=[]
            reacquire_frame=None
            for fi,cands in enumerate(frames):
                d=tracker.step(cands); states.append(d.state); ids.append(d.target_track_id)
                # Any explicit lock to a non-target identity is always unsafe.
                if d.state=="TARGET_LOCKED" and d.target_track_id not in {"target","target_reid"}:
                    wrong_locks+=1
                if d.state in {"TARGET_AMBIGUOUS","TARGET_LOST","TARGET_OCCLUDED"}:
                    pauses+=1
                if fi in rules.get("ambiguous_range",()) and d.state=="TARGET_LOCKED":
                    unsafe_locked_in_ambiguous+=1
                if rules.get("reid_track") and d.state=="TARGET_LOCKED" and d.target_track_id==rules["reid_track"] and reacquire_frame is None:
                    reacquire_frame=fi

            passed = wrong_locks==0 and unsafe_locked_in_ambiguous==0
            if rules.get("may_pause") is False:
                passed = passed and pauses==0
            if "max_reacquire" in rules:
                reid_start = next((i for i,f in enumerate(frames) if any(c.track_id==rules.get("reid_track","target") for c in f)), None)
                if rules.get("reid_track"):
                    passed = passed and reacquire_frame is not None and reacquire_frame-reid_start <= rules["max_reacquire"]
                else:
                    # Same track id after short occlusion should resume immediately or near-immediately.
                    after=max(rules["occlusion_range"])+1
                    resume=next((i for i in range(after,len(states)) if states[i]=="TARGET_LOCKED"),None)
                    passed=passed and resume is not None and resume-after <= rules["max_reacquire"]
            rows.append({
                "seed":seed,"scenario":scenario,"strategy":strategy.name,"passed":passed,
                "wrong_identity_locks":wrong_locks,"pause_frames":pauses,
                "unsafe_locked_in_ambiguous":unsafe_locked_in_ambiguous,
                "reacquire_frame":reacquire_frame,"final_state":states[-1],"final_target":ids[-1],
            })
    return pd.DataFrame(rows)
