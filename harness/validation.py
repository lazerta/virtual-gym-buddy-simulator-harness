from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

from .runner import Harness
from .profiles import external_subjects, PERSONAL_VARIANTS
from .false_cue import run_false_cue_benchmark
from .reporting import false_cue_summary, FalseCueGate
from .subject_lock import run_temporal_subject_lock_benchmark
from .gym_episode import run_commercial_gym_episode_benchmark
from .rep_detection import run_rep_detection_benchmark

@dataclass
class Stage:
    name: str
    status: str
    passed: bool | None
    details: dict


def _save_df(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_external_then_personal_validation(artifact_dir: str | Path = "artifacts/profile_ordered_validation", *, false_cue_n: int = 20_000, subject_lock_seeds: int = 500, false_cue_gate: FalseCueGate | None = None) -> dict:
    out=Path(artifact_dir); out.mkdir(parents=True,exist_ok=True)
    h=Harness(); stages=[]; ext=external_subjects(); gate=false_cue_gate or FalseCueGate()
    d=h.deterministic(ext, calibrated=False); _save_df(d,out/'01_external_base_deterministic.csv')
    ok=bool(d.passed.all()); stages.append(Stage('external_base_deterministic','PASS' if ok else 'FAIL',ok,{"passed":int(d.passed.sum()),"total":len(d)}))
    if not ok: return _finish(out,stages,personal_allowed=False)
    rd=run_rep_detection_benchmark(ext); _save_df(rd,out/'02_external_rep_detection.csv')
    ok=bool(rd.passed.all()); stages.append(Stage('external_rep_detection','PASS' if ok else 'FAIL',ok,{"passed":int(rd.passed.sum()),"total":len(rd),"cases":int(rd.case.nunique())}))
    if not ok: return _finish(out,stages,personal_allowed=False)
    g=h.commercial_gym_deterministic(ext, calibrated=False); _save_df(g,out/'03_external_commercial_gym.csv')
    ok=bool(g.passed.all()); stages.append(Stage('external_commercial_gym','PASS' if ok else 'FAIL',ok,{"passed":int(g.passed.sum()),"total":len(g),"scenes":int(g.gym_scene.nunique())}))
    if not ok: return _finish(out,stages,personal_allowed=False)
    ep=run_commercial_gym_episode_benchmark(ext,seeds_per_episode=1,base_seed=920000,fps=10); _save_df(ep,out/'04_external_commercial_gym_episodes.csv')
    ok=bool(ep.passed.all()); stages.append(Stage('external_commercial_gym_episodes','PASS' if ok else 'FAIL',ok,{"passed":int(ep.passed.sum()),"total":len(ep),"episode_templates":int(ep.episode.nunique()),"wrong_identity_locks":int(ep.wrong_identity_locks.sum()),"false_reps":int(ep.false_reps.sum()),"unsafe_biomechanics_frames":int(ep.unsafe_biomechanics_frames.sum()),"max_reacquire_s":float(ep.max_reacquire_s.max())}))
    if not ok: return _finish(out,stages,personal_allowed=False)
    sl=run_temporal_subject_lock_benchmark(subject_lock_seeds); _save_df(sl,out/'05_subject_lock_temporal.csv')
    ok=bool(sl.passed.all()); stages.append(Stage('subject_lock_temporal','PASS' if ok else 'FAIL',ok,{"passed":int(sl.passed.sum()),"total":len(sl)}))
    if not ok: return _finish(out,stages,personal_allowed=False)
    fc=run_false_cue_benchmark(false_cue_n,seed=830001,calibrated=False,mode='natural',subjects=ext); _save_df(fc,out/'06_external_natural_false_cue.csv')
    fcs=false_cue_summary(fc,gate); ok=bool(fcs['passed_gate']); stages.append(Stage('external_natural_false_cue','PASS' if ok else 'FAIL',ok,fcs))
    if not ok: return _finish(out,stages,personal_allowed=False)
    comp=h.composite_fuzz(10_000,seed=830002,calibrated=False,subjects=ext,max_components=3); _save_df(comp,out/'07_external_composite_fuzz.csv')
    stages.append(Stage('external_composite_fuzz','DIAGNOSTIC',None,{"pass_rate":float(comp.passed.mean()),"passed":int(comp.passed.sum()),"total":len(comp)}))
    srd=run_rep_detection_benchmark(PERSONAL_VARIANTS,base_seed=3760000); _save_df(srd,out/'08_personal_rep_detection.csv')
    ok=bool(srd.passed.all()); stages.append(Stage('personal_rep_detection','PASS' if ok else 'FAIL',ok,{"passed":int(srd.passed.sum()),"total":len(srd),"cases":int(srd.case.nunique())}))
    if not ok: return _finish(out,stages,personal_allowed=True)
    sd=h.deterministic(PERSONAL_VARIANTS,calibrated=True); _save_df(sd,out/'09_personal_base_deterministic.csv')
    ok=bool(sd.passed.all()); stages.append(Stage('personal_base_deterministic','PASS' if ok else 'FAIL',ok,{"passed":int(sd.passed.sum()),"total":len(sd)}))
    if not ok: return _finish(out,stages,personal_allowed=True)
    sg=h.commercial_gym_deterministic(PERSONAL_VARIANTS,calibrated=True); _save_df(sg,out/'10_personal_commercial_gym.csv')
    ok=bool(sg.passed.all()); stages.append(Stage('personal_commercial_gym','PASS' if ok else 'FAIL',ok,{"passed":int(sg.passed.sum()),"total":len(sg),"scenes":int(sg.gym_scene.nunique())}))
    if not ok: return _finish(out,stages,personal_allowed=True)
    sep=run_commercial_gym_episode_benchmark(PERSONAL_VARIANTS,seeds_per_episode=1,base_seed=3920000,fps=10); _save_df(sep,out/'11_personal_commercial_gym_episodes.csv')
    ok=bool(sep.passed.all()); stages.append(Stage('personal_commercial_gym_episodes','PASS' if ok else 'FAIL',ok,{"passed":int(sep.passed.sum()),"total":len(sep),"episode_templates":int(sep.episode.nunique()),"wrong_identity_locks":int(sep.wrong_identity_locks.sum()),"false_reps":int(sep.false_reps.sum()),"unsafe_biomechanics_frames":int(sep.unsafe_biomechanics_frames.sum()),"max_reacquire_s":float(sep.max_reacquire_s.max())}))
    if not ok: return _finish(out,stages,personal_allowed=True)
    sfc=run_false_cue_benchmark(max(5000,false_cue_n//4),seed=830003,calibrated=True,mode='natural',subjects=PERSONAL_VARIANTS); _save_df(sfc,out/'12_personal_natural_false_cue.csv')
    sfcs=false_cue_summary(sfc,gate); sok=bool(sfcs['passed_gate']); stages.append(Stage('personal_natural_false_cue','PASS' if sok else 'FAIL',sok,sfcs))
    scomp=h.composite_fuzz(5_000,seed=830004,calibrated=True,subjects=PERSONAL_VARIANTS,max_components=3); _save_df(scomp,out/'13_personal_composite_fuzz.csv')
    stages.append(Stage('personal_composite_fuzz','DIAGNOSTIC',None,{"pass_rate":float(scomp.passed.mean()),"passed":int(scomp.passed.sum()),"total":len(scomp)}))
    return _finish(out,stages,personal_allowed=True)


def _finish(out: Path, stages: list[Stage], personal_allowed: bool) -> dict:
    data={"policy":"external profiles must pass all blocking gates before personal profiles run","personal_stage_allowed":personal_allowed,"stages":[asdict(x) for x in stages]}
    (out/'summary.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    return data


def run_representative_then_personal_validation(artifact_dir: str | Path = "artifacts/representative_then_personal", *, fps: int = 10, representative_profile_id: str = "ansur_median_central") -> dict:
    out=Path(artifact_dir); out.mkdir(parents=True,exist_ok=True); stages=[]
    ext=next((p for p in external_subjects() if p.id==representative_profile_id),None)
    if ext is None: raise KeyError(f"unknown representative profile: {representative_profile_id}")
    rd=run_rep_detection_benchmark([ext],fps=20,base_seed=5010000); _save_df(rd,out/'01_representative_rep_detection.csv')
    ok=bool(rd.passed.all()); stages.append(Stage('representative_rep_detection','PASS' if ok else 'FAIL',ok,{'profile':ext.id,'passed':int(rd.passed.sum()),'total':len(rd),'cases':int(rd.case.nunique())}))
    if not ok: return _finish_quick(out,stages,personal_allowed=False,representative=ext.id)
    ep=run_commercial_gym_episode_benchmark([ext],seeds_per_episode=1,base_seed=5510000,fps=fps); _save_df(ep,out/'02_representative_commercial_gym_episodes.csv')
    ok=bool(ep.passed.all()); stages.append(Stage('representative_commercial_gym_episodes','PASS' if ok else 'FAIL',ok,{'profile':ext.id,'passed':int(ep.passed.sum()),'total':len(ep),'episode_templates':int(ep.episode.nunique()),'wrong_identity_locks':int(ep.wrong_identity_locks.sum()),'false_reps':int(ep.false_reps.sum()),'unsafe_biomechanics_frames':int(ep.unsafe_biomechanics_frames.sum()),'max_reacquire_s':float(ep.max_reacquire_s.max())}))
    if not ok: return _finish_quick(out,stages,personal_allowed=False,representative=ext.id)
    srd=run_rep_detection_benchmark([PERSONAL_VARIANTS[0]],fps=20,base_seed=6010000); _save_df(srd,out/'03_personal_rep_detection.csv')
    ok=bool(srd.passed.all()); stages.append(Stage('personal_rep_detection','PASS' if ok else 'FAIL',ok,{'profile':PERSONAL_VARIANTS[0].id,'passed':int(srd.passed.sum()),'total':len(srd),'cases':int(srd.case.nunique())}))
    if not ok: return _finish_quick(out,stages,personal_allowed=True,representative=ext.id)
    sep=run_commercial_gym_episode_benchmark([PERSONAL_VARIANTS[0]],seeds_per_episode=1,base_seed=6510000,fps=fps); _save_df(sep,out/'04_personal_commercial_gym_episodes.csv')
    ok=bool(sep.passed.all()); stages.append(Stage('personal_commercial_gym_episodes','PASS' if ok else 'FAIL',ok,{'profile':PERSONAL_VARIANTS[0].id,'passed':int(sep.passed.sum()),'total':len(sep),'episode_templates':int(sep.episode.nunique()),'wrong_identity_locks':int(sep.wrong_identity_locks.sum()),'false_reps':int(sep.false_reps.sum()),'unsafe_biomechanics_frames':int(sep.unsafe_biomechanics_frames.sum()),'max_reacquire_s':float(sep.max_reacquire_s.max())}))
    return _finish_quick(out,stages,personal_allowed=True,representative=ext.id)


def _finish_quick(out: Path, stages: list[Stage], personal_allowed: bool, representative: str) -> dict:
    data={'policy':'representative anonymous profile must pass blocking gates before personal profile runs','representative_profile':representative,'personal_stage_allowed':personal_allowed,'stages':[asdict(x) for x in stages]}
    (out/'summary.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    return data
