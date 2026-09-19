from __future__ import annotations
import argparse
import json
from pathlib import Path

from .envcheck import print_environment_report
from .subject_lock import run_subject_lock_benchmark, run_temporal_subject_lock_benchmark
from .false_cue import run_false_cue_benchmark
from .reporting import false_cue_summary, FalseCueGate, write_json_report
from .runner import Harness
from .gym_episode import run_commercial_gym_episode_benchmark
from .rep_detection import run_rep_detection_benchmark
from .exercise_kinematics import run_kinematic_model_benchmark
from .fit3d_adapter import Fit3DLocalAdapter, run_fit3d_rep_replay
from .profiles import external_subjects, PERSONAL_VARIANTS


def main(argv=None):
    p=argparse.ArgumentParser(prog="gym-buddy-harness")
    sub=p.add_subparsers(dest="cmd",required=True)
    sub.add_parser("env-check")


    f3=sub.add_parser("fit3d-check")
    f3.add_argument("--root",default=None,help="Fit3D dataset root; defaults to FIT3D_ROOT")
    f3.add_argument("--load",action="store_true",help="load and validate joint arrays + rep intervals")
    f3.add_argument("--limit",type=int,default=None)
    f3.add_argument("--out",default=None,help="optional JSON summary output")


    f3r=sub.add_parser("fit3d-rep-replay")
    f3r.add_argument("--root",default=None,help="Fit3D dataset root; defaults to FIT3D_ROOT")
    f3r.add_argument("--limit",type=int,default=None)
    f3r.add_argument("--out",default="artifacts/fit3d_rep_replay.csv")

    f3i=sub.add_parser("fit3d-index")
    f3i.add_argument("--root",default=None,help="Fit3D dataset root; defaults to FIT3D_ROOT")
    f3i.add_argument("--out",default="artifacts/fit3d_index.csv")

    rd=sub.add_parser("rep-detection")
    rd.add_argument("--personal",action="store_true")
    rd.add_argument("--fps",type=int,default=20)
    rd.add_argument("--out",default="artifacts/rep_detection.csv")

    kin=sub.add_parser("kinematic-models")
    kin.add_argument("--personal",action="store_true")
    kin.add_argument("--fps",type=int,default=20)
    kin.add_argument("--out",default="artifacts/kinematic_models.csv")

    det=sub.add_parser("deterministic")
    det.add_argument("--personal",action="store_true")
    det.add_argument("--calibrated",action="store_true")
    det.add_argument("--out",default="artifacts/deterministic.csv")

    sl=sub.add_parser("subject-lock")
    sl.add_argument("--seeds",type=int,default=500)
    sl.add_argument("--temporal",action="store_true")
    sl.add_argument("--out",default="artifacts/subject_lock_benchmark.csv")

    fc=sub.add_parser("false-cue")
    fc.add_argument("--n",type=int,default=100000)
    fc.add_argument("--personal",action="store_true")
    fc.add_argument("--calibrated",action="store_true"); fc.add_argument("--mode",choices=["natural","boundary"],default="natural")
    fc.add_argument("--out",default="artifacts/false_cue_benchmark.csv")
    fc.add_argument("--report",default="artifacts/false_cue_summary.json")
    fc.add_argument("--overall-max",type=float,default=.02)
    fc.add_argument("--exercise-max",type=float,default=.04)
    fc.add_argument("--subject-max",type=float,default=.05)
    fc.add_argument("--fail-on-gate",action="store_true")


    cfz=sub.add_parser("composite-fuzz")
    cfz.add_argument("--n",type=int,default=20000)
    cfz.add_argument("--personal",action="store_true")
    cfz.add_argument("--seed",type=int,default=20260919)
    cfz.add_argument("--calibrated",action="store_true")
    cfz.add_argument("--max-components",type=int,default=3)
    cfz.add_argument("--out",default="artifacts/composite_fuzz.csv")

    cg=sub.add_parser("commercial-gym")
    cg.add_argument("--personal",action="store_true")
    cg.add_argument("--calibrated",action="store_true")
    cg.add_argument("--out",default="artifacts/commercial_gym.csv")


    cge=sub.add_parser("commercial-gym-episodes")
    cge.add_argument("--personal",action="store_true")
    cge.add_argument("--seeds-per-episode",type=int,default=1)
    cge.add_argument("--fps",type=int,default=10)
    cge.add_argument("--out",default="artifacts/commercial_gym_episodes.csv")

    val=sub.add_parser("validate-profiles")
    val.add_argument("--out-dir",default="artifacts/profile_ordered_validation")
    val.add_argument("--false-cue-n",type=int,default=20000)
    val.add_argument("--subject-lock-seeds",type=int,default=500)
    val.add_argument("--quick",action="store_true",help="run one representative anonymous profile, then the personal profile stage")
    val.add_argument("--fps",type=int,default=10,help="episode sampling rate for --quick")

    fz=sub.add_parser("fuzz")
    fz.add_argument("--n",type=int,default=20000)
    fz.add_argument("--seed",type=int,default=20260919)
    fz.add_argument("--calibrated",action="store_true")
    fz.add_argument("--out",default="artifacts/fuzz.csv")

    a=p.parse_args(argv)
    if a.cmd=="env-check": return print_environment_report()
    if a.cmd=="fit3d-check":
        adapter=Fit3DLocalAdapter(a.root)
        summary=adapter.summary(load_records=a.load,limit=a.limit)
        if a.out:
            Path(a.out).parent.mkdir(parents=True,exist_ok=True)
            Path(a.out).write_text(json.dumps(summary,indent=2),encoding="utf-8")
        print(json.dumps(summary,indent=2))
        ok=summary["exists"] and summary["train_exists"] and summary["record_count"]>0
        return 0 if ok else 2
    if a.cmd=="fit3d-rep-replay":
        adapter=Fit3DLocalAdapter(a.root)
        df=run_fit3d_rep_replay(adapter,limit=a.limit)
        Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False)
        run=df[df.status=="RUN"] if len(df) else df
        passed=int(run.passed.fillna(False).sum()) if len(run) else 0
        print(f"fit3d_rep_replay_run={len(run)} pass={passed} skipped={len(df)-len(run)}")
        if len(run) and (~run.passed.astype(bool)).any():
            print(run.loc[~run.passed.astype(bool)].head(20).to_string(index=False))
        return 0
    if a.cmd=="fit3d-index":
        adapter=Fit3DLocalAdapter(a.root)
        df=adapter.index_dataframe()
        Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False)
        print(f"fit3d_records={len(df)} actions={df.action.nunique() if len(df) else 0} subjects={df.subject.nunique() if len(df) else 0}")
        return 0 if len(df)>0 else 2
    if a.cmd=="rep-detection":
        subjects=PERSONAL_VARIANTS if a.personal else external_subjects()
        df=run_rep_detection_benchmark(subjects,fps=a.fps)
        Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False)
        print(f"rep_detection_pass_rate={df.passed.mean():.6f} ({int(df.passed.sum())}/{len(df)})")
        if (~df.passed).any(): print(df.loc[~df.passed].head(20).to_string(index=False))
        return 0 if df.passed.all() else 2
    if a.cmd=="kinematic-models":
        subjects=PERSONAL_VARIANTS if a.personal else external_subjects()
        df=run_kinematic_model_benchmark(subjects,fps=a.fps)
        Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False)
        print(f"kinematic_model_pass_rate={df.passed.mean():.6f} ({int(df.passed.sum())}/{len(df)})")
        if (~df.passed).any(): print(df.loc[~df.passed].head(20).to_string(index=False))
        return 0 if df.passed.all() else 2
    if a.cmd=="deterministic":
        subjects=PERSONAL_VARIANTS if a.personal else external_subjects()
        df=Harness().deterministic(subjects,calibrated=a.calibrated)
        Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False)
        print(f"deterministic_pass_rate={df.passed.mean():.6f} ({int(df.passed.sum())}/{len(df)})")
        return 0 if df.passed.all() else 2
    if a.cmd=="subject-lock":
        df=run_temporal_subject_lock_benchmark(a.seeds) if a.temporal else run_subject_lock_benchmark(a.seeds)
        Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False)
        if a.temporal:
            print(df.groupby("scenario").passed.mean().sort_values().to_string())
        else:
            print(df.groupby("strategy").passed.mean().sort_values(ascending=False).to_string())
        return 0 if df.passed.all() else 2
    if a.cmd=="false-cue":
        subjects=PERSONAL_VARIANTS if a.personal else external_subjects()
        df=run_false_cue_benchmark(a.n,calibrated=a.calibrated,mode=a.mode,subjects=subjects)
        Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False)
        summary=false_cue_summary(df,FalseCueGate(a.overall_max,a.exercise_max,a.subject_max))
        write_json_report(summary,a.report)
        print(json.dumps(summary,indent=2))
        return 2 if a.fail_on_gate and not summary["passed_gate"] else 0
    if a.cmd=="composite-fuzz":
        subjects=PERSONAL_VARIANTS if a.personal else external_subjects()
        df=Harness().composite_fuzz(a.n,seed=a.seed,calibrated=a.calibrated,subjects=subjects,max_components=a.max_components)
        Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False)
        print(f"composite_fuzz_pass_rate={df.passed.mean():.6f} ({int(df.passed.sum())}/{len(df)})")
        if (~df.passed).any():
            print(df.loc[~df.passed,'components'].value_counts().head(15).to_string())
        return 0

    if a.cmd=="commercial-gym":
        subjects=PERSONAL_VARIANTS if a.personal else external_subjects()
        df=Harness().commercial_gym_deterministic(subjects,calibrated=a.calibrated)
        Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False)
        print(f"commercial_gym_pass_rate={df.passed.mean():.6f} ({int(df.passed.sum())}/{len(df)})")
        return 0 if df.passed.all() else 2


    if a.cmd=="commercial-gym-episodes":
        subjects=PERSONAL_VARIANTS if a.personal else external_subjects()
        df=run_commercial_gym_episode_benchmark(subjects,seeds_per_episode=a.seeds_per_episode,fps=a.fps)
        Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False)
        print(f"commercial_gym_episode_pass_rate={df.passed.mean():.6f} ({int(df.passed.sum())}/{len(df)})")
        print(f"wrong_identity_locks={int(df.wrong_identity_locks.sum())} false_reps={int(df.false_reps.sum())} unsafe_biomechanics_frames={int(df.unsafe_biomechanics_frames.sum())} max_reacquire_s={float(df.max_reacquire_s.max()):.3f}")
        return 0 if df.passed.all() else 2

    if a.cmd=="validate-profiles":
        from .validation import run_external_then_personal_validation, run_representative_then_personal_validation
        if a.quick:
            result=run_representative_then_personal_validation(a.out_dir,fps=a.fps)
        else:
            result=run_external_then_personal_validation(a.out_dir,false_cue_n=a.false_cue_n,subject_lock_seeds=a.subject_lock_seeds)
        print(json.dumps(result,indent=2))
        blocking=[x for x in result["stages"] if x["passed"] is not None]
        return 0 if all(x["passed"] for x in blocking) else 2

    if a.cmd=="fuzz":
        subjects=PERSONAL_VARIANTS if a.personal else external_subjects()
        df=Harness().fuzz(a.n,seed=a.seed,calibrated=a.calibrated,subjects=subjects)
        Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False)
        print(f"fuzz_pass_rate={df.passed.mean():.6f} ({int(df.passed.sum())}/{len(df)})")
        return 0

if __name__=="__main__": raise SystemExit(main())
