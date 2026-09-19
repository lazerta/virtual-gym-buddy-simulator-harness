from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.false_cue import run_false_cue_benchmark
from harness.profiles import SHAWN_VARIANTS, external_subjects
from harness.render import render_demo
from harness.runner import Harness
from harness.scenarios import FAMILIES
from harness.subject_lock import run_subject_lock_benchmark


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fuzz-n',type=int,default=20_000)
    p.add_argument('--false-cue-n',type=int,default=20_000)
    p.add_argument('--subject-lock-seeds',type=int,default=500)
    p.add_argument('--full',action='store_true',help='Use 100k fuzz + 100k false-cue cases per mode')
    a=p.parse_args()
    if a.full:
        a.fuzz_n=100_000; a.false_cue_n=100_000; a.subject_lock_seeds=max(a.subject_lock_seeds,1000)

    root=Path(__file__).parent; outdir=root/'artifacts'; outdir.mkdir(exist_ok=True)
    h=Harness()
    d=h.deterministic(external_subjects(),calibrated=False); d.to_csv(outdir/'deterministic_subject_lock.csv',index=False)
    s=h.deterministic(SHAWN_VARIANTS,calibrated=True); s.to_csv(outdir/'shawn_subject_lock.csv',index=False)
    fg=h.fuzz(a.fuzz_n,seed=20260919,calibrated=False); fc=h.fuzz(a.fuzz_n,seed=20260919,calibrated=True)
    fg.to_csv(outdir/f'fuzz_generic_{a.fuzz_n}.csv',index=False); fc.to_csv(outdir/f'fuzz_calibrated_{a.fuzz_n}.csv',index=False)
    sl=run_subject_lock_benchmark(a.subject_lock_seeds); sl.to_csv(outdir/'subject_lock_benchmark.csv',index=False)
    fcg=run_false_cue_benchmark(a.false_cue_n,seed=20260919,calibrated=False); fcg.to_csv(outdir/f'false_cue_generic_{a.false_cue_n}.csv',index=False)
    fcc=run_false_cue_benchmark(a.false_cue_n,seed=20260919,calibrated=True); fcc.to_csv(outdir/f'false_cue_calibrated_{a.false_cue_n}.csv',index=False)
    summary={
      'scenario_family_count':len(FAMILIES),
      'deterministic':{'pass':int(d.passed.sum()),'total':len(d),'rate':float(d.passed.mean())},
      'shawn_variants':{'pass':int(s.passed.sum()),'total':len(s),'rate':float(s.passed.mean())},
      f'fuzz_generic_{a.fuzz_n}':{'pass':int(fg.passed.sum()),'total':len(fg),'rate':float(fg.passed.mean())},
      f'fuzz_calibrated_{a.fuzz_n}':{'pass':int(fc.passed.sum()),'total':len(fc),'rate':float(fc.passed.mean())},
      'generic_failures_by_scenario':fg.loc[~fg.passed].scenario.value_counts().to_dict(),
      'calibrated_failures_by_scenario':fc.loc[~fc.passed].scenario.value_counts().to_dict(),
      'subject_lock_pass_rate_by_strategy':sl.groupby('strategy').passed.mean().to_dict(),
      'subject_lock_failures_by_scenario_strategy':{f'{k[0]}::{k[1]}':int(v) for k,v in sl.loc[~sl.passed].groupby(['strategy','scenario']).size().to_dict().items()},
      f'false_cue_generic_{a.false_cue_n}':{'false_cues':int(fcg.false_cue.sum()),'total':len(fcg),'rate':float(fcg.false_cue.mean())},
      f'false_cue_calibrated_{a.false_cue_n}':{'false_cues':int(fcc.false_cue.sum()),'total':len(fcc),'rate':float(fcc.false_cue.mean())},
    }
    (outdir/'summary_latest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    render_demo(outdir/'demo_subject_lock.mp4')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
