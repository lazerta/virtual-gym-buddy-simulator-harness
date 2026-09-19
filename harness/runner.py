from __future__ import annotations
import numpy as np
import pandas as pd
from .models import SubjectProfile, ExerciseProfile, Scenario, CaseResult
from .profiles import EXERCISES, external_subjects, SHAWN_VARIANTS
from .scenarios import FAMILIES, deterministic_scenario, fuzz_scenario, fuzz_composite_scenario
from .simulator import simulate
from .oracle import expected_for
from .adapter import ReferenceProductionAdapter, PersonalCalibration

class Harness:
    def __init__(self, adapter=None): self.adapter=adapter or ReferenceProductionAdapter()
    def run_case(self, subj:SubjectProfile, ex:ExerciseProfile, s:Scenario, calibrated=False)->CaseResult:
        obs=simulate(subj,ex,s); exp=expected_for(s)
        cal=None
        if calibrated:
            bias=(subj.torso_ratio-.31)*2.8+(subj.upper_arm_ratio-.186)*3.2
            cal=PersonalCalibration(issue_baseline=.10+bias, confidence=.7)
        out=self.adapter.analyze(obs,ex,cal)
        passed=(out.ready==exp.ready and out.pause==exp.pause and out.reset_count==exp.reset_once and out.rep_count==exp.expected_reps and (out.cue_count>0)==exp.cue_expected)
        return CaseResult(f"{subj.id}:{ex.id}:{s.family}:{s.seed}",passed,exp,out,{"subject":subj.id,"exercise":ex.id,"scenario":s.family,"seed":s.seed,"calibrated":calibrated})

    def deterministic(self, subjects=None, calibrated=False):
        subjects=subjects or external_subjects()
        rows=[]
        seed=1000
        for subj in subjects:
            for ex in EXERCISES:
                for fam in FAMILIES:
                    rows.append(self.run_case(subj,ex,deterministic_scenario(fam,seed),calibrated).row()); seed+=1
        return pd.DataFrame(rows)

    def fuzz(self,n=10000,seed=42,calibrated=False,subjects=None):
        subjects=subjects or external_subjects()
        rng=np.random.default_rng(seed); rows=[]
        for i in range(n):
            subj=subjects[int(rng.integers(0,len(subjects)))]; ex=EXERCISES[int(rng.integers(0,len(EXERCISES)))]
            sc=fuzz_scenario(rng,seed+i+1)
            rows.append(self.run_case(subj,ex,sc,calibrated).row())
        return pd.DataFrame(rows)

    def composite_fuzz(self,n=10000,seed=4242,calibrated=False,subjects=None,max_components=3):
        subjects=subjects or external_subjects()
        rng=np.random.default_rng(seed); rows=[]
        for i in range(n):
            subj=subjects[int(rng.integers(0,len(subjects)))]; ex=EXERCISES[int(rng.integers(0,len(EXERCISES)))]
            sc=fuzz_composite_scenario(rng,seed+i+1,max_components=max_components)
            r=self.run_case(subj,ex,sc,calibrated)
            row=r.row(); row['components']='|'.join(sc.components); row['component_count']=len(sc.components)
            rows.append(row)
        return pd.DataFrame(rows)

    def commercial_gym_deterministic(self, subjects=None, calibrated=False):
        from .gym_env import COMMERCIAL_GYM_SCENES, scenario_from_gym_scene
        subjects=subjects or external_subjects()
        rows=[]
        seed=700000
        for subj in subjects:
            for ex in EXERCISES:
                for scene in COMMERCIAL_GYM_SCENES:
                    sc=scenario_from_gym_scene(scene,seed)
                    r=self.run_case(subj,ex,sc,calibrated)
                    row=r.row()
                    row.update({
                        "gym_scene":scene.id,
                        "crowd_level":scene.crowd_level,
                        "mirror_level":scene.mirror_level,
                        "lighting":scene.lighting,
                        "equipment_clutter":scene.equipment_clutter,
                        "ambient_noise":scene.ambient_noise,
                    })
                    rows.append(row); seed+=1
        return pd.DataFrame(rows)
