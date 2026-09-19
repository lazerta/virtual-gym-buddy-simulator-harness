from harness.profiles import EXERCISES, external_subjects, SHAWN_VARIANTS
from harness.scenarios import deterministic_scenario
from harness.simulator import simulate
from harness.oracle import expected_for
from harness.runner import Harness

def test_seed_reproducible():
    s=deterministic_scenario('clean',123); a=simulate(external_subjects()[0],EXERCISES[0],s); b=simulate(external_subjects()[0],EXERCISES[0],s); assert a==b

def test_oracle_does_not_take_observation():
    s=deterministic_scenario('repeated_issue',1); e=expected_for(s); assert e.cue_expected

def test_single_issue_forbids_cue():
    r=Harness().run_case(external_subjects()[0],EXERCISES[0],deterministic_scenario('single_issue',4)); assert r.expected.cue_forbidden

def test_shawn_variants_all_execute():
    h=Harness(); rs=[h.run_case(x,EXERCISES[0],deterministic_scenario('clean',9),True) for x in SHAWN_VARIANTS]; assert len(rs)==6

def test_bystander_is_ignored_with_strong_primary_lock():
    r=Harness().run_case(external_subjects()[0],EXERCISES[0],deterministic_scenario('bystander_bg',21)); assert r.passed and not r.observed.pause

def test_spotter_is_ignored_when_target_visible():
    r=Harness().run_case(external_subjects()[0],EXERCISES[2],deterministic_scenario('spotter',22)); assert r.passed and not r.observed.pause

def test_subject_swap_pauses_and_resets():
    r=Harness().run_case(external_subjects()[0],EXERCISES[0],deterministic_scenario('subject_swap',23)); assert r.passed and r.observed.pause and r.observed.reset_count==1

def test_known_mirror_region_does_not_force_pause():
    r=Harness().run_case(external_subjects()[0],EXERCISES[0],deterministic_scenario('mirror_known',24)); assert r.passed and not r.observed.pause

from harness.subject_lock import run_subject_lock_benchmark
from harness.false_cue import run_false_cue_benchmark
from harness.production import request_payload, parse_output
from harness.models import AnalyzerOutput


def test_production_adapter_wire_schema_roundtrip():
    subj=external_subjects()[0]; ex=EXERCISES[0]; sc=deterministic_scenario('clean',77)
    obs=simulate(subj,ex,sc)
    payload=request_payload(obs,ex,None)
    assert payload['schema_version']==1 and payload['observation']['scenario_id']==sc.id
    out=parse_output({'ready':True,'pause':False,'reset_count':0,'rep_count':10,'cue_count':0,'reason':'active'})
    assert out==AnalyzerOutput(True,False,0,10,0,'active')


def test_subject_lock_benchmark_exposes_strategy_tradeoffs():
    df=run_subject_lock_benchmark(120)
    swap=df[df.scenario=='subject_swap'].groupby('strategy').passed.mean()
    assert swap['motion_only'] < .10
    assert swap['multimodal_personal'] > .90


def test_full_occlusion_preserves_identity_but_pauses_biomechanics():
    df=run_subject_lock_benchmark(30)
    x=df[(df.scenario=='full_occlusion') & (df.strategy=='multimodal_personal')]
    assert (x.observed_state=='TARGET_OCCLUDED').mean() > .90
    assert (x.observed_target=='target').mean() > .90


def test_false_cue_benchmark_exercises_personalization_boundary():
    g=run_false_cue_benchmark(2500,seed=100,calibrated=False,mode='boundary')
    c=run_false_cue_benchmark(2500,seed=100,calibrated=True,mode='boundary')
    assert g.false_cue.sum() > 0
    assert c.false_cue.mean() <= g.false_cue.mean()


def test_fixture_seed_is_stable_not_python_hash_based():
    from harness.simulator import _stable_id_seed
    assert _stable_id_seed('abc') == _stable_id_seed('abc')
    assert _stable_id_seed('abc') != _stable_id_seed('abcd')


def test_jsonl_subprocess_adapter_contract(tmp_path):
    import sys
    from harness.production import JsonlSubprocessAdapter
    script=tmp_path/'fake_analyzer.py'
    script.write_text('''import json,sys\nx=json.loads(sys.stdin.readline())\nassert x["schema_version"]==1\nprint(json.dumps({"ready": True,"pause": False,"reset_count": 0,"rep_count": 10,"cue_count": 0,"reason": "active"}))\n''')
    subj=external_subjects()[0]; ex=EXERCISES[0]; sc=deterministic_scenario('clean',88); obs=simulate(subj,ex,sc)
    out=JsonlSubprocessAdapter([sys.executable,str(script)]).analyze(obs,ex,None)
    assert out.rep_count==10 and not out.pause


def test_rgb_video_runner_with_fake_pose_provider(tmp_path):
    import cv2, numpy as np
    from harness.rgb import RgbVideoRunner, PoseFrame
    path=tmp_path/'tiny.mp4'
    w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(64,64))
    for i in range(5): w.write(np.full((64,64,3),i*20,np.uint8))
    w.release()
    class Fake:
        name='fake'
        def infer(self,frame_bgr,frame_index,timestamp_ms):
            return PoseFrame(frame_index,timestamp_ms,np.zeros((33,3),np.float32),np.ones(33,np.float32))
    poses=RgbVideoRunner(Fake()).run(path)
    assert len(poses)==5 and poses[-1].frame_index==4


def test_temporal_subject_lock_background_people_do_not_pause():
    from harness.subject_lock import run_temporal_subject_lock_benchmark
    df=run_temporal_subject_lock_benchmark(60)
    x=df[df.scenario.isin(['background_bystander','spotter','foreground_cross','known_mirror'])]
    assert x.passed.all()
    assert (x.pause_frames==0).all()


def test_temporal_subject_lock_never_silently_locks_lookalike():
    from harness.subject_lock import run_temporal_subject_lock_benchmark
    df=run_temporal_subject_lock_benchmark(80)
    x=df[df.scenario=='lookalike']
    assert (x.wrong_identity_locks==0).all()
    assert (x.unsafe_locked_in_ambiguous==0).all()


def test_temporal_subject_lock_reidentifies_after_track_id_change():
    from harness.subject_lock import run_temporal_subject_lock_benchmark
    df=run_temporal_subject_lock_benchmark(50)
    x=df[df.scenario=='track_id_change']
    assert x.passed.mean() > .95
    assert (x.final_target=='target_reid').mean() > .95


def test_false_cue_summary_is_stratified_and_gated():
    from harness.false_cue import run_false_cue_benchmark
    from harness.reporting import false_cue_summary, FalseCueGate
    df=run_false_cue_benchmark(1500,seed=811,calibrated=True)
    s=false_cue_summary(df,FalseCueGate(overall_max=1,exercise_max=1,subject_max=1))
    assert s['total']==1500
    assert len(s['by_exercise'])==10
    assert len(s['by_subject'])>=18
    assert s['passed_gate']


def test_fixture_seed_stable_across_processes():
    import subprocess,sys
    code='from harness.simulator import _stable_id_seed; print(_stable_id_seed("stable-subject"))'
    cwd=str(__import__('pathlib').Path(__file__).resolve().parents[1])
    a=subprocess.check_output([sys.executable,'-c',code],cwd=cwd,text=True).strip()
    b=subprocess.check_output([sys.executable,'-c',code],cwd=cwd,text=True).strip()
    assert a==b


def test_natural_and_boundary_false_cue_suites_are_separate():
    from harness.false_cue import run_false_cue_benchmark
    n=run_false_cue_benchmark(3000,seed=991,calibrated=False,mode='natural')
    b=run_false_cue_benchmark(3000,seed=991,calibrated=False,mode='boundary')
    assert n.false_cue.mean() < b.false_cue.mean()
    assert set(n['mode'])=={'natural'} and set(b['mode'])=={'boundary'}


def test_composite_oracle_identity_ambiguity_dominates_form_issue():
    from harness.scenarios import compose_scenario
    from harness.oracle import expected_for
    s=compose_scenario(['lookalike_competitor','repeated_issue'],1234)
    e=expected_for(s)
    assert e.pause and e.reset_once and not e.cue_expected and e.reason=='target_ambiguous'


def test_composite_bystander_plus_repeated_issue_can_still_cue():
    from harness.scenarios import compose_scenario
    from harness.oracle import expected_for
    s=compose_scenario(['spotter','repeated_issue'],1235)
    e=expected_for(s)
    assert not e.pause and e.cue_expected


def test_composite_fuzz_is_reproducible():
    from harness.runner import Harness
    a=Harness().composite_fuzz(100,seed=5050)
    b=Harness().composite_fuzz(100,seed=5050)
    assert a[['case_id','passed','components']].equals(b[['case_id','passed','components']])


def test_composite_simulator_applies_wrong_view_component():
    from harness.scenarios import compose_scenario
    from harness.simulator import simulate
    from harness.profiles import external_subjects,EXERCISES
    s=compose_scenario(['wrong_view','camera_bump'],8181)
    o=simulate(external_subjects()[0],EXERCISES[0],s)
    assert o.yaw_error_deg > EXERCISES[0].yaw_tolerance_deg


def test_composite_simulator_applies_distance_component():
    from harness.scenarios import compose_scenario
    from harness.simulator import simulate
    from harness.profiles import external_subjects,EXERCISES
    s=compose_scenario(['too_close','spotter'],8182)
    o=simulate(external_subjects()[0],EXERCISES[0],s)
    assert o.frame_fill > EXERCISES[0].max_frame_fill


def test_commercial_gym_scene_pack_is_multi_factor_and_reproducible():
    from harness.gym_env import COMMERCIAL_GYM_SCENES, scenario_from_gym_scene
    assert len(COMMERCIAL_GYM_SCENES) >= 12
    assert any(len(x.components) >= 2 for x in COMMERCIAL_GYM_SCENES)
    a=scenario_from_gym_scene(COMMERCIAL_GYM_SCENES[-1],901)
    b=scenario_from_gym_scene(COMMERCIAL_GYM_SCENES[-1],901)
    assert a==b


def test_commercial_gym_external_matrix_executes():
    from harness.runner import Harness
    from harness.profiles import external_subjects
    df=Harness().commercial_gym_deterministic(external_subjects()[:2])
    assert len(df) == 2*10*24
    assert df.passed.all()


def test_false_cue_benchmark_can_be_scoped_to_external_only():
    from harness.false_cue import run_false_cue_benchmark
    from harness.profiles import external_subjects
    subs=external_subjects()[:3]
    df=run_false_cue_benchmark(400,seed=177,subjects=subs)
    assert set(df.subject).issubset({s.id for s in subs})


def test_commercial_gym_environment_parameters_affect_observation():
    from harness.gym_env import COMMERCIAL_GYM_SCENES, scenario_from_gym_scene
    from harness.simulator import simulate
    from harness.profiles import external_subjects, EXERCISES
    clean=next(x for x in COMMERCIAL_GYM_SCENES if x.id=="off_peak_clean")
    busy=next(x for x in COMMERCIAL_GYM_SCENES if x.id=="very_busy_no_occlusion")
    a=simulate(external_subjects()[0],EXERCISES[0],scenario_from_gym_scene(clean,55001))
    b=simulate(external_subjects()[0],EXERCISES[0],scenario_from_gym_scene(busy,55001))
    assert b.detected_people > a.detected_people
    assert b.tracking_quality < a.tracking_quality
    assert b.visible_required_fraction < a.visible_required_fraction


def test_commercial_gym_audio_noise_is_metadata_only():
    from harness.gym_env import COMMERCIAL_GYM_SCENES, scenario_from_gym_scene
    from harness.simulator import simulate
    from harness.profiles import external_subjects, EXERCISES
    from dataclasses import replace
    scene=COMMERCIAL_GYM_SCENES[0]
    loud=replace(scene,ambient_noise="extreme")
    s1=scenario_from_gym_scene(scene,55100); s2=scenario_from_gym_scene(loud,55100)
    assert simulate(external_subjects()[0],EXERCISES[0],s1)==simulate(external_subjects()[0],EXERCISES[0],s2)
