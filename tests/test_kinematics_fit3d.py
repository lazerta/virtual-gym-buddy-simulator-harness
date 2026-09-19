def test_rep_generation_has_seeded_natural_variation_but_is_reproducible():
    from harness.rep_detection import generate_rep_case
    a=generate_rep_case('clean_5',seed=44001)
    b=generate_rep_case('clean_5',seed=44001)
    c=generate_rep_case('clean_5',seed=44002)
    assert a==b
    assert a!=c
    from harness.rep_detection import detect_reps
    assert len(detect_reps(a))==5
    assert len(detect_reps(c))==5


def test_exercise_kinematic_models_are_reproducible_and_not_truth_depth_passthrough():
    from harness.exercise_kinematics import MODELS
    from harness.rep_detection import generate_rep_case
    from harness.profiles import external_subjects, EXERCISES
    subj=external_subjects()[0]; ex=next(x for x in EXERCISES if x.id=='incline_db_press')
    base=generate_rep_case('clean_5',seed=12345,subject=subj,exercise=ex,fps=20)
    model=MODELS['incline_db_press']; v=model.variants[1]
    a=model.render(base,subject=subj,variant=v,seed=321)
    b=model.render(base,subject=subj,variant=v,seed=321)
    assert a==b
    assert any(abs(x.observed_depth-x.truth_depth)>1e-5 for x in a if x.valid)


def test_smith_kinematic_variants_change_joint_coordination_but_preserve_rep_semantics():
    from harness.exercise_kinematics import MODELS, as_rep_frames
    from harness.rep_detection import generate_rep_case, detect_reps
    from harness.profiles import external_subjects, EXERCISES
    subj=external_subjects()[0]; ex=next(x for x in EXERCISES if x.id=='smith_squat')
    base=generate_rep_case('clean_5',seed=22345,subject=subj,exercise=ex,fps=20)
    m=MODELS['smith_squat']
    a=m.render(base,subject=subj,variant=m.variants[0],seed=10)
    b=m.render(base,subject=subj,variant=m.variants[-1],seed=10)
    assert a[-20].channels['guide_slope_deg'] != b[-20].channels['guide_slope_deg']
    assert a[-20].channels['foot_offset_height_ratio'] != b[-20].channels['foot_offset_height_ratio']
    assert len(detect_reps(as_rep_frames(a)))==5
    assert len(detect_reps(as_rep_frames(b)))==5


def test_lateral_raise_model_has_variable_scapulohumeral_rhythm_and_supported_variants():
    from harness.exercise_kinematics import MODELS
    ids={x.id for x in MODELS['lateral_raise'].variants}
    assert {'neutral_90','external_90','internal_90','scaption_90','flexed_elbow_90'} <= ids
    ratios=[]
    from harness.rep_detection import generate_rep_case
    from harness.profiles import external_subjects, EXERCISES
    subj=external_subjects()[0]; ex=next(x for x in EXERCISES if x.id=='lateral_raise')
    base=generate_rep_case('clean_5',seed=333,subject=subj,exercise=ex,fps=20)
    for seed in range(10):
        x=MODELS['lateral_raise'].render(base,subject=subj,variant=MODELS['lateral_raise'].variants[0],seed=seed)
        ratios.append(x[len(x)//3].channels['scapulohumeral_ratio'])
    assert max(ratios)-min(ratios) > .2
    assert all(1.1 <= r <= 2.7 for r in ratios)


def test_kinematic_model_benchmark_external_and_personal_smoke():
    from harness.exercise_kinematics import run_kinematic_model_benchmark
    from harness.profiles import external_subjects, PERSONAL_REFERENCE
    a=run_kinematic_model_benchmark([external_subjects()[6]],fps=20)
    b=run_kinematic_model_benchmark([PERSONAL_REFERENCE],fps=20)
    assert a.passed.all(), a.loc[~a.passed].head().to_dict('records')
    assert b.passed.all(), b.loc[~b.passed].head().to_dict('records')


def _write_mock_fit3d(tmp_path):
    import json, numpy as np
    root=tmp_path/'fit3d'
    subj=root/'train'/'s03'
    (subj/'joints3d_25').mkdir(parents=True)
    (subj/'camera_parameters'/'50591643').mkdir(parents=True)
    base=np.arange(75,dtype=float).reshape(1,25,3)/100.0
    signal=np.zeros(240,dtype=float)
    for a,b in [(10,69),(80,139),(150,209)]:
        mid=(a+b)//2
        signal[a:mid+1]=np.linspace(0,1,mid-a+1)
        signal[mid:b+1]=np.linspace(1,0,b-mid+1)
    delta=np.linspace(.02,.18,75,dtype=float).reshape(1,25,3)
    joints=base + signal[:,None,None]*delta
    for action in ['dumbbell_shoulder_press','clean_and_press']:
        (subj/'joints3d_25'/f'{action}.json').write_text(json.dumps({'joints3d_25':joints.tolist()}))
        (subj/'camera_parameters'/'50591643'/f'{action}.json').write_text(json.dumps({'dummy':True}))
    (subj/'rep_ann.json').write_text(json.dumps({
        'dumbbell_shoulder_press': [[10,69],[80,139],[150,209]],
        'clean_and_press': {'rep1':[12,74],'rep2':[82,146],'rep3':[154,222]},
    }))
    return root


def test_movement_primitives_share_structure_across_variants():
    from harness.movement_primitives import infer_primitive_composition, MovementPrimitive
    a=infer_primitive_composition('incline dumbbell bench press')
    b=infer_primitive_composition('machine shoulder press')
    c=infer_primitive_composition('clean and press')
    assert a.primitives==(MovementPrimitive.PRESS,)
    assert b.primitives==(MovementPrimitive.PRESS,)
    assert c.primitives[-1]==MovementPrimitive.PRESS
    assert MovementPrimitive.HINGE in c.primitives and MovementPrimitive.PULL in c.primitives


def test_fit3d_adapter_discovers_and_loads_local_motion(tmp_path):
    from harness.fit3d_adapter import Fit3DLocalAdapter
    root=_write_mock_fit3d(tmp_path)
    a=Fit3DLocalAdapter(root)
    records=a.index()
    assert len(records)==2
    rec=next(x for x in records if x.action=='dumbbell_shoulder_press')
    seq=a.load(rec)
    assert seq.joints_3d.shape==(240,25,3)
    assert len(seq.rep_intervals)==3
    assert [p.value for p in seq.primitives]==['PRESS']
    assert seq.camera_names==('50591643',)


def test_fit3d_rep_annotations_are_oracle_only_and_support_common_shapes(tmp_path):
    from harness.fit3d_adapter import Fit3DLocalAdapter
    root=_write_mock_fit3d(tmp_path)
    a=Fit3DLocalAdapter(root)
    rec=next(x for x in a.index() if x.action=='clean_and_press')
    seq=a.load(rec)
    assert len(seq.rep_intervals)==3
    assert seq.oracle_rep_index(20)==0
    assert seq.oracle_rep_index(75) is None
    assert seq.oracle_phase_t(12)==0.0
    assert seq.oracle_phase_t(74)==1.0
    assert seq.rep_slice(0).shape[0]==63


def test_fit3d_normalization_is_translation_and_scale_robust(tmp_path):
    from harness.fit3d_adapter import Fit3DLocalAdapter
    root=_write_mock_fit3d(tmp_path)
    seq=Fit3DLocalAdapter(root).load(Fit3DLocalAdapter(root).index()[0])
    x=seq.centered_scaled_joints()
    assert x.shape==seq.joints_3d.shape
    assert abs(float(x.mean(axis=1).max())) < 1e-9
    assert __import__('numpy').isfinite(x).all()


def test_fit3d_adapter_summary_contains_primitive_map(tmp_path):
    from harness.fit3d_adapter import Fit3DLocalAdapter
    root=_write_mock_fit3d(tmp_path)
    s=Fit3DLocalAdapter(root).summary(load_records=True)
    assert s['record_count']==2
    assert s['subject_count']==1
    assert s['action_count']==2
    assert s['total_reps']==6
    assert s['primitive_compositions']['dumbbell_shoulder_press']==['PRESS']
    assert s['primitive_compositions']['clean_and_press']==['HINGE','PULL','TRANSITION','PRESS']


def test_fit3d_interval_parser_rejects_overlap():
    import pytest
    from harness.fit3d_adapter import parse_rep_intervals, Fit3DFormatError
    with pytest.raises(Fit3DFormatError):
        parse_rep_intervals([[0,20],[20,40]],100)


def test_fit3d_principal_motion_replay_uses_motion_not_rep_annotations(tmp_path):
    from harness.fit3d_adapter import Fit3DLocalAdapter, principal_motion_depth, evaluate_rep_replay
    root=_write_mock_fit3d(tmp_path)
    adapter=Fit3DLocalAdapter(root)
    seq=adapter.load(next(x for x in adapter.index() if x.action=='dumbbell_shoulder_press'))
    depth=principal_motion_depth(seq,smooth_frames=1)
    assert len(depth)==seq.frame_count
    assert depth.max() > .85 and depth.min() < .15
    result=evaluate_rep_replay(seq)
    assert result['status']=='RUN'
    assert result['expected_reps']==3
    assert result['detected_reps']==3
    assert result['passed']


def test_fit3d_compound_motion_is_composed_and_not_forced_through_single_cycle_detector(tmp_path):
    from harness.fit3d_adapter import Fit3DLocalAdapter, evaluate_rep_replay
    root=_write_mock_fit3d(tmp_path)
    adapter=Fit3DLocalAdapter(root)
    seq=adapter.load(next(x for x in adapter.index() if x.action=='clean_and_press'))
    result=evaluate_rep_replay(seq)
    assert result['status']=='SKIPPED_COMPOUND_OR_UNKNOWN'
    assert result['passed'] is None
