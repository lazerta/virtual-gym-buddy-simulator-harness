def test_commercial_gym_episode_templates_are_30_to_90_seconds_and_multievent():
    from harness.gym_episode import COMMERCIAL_GYM_EPISODES
    assert len(COMMERCIAL_GYM_EPISODES) >= 8
    assert all(30 <= e.duration_s <= 90 for e in COMMERCIAL_GYM_EPISODES)
    assert any(len(e.events) >= 6 for e in COMMERCIAL_GYM_EPISODES)
    assert any(e.crowd_level == 'high' for e in COMMERCIAL_GYM_EPISODES)


def test_commercial_gym_episode_generation_is_reproducible():
    from harness.gym_episode import COMMERCIAL_GYM_EPISODES, generate_episode_frames
    from harness.profiles import external_subjects, EXERCISES
    a=generate_episode_frames(external_subjects()[0],EXERCISES[0],COMMERCIAL_GYM_EPISODES[-1],4401)
    b=generate_episode_frames(external_subjects()[0],EXERCISES[0],COMMERCIAL_GYM_EPISODES[-1],4401)
    assert a==b


def test_episode_background_people_and_spotter_do_not_create_wrong_lock_or_false_rep():
    from harness.gym_episode import COMMERCIAL_GYM_EPISODES, run_episode
    from harness.profiles import external_subjects, EXERCISES
    ep=next(x for x in COMMERCIAL_GYM_EPISODES if x.id=='rush_hour_bench_spotter')
    r=run_episode(external_subjects()[0],EXERCISES[0],ep,4410)
    assert r.wrong_identity_locks==0
    assert r.false_reps==0
    assert r.unsafe_biomechanics_frames==0


def test_episode_lookalike_never_silently_becomes_primary_subject():
    from harness.gym_episode import COMMERCIAL_GYM_EPISODES, run_episode
    from harness.profiles import external_subjects, EXERCISES
    ep=next(x for x in COMMERCIAL_GYM_EPISODES if x.id=='lookalike_overlap')
    r=run_episode(external_subjects()[1],EXERCISES[1],ep,4420)
    assert r.wrong_identity_locks==0
    assert r.unsafe_biomechanics_frames==0


def test_episode_reidentify_after_occlusion_is_safe():
    from harness.gym_episode import COMMERCIAL_GYM_EPISODES, run_episode
    from harness.profiles import external_subjects, EXERCISES
    ep=next(x for x in COMMERCIAL_GYM_EPISODES if x.id=='occlusion_reidentify')
    r=run_episode(external_subjects()[2],EXERCISES[2],ep,4430)
    assert r.wrong_identity_locks==0
    assert r.false_reps==0
    assert r.max_reacquire_s <= 1.0


def test_episode_external_smoke_matrix_passes():
    from harness.gym_episode import run_commercial_gym_episode_benchmark
    from harness.profiles import external_subjects, EXERCISES
    df=run_commercial_gym_episode_benchmark(external_subjects()[:2],exercises=EXERCISES[:2])
    from harness.gym_episode import COMMERCIAL_GYM_EPISODES
    assert len(df)==2*2*len(COMMERCIAL_GYM_EPISODES)
    assert df.passed.all(), df.loc[~df.passed].to_dict('records')[:5]


def test_temporal_rep_detector_core_cases():
    from harness.rep_detection import REP_CASES, generate_rep_case, detect_reps
    from harness.profiles import external_subjects, EXERCISES
    subj=external_subjects()[0]; ex=EXERCISES[0]
    for truth in REP_CASES:
        ev=detect_reps(generate_rep_case(truth.case_id,seed=90210,subject=subj,exercise=ex))
        assert len(ev)==truth.expected_total, (truth.case_id,len(ev),truth.expected_total)
        assert sum(x.classification=='ASSISTED' for x in ev)==truth.expected_assisted
        assert sum(x.classification=='UNCERTAIN' for x in ev)==truth.expected_uncertain


def test_rep_detector_never_stitches_across_tracking_gap():
    from harness.rep_detection import generate_rep_case, detect_reps
    ev=detect_reps(generate_rep_case('tracking_gap_mid_rep_then_clean',seed=90211))
    assert len(ev)==1


def test_rep_detection_external_profile_matrix_smoke():
    from harness.rep_detection import run_rep_detection_benchmark
    from harness.profiles import external_subjects, EXERCISES
    df=run_rep_detection_benchmark(external_subjects()[:2],exercises=EXERCISES[:2])
    assert len(df)==2*2*14
    assert df.passed.all(), df.loc[~df.passed].to_dict('records')[:5]


def test_trainer_demonstration_does_not_create_false_reps():
    from harness.gym_episode import COMMERCIAL_GYM_EPISODES, run_episode
    from harness.profiles import external_subjects, EXERCISES
    ep=next(x for x in COMMERCIAL_GYM_EPISODES if x.id=='trainer_demonstrates_same_exercise')
    r=run_episode(external_subjects()[0],EXERCISES[0],ep,90300,fps=5)
    assert r.passed
    assert r.false_reps==0 and r.wrong_identity_locks==0
    assert r.counted_reps==r.expected_counted_reps


def test_trainer_assisted_finish_is_classified():
    from harness.gym_episode import COMMERCIAL_GYM_EPISODES, run_episode
    from harness.profiles import external_subjects, EXERCISES
    ep=next(x for x in COMMERCIAL_GYM_EPISODES if x.id=='trainer_assisted_finish')
    r=run_episode(external_subjects()[0],EXERCISES[0],ep,90301,fps=5)
    assert r.passed
    assert r.expected_assisted_reps>=1 and r.assisted_reps==r.expected_assisted_reps
    assert r.expected_uncertain_reps>=1 and r.uncertain_reps==r.expected_uncertain_reps


def test_trainer_edge_rep_set_rejects_partial_and_failed_attempts():
    from harness.gym_episode import COMMERCIAL_GYM_EPISODES, run_episode
    from harness.profiles import external_subjects, EXERCISES
    ep=next(x for x in COMMERCIAL_GYM_EPISODES if x.id=='trainer_edge_rep_set')
    r=run_episode(external_subjects()[0],EXERCISES[0],ep,90302,fps=5)
    assert r.passed
    assert r.total_target_attempts==10
    assert r.expected_counted_reps==8
    assert r.counted_reps==8


def test_reality_tolerance_keeps_safety_discrete_but_allows_noisy_class_boundaries():
    from harness.tolerance import IntegerRange, NumericRange, allowed_assistance_classes
    assert IntegerRange.exact(0).contains(0)
    assert not IntegerRange.exact(0).contains(1)
    assert NumericRange.around(1.0,.1).contains(1.08)
    assert allowed_assistance_classes(.28)==frozenset({'NORMAL','ASSISTED'})
    assert allowed_assistance_classes(.78)==frozenset({'ASSISTED','UNCERTAIN'})
