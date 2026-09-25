from __future__ import annotations

import json

from harness.adapter import ReferenceProductionAdapter
from harness.profiles import (
    EXERCISES,
    PERSONAL_VARIANTS,
    SYNTHETIC_PERSONAL_PROFILE,
    external_subjects,
    external_synthetic_profiles,
    load_local_synthetic_subject_profile,
)
from harness.runner import Harness
from harness.scenarios import deterministic_scenario
from harness.simulator import simulate

REP_FORM_FAMILIES = (
    "clean",
    "single_issue",
    "repeated_issue",
    "short_rom",
    "asymmetry",
)


def test_synthetic_subject_contract_covers_external_and_visual_variants():
    external = external_synthetic_profiles()
    assert len(external) == 18
    assert len({x.synthetic_subject_profile_id for x in external}) == 18
    assert all(x.version >= 1 for x in external)
    assert all(x.source_confidence > 0 for x in external)
    assert all(len(x.normalized_skeleton_proportions.__dict__) == 8 for x in external)

    personal = SYNTHETIC_PERSONAL_PROFILE
    assert len(personal.variants) == 6
    assert {"neutral", "loose", "fitted"}.issubset(
        set(personal.clothing_variants)
    )
    assert "generic_accessory" in personal.accessory_variants
    assert "appearance_stress" in personal.rendering_variants


def test_private_local_profile_is_normalized_and_versioned(tmp_path):
    path = tmp_path / "private-profile.json"
    path.write_text(
        json.dumps(
            {
                "id": "private-local",
                "version": 4,
                "source_confidence": 0.72,
                "source_notes": ["normalized local approximation"],
                "stature_scale": 1.03,
                "torso_ratio": 0.312,
                "upper_arm_ratio": 0.188,
                "forearm_ratio": 0.147,
                "thigh_ratio": 0.247,
                "shin_ratio": 0.248,
                "shoulder_width_ratio": 0.246,
                "hip_width_ratio": 0.186,
                "silhouette_scale": 1.01,
                "visual_variant": "current_like",
            }
        ),
        encoding="utf-8",
    )
    profile = load_local_synthetic_subject_profile(path)
    assert profile is not None
    assert profile.synthetic_subject_profile_id == "private-local"
    assert profile.version == 4
    assert profile.source == "local-private-profile"
    assert profile.source_confidence == 0.72
    assert profile.normalized_skeleton_proportions.body_scale_factor == 1.03
    assert len(profile.resolve_all()) == 6


def test_visual_only_variants_preserve_underlying_movement_evidence():
    reference = PERSONAL_VARIANTS[0]
    assert reference.simulation_identity is not None

    for exercise in EXERCISES:
        for family in REP_FORM_FAMILIES:
            scenario = deterministic_scenario(family, seed=29000 + EXERCISES.index(exercise))
            expected = simulate(reference, exercise, scenario)
            for subject in PERSONAL_VARIANTS[1:]:
                observed = simulate(subject, exercise, scenario)
                assert subject.simulation_identity == reference.simulation_identity
                assert observed.rep_candidates == expected.rep_candidates
                assert observed.issue_evidence_by_rep == expected.issue_evidence_by_rep


def test_visual_only_variants_keep_rep_and_form_conclusions_identical():
    adapter = ReferenceProductionAdapter()
    reference = PERSONAL_VARIANTS[0]

    for exercise in EXERCISES:
        for family in REP_FORM_FAMILIES:
            scenario = deterministic_scenario(
                family,
                seed=39000 + EXERCISES.index(exercise),
            )
            baseline = adapter.analyze(
                simulate(reference, exercise, scenario),
                exercise,
                None,
            )
            baseline_semantics = (
                baseline.rep_count,
                baseline.cue_count,
                baseline.pause,
                baseline.reason,
            )
            for subject in PERSONAL_VARIANTS[1:]:
                result = adapter.analyze(
                    simulate(subject, exercise, scenario),
                    exercise,
                    None,
                )
                assert (
                    result.rep_count,
                    result.cue_count,
                    result.pause,
                    result.reason,
                ) == baseline_semantics


def test_external_normalized_morphology_preserves_rep_form_oracle_contract():
    harness = Harness()
    subjects = external_subjects()
    failures = []
    seed = 49000

    for subject in subjects:
        for exercise in EXERCISES:
            for family in REP_FORM_FAMILIES:
                result = harness.run_case(
                    subject,
                    exercise,
                    deterministic_scenario(family, seed),
                )
                if not result.passed:
                    failures.append(result.case_id)
                seed += 1

    assert not failures, failures[:20]
