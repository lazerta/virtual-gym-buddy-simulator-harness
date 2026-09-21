from __future__ import annotations

import csv

import numpy as np

from harness.mmfit_adapter import (
    MMFitLocalAdapter,
    MMFIT_CANONICAL_BRIDGES,
)


def _write_workout(root):
    w = root / "w00"
    w.mkdir()

    t = 12
    raw = np.zeros((3, t, 17), dtype=float)
    for i in range(t):
        raw[0, i, 0] = i
        raw[1, i, 0] = i
        raw[2, i, 0] = i
        for j in range(16):
            raw[:, i, j + 1] = [i * 0.01 + j, j * 0.1, 1.0 + i * 0.02]
    np.save(w / "w00_pose_3d.npy", raw)

    with (w / "w00_labels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([1, 5, 10, "squats"])
        writer.writerow([6, 10, 10, "lateral_shoulder_raises"])


def test_mmfit_adapter_reads_official_pose_layout(tmp_path):
    _write_workout(tmp_path)
    adapter = MMFitLocalAdapter(tmp_path)
    records = adapter.index()
    assert len(records) == 1

    workout = adapter.load(records[0])
    assert workout.joints_3d.shape == (12, 16, 3)
    assert workout.frame_ids.tolist() == list(range(12))
    assert workout.actions == ("lateral_shoulder_raises", "squats")
    assert workout.sets[0].rep_count == 10
    assert workout.set_slice(workout.sets[0]).shape[0] == 5


def test_mmfit_is_family_bridge_not_exact_equipment_claim(tmp_path):
    _write_workout(tmp_path)
    adapter = MMFitLocalAdapter(tmp_path)
    coverage = adapter.canonical_bridge_coverage()
    by_action = {x["mmfit_action"]: x for x in coverage}

    assert set(by_action) == set(MMFIT_CANONICAL_BRIDGES)
    assert by_action["squats"]["canonical_exercise"] == "smith_squat"
    assert by_action["squats"]["available"] is True
    assert by_action["squats"]["exact_equipment_match"] is False


def test_mmfit_summary_marks_fit3d_non_required(tmp_path):
    _write_workout(tmp_path)
    summary = MMFitLocalAdapter(tmp_path).summary(load_records=True)
    assert summary["default_public_baseline"] is True
    assert summary["fit3d_required"] is False
    assert summary["total_annotated_reps"] == 20
