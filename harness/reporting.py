from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class FalseCueGate:
    overall_max: float = 0.02
    exercise_max: float = 0.04
    subject_max: float = 0.05


def false_cue_summary(df: pd.DataFrame, gate: FalseCueGate | None = None) -> dict:
    gate = gate or FalseCueGate()
    overall=float(df.false_cue.mean()) if len(df) else 0.0
    by_ex=df.groupby("exercise").false_cue.mean().sort_values(ascending=False)
    by_subj=df.groupby("subject").false_cue.mean().sort_values(ascending=False)
    summary={
        "total":int(len(df)),
        "false_cues":int(df.false_cue.sum()),
        "overall_rate":overall,
        "worst_exercise_rate":float(by_ex.iloc[0]) if len(by_ex) else 0.0,
        "worst_exercise":str(by_ex.index[0]) if len(by_ex) else None,
        "worst_subject_rate":float(by_subj.iloc[0]) if len(by_subj) else 0.0,
        "worst_subject":str(by_subj.index[0]) if len(by_subj) else None,
        "by_exercise":{str(k):float(v) for k,v in by_ex.items()},
        "by_subject":{str(k):float(v) for k,v in by_subj.items()},
        "gate":asdict(gate),
    }
    summary["passed_gate"]=(
        summary["overall_rate"] <= gate.overall_max
        and summary["worst_exercise_rate"] <= gate.exercise_max
        and summary["worst_subject_rate"] <= gate.subject_max
    )
    return summary


def write_json_report(data: dict, path: str | Path) -> None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2,sort_keys=True),encoding="utf-8")
