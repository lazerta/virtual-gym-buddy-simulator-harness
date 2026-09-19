from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Iterable


class MovementPrimitive(str, Enum):
    """Small reusable movement vocabulary used by the harness.

    These are intentionally broader than exercise names.  Exercise variants are
    expected to share primitive topology while differing in equipment/profile
    constraints and observed joint coordination.
    """

    PRESS = "PRESS"
    PULL = "PULL"
    SQUAT = "SQUAT"
    HINGE = "HINGE"
    RAISE = "RAISE"
    CURL = "CURL"
    EXTENSION = "EXTENSION"
    CORE = "CORE"
    LOCOMOTION = "LOCOMOTION"
    TRANSITION = "TRANSITION"
    HOLD = "HOLD"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PrimitiveComposition:
    action: str
    primitives: tuple[MovementPrimitive, ...]
    source: str = "heuristic_v1"


def normalize_action_name(action: str) -> str:
    x = action.strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    return re.sub(r"_+", "_", x).strip("_")


_EXPLICIT: dict[str, tuple[MovementPrimitive, ...]] = {
    "clean_and_press": (
        MovementPrimitive.HINGE,
        MovementPrimitive.PULL,
        MovementPrimitive.TRANSITION,
        MovementPrimitive.PRESS,
    ),
    "burpee": (
        MovementPrimitive.SQUAT,
        MovementPrimitive.TRANSITION,
        MovementPrimitive.PRESS,
        MovementPrimitive.TRANSITION,
        MovementPrimitive.LOCOMOTION,
    ),
    "burpees": (
        MovementPrimitive.SQUAT,
        MovementPrimitive.TRANSITION,
        MovementPrimitive.PRESS,
        MovementPrimitive.TRANSITION,
        MovementPrimitive.LOCOMOTION,
    ),
    "jumping_jack": (MovementPrimitive.RAISE, MovementPrimitive.LOCOMOTION),
    "jumping_jacks": (MovementPrimitive.RAISE, MovementPrimitive.LOCOMOTION),
}


_TOKEN_RULES: tuple[tuple[tuple[str, ...], MovementPrimitive], ...] = (
    (("bench", "press", "pushup", "push_up", "shoulder_press", "overhead_press", "chest_press"), MovementPrimitive.PRESS),
    (("row", "pullup", "pull_up", "pulldown", "pull_down", "lat_pull", "high_row"), MovementPrimitive.PULL),
    (("squat", "lunge", "split_squat", "leg_press", "step_up"), MovementPrimitive.SQUAT),
    (("deadlift", "hinge", "good_morning", "romanian"), MovementPrimitive.HINGE),
    (("lateral_raise", "front_raise", "raise", "abduction"), MovementPrimitive.RAISE),
    (("curl", "biceps"), MovementPrimitive.CURL),
    (("triceps", "extension", "kickback", "pushdown", "push_down"), MovementPrimitive.EXTENSION),
    (("plank", "situp", "sit_up", "crunch", "core", "russian_twist"), MovementPrimitive.CORE),
    (("walk", "run", "jump", "skip"), MovementPrimitive.LOCOMOTION),
)


def infer_primitive_composition(action: str) -> PrimitiveComposition:
    name = normalize_action_name(action)
    if name in _EXPLICIT:
        return PrimitiveComposition(action=action, primitives=_EXPLICIT[name], source="explicit_v1")

    found: list[MovementPrimitive] = []
    for tokens, primitive in _TOKEN_RULES:
        if any(token in name for token in tokens):
            if primitive not in found:
                found.append(primitive)

    if not found:
        found = [MovementPrimitive.UNKNOWN]
    return PrimitiveComposition(action=action, primitives=tuple(found))


def summarize_compositions(actions: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in actions:
        comp = infer_primitive_composition(action)
        key = "+".join(p.value for p in comp.primitives)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
