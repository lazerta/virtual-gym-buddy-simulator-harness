from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NumericRange:
    low: float
    high: float
    def contains(self, value: float) -> bool: return self.low <= value <= self.high
    @classmethod
    def around(cls, center: float, minus: float, plus: float | None = None) -> "NumericRange":
        if plus is None: plus = minus
        return cls(center - minus, center + plus)


@dataclass(frozen=True)
class IntegerRange:
    low: int
    high: int
    def contains(self, value: int) -> bool: return self.low <= value <= self.high
    @classmethod
    def exact(cls, value: int) -> "IntegerRange": return cls(value, value)


@dataclass(frozen=True)
class RealityTolerancePolicy:
    rep_completion_early_s: float = 0.35
    rep_completion_early_fraction_of_concentric: float = 0.30
    rep_completion_late_s: float = 0.75
    reacquire_max_s: float = 0.80
    post_interrupt_pause_grace_s: float = 0.80
    assistance_transition_margin: float = 0.06
    tracking_threshold_margin: float = 0.04


DEFAULT_REALITY_TOLERANCE = RealityTolerancePolicy()


def allowed_assistance_classes(evidence: float, *, assisted_threshold: float = 0.28, uncertain_threshold: float = 0.78, margin: float = DEFAULT_REALITY_TOLERANCE.assistance_transition_margin) -> frozenset[str]:
    x = max(0.0, min(1.0, float(evidence)))
    if x < assisted_threshold - margin: return frozenset({"NORMAL"})
    if x <= assisted_threshold + margin: return frozenset({"NORMAL", "ASSISTED"})
    if x < uncertain_threshold - margin: return frozenset({"ASSISTED"})
    if x <= uncertain_threshold + margin: return frozenset({"ASSISTED", "UNCERTAIN"})
    return frozenset({"UNCERTAIN"})
