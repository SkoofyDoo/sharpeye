"""Single gate rule evaluation."""

from __future__ import annotations

from sharpeye.config import GateRule
from sharpeye.report import Issue


def _failure_kind(rule: GateRule, value: float) -> str | None:
    if rule.min is not None and value < rule.min:
        return "below_min"
    if rule.max is not None and value > rule.max:  
        return "above_max"
    return None

def issue_code(rule: GateRule, failure: str) -> str:
    if failure == "below_min":
        return f"{rule.metric}_too_low"
    if failure == "above_max":
        return f"{rule.metric}_too_high"
    return f"{rule.metric}_failed"

def check_rule(rule: GateRule, metrics: dict[str, float]) -> Issue | None:
    if rule.metric not in metrics:
        return Issue(
            code = "metric_missing",
            severity = "soft",
            message = f"Metric '{rule.metric}' was not computed.",
            suggestion = "Enable this metric in preset enabled_metrics.",
            metric = rule.metric,
            value = None,
        )

    value = metrics[rule.metric]
    failure = _failure_kind(rule, value)
    if failure is None:
        return None
    
    return Issue(
        code = issue_code(rule, failure),
        severity = rule.severity,
        message = rule.message or f"{rule.metric} out of range",
        suggestion = rule.suggestion,
        metric = rule.metric,
        value = value,
    )
