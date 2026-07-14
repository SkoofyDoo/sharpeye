"""Gate engine — evaluate metrics against preset rules."""

from __future__ import annotations

from sharpeye.config import GateGroup, GateRule
from sharpeye.gates.rules import check_rule
from sharpeye.report import Issue


def evaluate_gates(
    metrics: dict[str, float],
    rules: list[GateRule],
    groups: list[GateGroup] | None = None,
) -> list[Issue]:
    issues: list[Issue] = []

    for rule in rules: 
        issue = check_rule(rule, metrics)
        if issue is not None:
            issues.append(issue)
    for group in groups or []:
        issues.extend(_evaluate_group(group, metrics))

    return issues

def _evaluate_group(group: GateGroup, metrics: dict[str, float]) -> list[Issue]:
    failed: list[Issue] = []
    for rule in group.rules:
        issue = check_rule(rule, metrics)
        if issue is not None:
            failed.append(issue)

    if not failed:
        return []
    
    if group.mode == "any_of":
        return failed
    
    if len(failed) == len(group.rules):
        return failed
    return []

def gates_passed(issues: list[Issue]) -> bool:
    return not any(i.severity == "catastrophic" for i in issues)