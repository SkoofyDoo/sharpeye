"""Milestone 2.1 — gate engine tests."""

from pathlib import Path

from sharpeye.config import GateGroup, GateRule, load_preset
from sharpeye.gates.engine import evaluate_gates, gates_passed
from sharpeye.gates.rules import check_rule

PRESETS_DIR = Path(__file__).resolve().parents[1] / "presets"


def test_brightness_too_dark():
    rule = GateRule(
        metric="brightness_mean",
        min=20,
        max=245,
        severity="catastrophic",
        message="Image is too dark or too bright.",
        suggestion="Move closer to a window.",
    )
    issue = check_rule(rule, {"brightness_mean": 10.0})
    assert issue is not None
    assert issue.code == "brightness_mean_too_low"
    assert issue.severity == "catastrophic"
    assert "window" in issue.suggestion.lower()


def test_brightness_passes():
    rule = GateRule(metric="brightness_mean", min=20, max=245, severity="catastrophic")
    assert check_rule(rule, {"brightness_mean": 128.0}) is None


def test_blur_fails_laplacian():
    rule = GateRule(
        metric="laplacian_variance",
        min=30,
        severity="catastrophic",
        message="Image appears blurry.",
        suggestion="Hold the camera steady.",
    )
    issue = check_rule(rule, {"laplacian_variance": 5.0})
    assert issue is not None
    assert issue.metric == "laplacian_variance"


def test_evaluate_gates_from_default_preset():
    preset = load_preset("default", presets_dir=PRESETS_DIR)
    issues = evaluate_gates(
        {"brightness_mean": 10.0, "laplacian_variance": 5.0},
        preset.gates.rules,
        preset.gates.groups,
    )
    assert len(issues) >= 2
    assert not gates_passed(issues)


def test_gates_passed_no_catastrophic():
    issues = evaluate_gates(
        {"brightness_mean": 128.0, "laplacian_variance": 100.0},
        [
            GateRule(metric="brightness_mean", min=20, severity="catastrophic"),
            GateRule(metric="laplacian_variance", min=30, severity="catastrophic"),
        ],
    )
    assert issues == []
    assert gates_passed(issues)


def test_any_of_group_one_failure():
    group = GateGroup(
        mode="any_of",
        rules=[
            GateRule(metric="brightness_mean", min=20, severity="soft", message="dark"),
            GateRule(metric="laplacian_variance", min=30, severity="soft", message="blur"),
        ],
    )
    issues = evaluate_gates({"brightness_mean": 10.0, "laplacian_variance": 100.0}, [], [group])
    assert len(issues) == 1
    assert issues[0].message == "dark"


def test_all_of_group_partial_fail():
    group = GateGroup(
        mode="all_of",
        rules=[
            GateRule(metric="brightness_mean", min=20, severity="soft", message="dark"),
            GateRule(metric="contrast_std", min=8, severity="soft", message="low contrast"),
        ],
    )
    issues = evaluate_gates({"brightness_mean": 10.0, "contrast_std": 20.0}, [], [group])
    assert issues == []


def test_all_of_group_both_fail():
    group = GateGroup(
        mode="all_of",
        rules=[
            GateRule(metric="brightness_mean", min=20, severity="soft", message="dark"),
            GateRule(metric="contrast_std", min=8, severity="soft", message="low contrast"),
        ],
    )
    issues = evaluate_gates({"brightness_mean": 10.0, "contrast_std": 3.0}, [], [group])
    assert len(issues) == 2