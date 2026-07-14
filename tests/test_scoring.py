"""Milestone 3.1 — batch scoring tests."""

from sharpeye.scoring.classify import classify_by_percentile
from sharpeye.scoring.composite import composite_score
from sharpeye.scoring.normalize import normalize_metrics_batch


def test_normalize_higher_is_better():
    metrics_list = [
        {"laplacian_variance": 10.0},
        {"laplacian_variance": 50.0},
        {"laplacian_variance": 30.0},
    ]
    result = normalize_metrics_batch(metrics_list, ["laplacian_variance"])
    assert result[0]["laplacian_variance"] == 0.0
    assert result[1]["laplacian_variance"] == 1.0
    assert result[2]["laplacian_variance"] == 0.5


def test_normalize_lower_is_better_noise():
    metrics_list = [
        {"noise_std": 5.0},
        {"noise_std": 1.0},
        {"noise_std": 3.0},
    ]
    result = normalize_metrics_batch(metrics_list, ["noise_std"])
    assert result[0]["noise_std"] == 0.0
    assert result[1]["noise_std"] == 1.0
    assert result[2]["noise_std"] == 0.5


def test_normalize_all_equal_returns_one():
    metrics_list = [{"contrast_std": 20.0}, {"contrast_std": 20.0}]
    result = normalize_metrics_batch(metrics_list, ["contrast_std"])
    assert result[0]["contrast_std"] == 1.0
    assert result[1]["contrast_std"] == 1.0


def test_normalize_multiple_metrics():
    metrics_list = [
        {"laplacian_variance": 10.0, "noise_std": 5.0},
        {"laplacian_variance": 30.0, "noise_std": 1.0},
    ]
    result = normalize_metrics_batch(metrics_list, ["laplacian_variance", "noise_std"])
    assert len(result) == 2
    assert "laplacian_variance" in result[0]
    assert "noise_std" in result[0]


def test_composite_score_weighted_sum():
    normalized = {
        "laplacian_variance": 1.0,
        "tenengrad": 0.5,
        "contrast_std": 0.0,
        "noise_std": 1.0,
    }
    weights = {
        "laplacian_variance": 0.40,
        "tenengrad": 0.30,
        "contrast_std": 0.15,
        "noise_std": 0.15,
    }
    score = composite_score(normalized, weights)
    assert score == 0.40 * 1.0 + 0.30 * 0.5 + 0.15 * 0.0 + 0.15 * 1.0


def test_classify_by_percentile_three_tiers():
    scores = [0.9, 0.5, 0.1]
    labels = classify_by_percentile(scores, good_percentile = 0.67, medium_percentile = 0.33)
    assert len(labels) == 3
    assert labels[0] == "good"
    assert labels.count("medium") >= 1
    assert labels[-1] == "bad"


def test_classify_empty_returns_empty():
    assert classify_by_percentile([], 0.67, 0.33) == []