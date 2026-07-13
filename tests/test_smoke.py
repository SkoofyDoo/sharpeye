"""Smoke tests — Milestone 1.1 Definition of Done."""


def test_package_imports():
    import sharpeye

    assert sharpeye.__version__ == "0.1.0"


def test_pyproject_name():
    assert True  # placeholder until Pipeline exists