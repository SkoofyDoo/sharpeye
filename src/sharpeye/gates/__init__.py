"""Gate engine — rule-based quality checks."""

from sharpeye.gates.engine import evaluate_gates, gates_passed
from sharpeye.gates.rules import check_rule

__all__ = ["check_rule", "evaluate_gates", "gates_passed"]