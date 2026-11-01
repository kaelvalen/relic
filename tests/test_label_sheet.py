"""Cohen's kappa of the label sheet."""

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / "experiments/sandbox/live_learning/confirm"
    ),
)

from label_sheet import cohen_kappa  # noqa: E402


def test_kappa_perfect_chance_and_known_value():
    assert cohen_kappa(["a", "b"], ["a", "b"]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    # 20 yes-yes, 5 yes-no, 10 no-yes, 15 no-no: the textbook table gives 0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_kappa_of_one_shared_class_is_one():
    assert cohen_kappa(["a", "a"], ["a", "a"]) == 1.0
