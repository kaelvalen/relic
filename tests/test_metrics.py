"""The answer scores."""

import pytest

pytest.importorskip("torch")

from relic.metrics import hit, word_hit  # noqa: E402


@pytest.mark.parametrize(
    "resp,token,sub,word",
    [
        ("She is female.", "male", True, False),
        ("Romanian", "Romania", True, False),
        ("In Paris.", "paris", True, True),
        ("written in C.", "C", True, True),
        ("Coding", "C", True, False),
        ("F_X", "X", True, True),
        ("C++ it is", "C++", True, True),
        ("nothing here", "Oslo", False, False),
    ],
)
def test_word_hit_is_stricter_than_hit(resp, token, sub, word):
    assert hit(resp, token) is sub
    assert word_hit(resp, token) is word
