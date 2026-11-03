"""Readings shared by the store, the serving path and the evaluations."""

from __future__ import annotations

import re

import numpy as np
import torch


def hit(resp, token):
    return token.lower() in resp.lower()


def word_hit(resp, token):
    """`hit` with the target between non-alphanumeric characters ("male" not in "female")."""
    pattern = rf"(?<![^\W_]){re.escape(token)}(?![^\W_])"
    return re.search(pattern, resp, flags=re.IGNORECASE) is not None


def ci(values, n=2000, seed=0):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n, arr.size), replace=True).mean(axis=1)
    return [
        round(float(np.percentile(boots, 2.5)), 4),
        round(float(np.percentile(boots, 97.5)), 4),
    ]


def sym_kl(x: torch.Tensor, y: torch.Tensor) -> float:
    """Symmetric KL between the next-token distributions of two logit vectors.

    Computed from log-probabilities: a probability that underflows to 0 would make the
    ratio form NaN, and a NaN compares False against the canary limit.
    """
    lx, ly = torch.log_softmax(x, -1), torch.log_softmax(y, -1)
    return float((lx.exp() * (lx - ly)).sum() + (ly.exp() * (ly - lx)).sum())
