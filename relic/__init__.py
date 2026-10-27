"""RELIC: closed-form, exactly reversible, auditable learning after deployment.

RELIC stands for Revocable Entries for Language model Insertions and Corrections.
A frozen base provides a fixed address space; learning is an API call
on three time scales (a key-value memory, a closed-form edit, a frozen expert bank),
and every `write` / `forget` returns measured locality, reversibility and
order-invariance reports:

    from relic.api import Relic, Batch, Example

The project was called PAL-MoE until 2026-09-26; the v1 design of that name lives in
`relic.legacy`, and the `pal_moe` package keeps the old import paths working.
"""

from .api import (
    Batch,
    Relic,
    ConsolidationReport,
    EditRecord,
    Example,
    GuardConfig,
    Prediction,
)

__version__ = "0.2.0"

__all__ = [
    "Batch",
    "Relic",
    "ConsolidationReport",
    "EditRecord",
    "Example",
    "GuardConfig",
    "Prediction",
]
