"""SS125: adapter alone against note alone, paired, on saved outputs (no model run).

.venv/bin/python experiments/sandbox/live_learning/confirm/adapter_vs_note.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from locality_eval import paired_diff_ci  # noqa: E402

from relic.metrics import word_hit  # noqa: E402

CONFIRM = Path("results/live_learning/confirm")
STORES = {"r16": "{}_n1000_v8_eval.json", "r2": "{}_n1000_r2_v8_eval.json"}


def main() -> None:
    out = CONFIRM / "adapter_vs_note.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    report = {"prereg": "SS125", "score": "word boundary", "cells": {}}
    for rank, pattern in STORES.items():
        for tag in ("cf", "zsre"):
            saved = json.loads((CONFIRM / pattern.format(tag)).read_text())
            rows = saved["examples"]["serving"]
            for kind in ("probe", "paraphrase"):
                scored = {
                    arm: [
                        word_hit(r[kind]["responses"][arm], r["answer"]) for r in rows
                    ]
                    for arm in ("adapter", "retrieval", "both")
                }
                rates = {a: round(sum(v) / len(v), 4) for a, v in scored.items()}
                report["cells"][f"{rank}/{tag}/{kind}"] = {
                    **rates,
                    "adapter_minus_note": paired_diff_ci(
                        scored["adapter"], scored["retrieval"]
                    ),
                    "adapter_minus_both": paired_diff_ci(
                        scored["adapter"], scored["both"]
                    ),
                }
    report["rule_holds"] = all(
        c["adapter_minus_note"]["ci95"][0] > 0
        for name, c in report["cells"].items()
        if name.startswith("r2/")
    )
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
