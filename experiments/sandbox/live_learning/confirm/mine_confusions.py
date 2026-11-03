"""SS110: which CounterFact relations the relation encoder confuses (templates only).

.venv/bin/python experiments/sandbox/live_learning/confirm/mine_confusions.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from router_v4_calibrate import _templates_by_relation  # noqa: E402

from relic.router import make_scorer  # noqa: E402

THETA = 0.7501
MIN_SHARE = 0.05
TOP = 3
CONFIRM = Path("results/live_learning/confirm")


def main() -> None:
    out = CONFIRM / "relation_confusions.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    raw = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    keys = {
        r: sorted({t.replace("{}", "X") for t in ts})
        for r, ts in _templates_by_relation(raw).items()
    }
    scorer = make_scorer("relenc", device="cpu")
    scorer.prefetch([k for ks in keys.values() for k in ks])
    matrix, identical = {}, {}
    for r1, r2 in itertools.combinations(sorted(keys), 2):
        same = set(keys[r1]) & set(keys[r2])
        if same:
            identical[f"{r1}|{r2}"] = sorted(same)
        a = [k for k in keys[r1] if k not in same]
        b = [k for k in keys[r2] if k not in same]
        if not a or not b:
            continue
        sims = np.asarray([scorer.sims(x, b) for x in a])
        share = float((sims >= THETA).mean())
        matrix.setdefault(r1, {})[r2] = share
        matrix.setdefault(r2, {})[r1] = share
    partners = {
        r: [
            p
            for p, s in sorted(row.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP]
            if s >= MIN_SHARE
        ]
        for r, row in matrix.items()
    }
    out.write_text(
        json.dumps(
            {
                "prereg": "SS110",
                "partners": partners,
                "identical_templates": identical,
                "matrix": matrix,
            },
            indent=1,
        )
    )
    print(json.dumps({k: v for k, v in partners.items() if v}))
    print("pairs with identical template text:", len(identical), sorted(identical))


if __name__ == "__main__":
    main()
