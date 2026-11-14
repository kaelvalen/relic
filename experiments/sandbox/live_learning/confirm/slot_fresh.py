"""SS104 (b): residual routes after removal, on zsRE facts unused so far (no language model).

.venv/bin/python experiments/sandbox/live_learning/confirm/slot_fresh.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from external import adapt_zsre  # noqa: E402
from router_v4_calibrate import build_dev  # noqa: E402

from relic.ledger import Ledger  # noqa: E402
from relic.router import (
    make_scorer,  # noqa: E402
    normalize,  # noqa: E402
)

SEED = 104
N, REMOVED = 1000, 100
BOOT = 10000
THETA_ROUTE, THETA_SLOT = 0.7501, 0.8985
LIMIT = 0.05
CONFIRM = Path("results/live_learning/confirm")


class NoStore:
    """The ledger's view of a store when only routing is measured."""

    base_hash, router_config = "", ""

    def __init__(self):
        self.deltas, self.keys, self.notes = {}, {}, {}


def fresh_facts(raw_cf, raw_zs):
    used = {
        f["subject"]
        for name in ("external_counterfact_n1000", "external_zsre_n1000")
        for f in json.loads((CONFIRM / f"{name}.json").read_text())["facts"]
    }
    used |= {f["subject"] for f in build_dev(raw_cf, raw_zs, used)}
    pool = [
        f
        for f in (adapt_zsre(r) for r in raw_zs)
        if f is not None and f["subject"] not in used
    ]
    by_subject = {}
    for f in pool:
        by_subject.setdefault(f["subject"], []).append(f)
    rng = random.Random(SEED)
    subjects = sorted(by_subject)
    rng.shuffle(subjects)
    # subjects with several records first: a duplicate has to exist to be recognised
    subjects.sort(key=lambda s: len(by_subject[s]) < 2)
    facts = []
    for s in subjects:
        for f in by_subject[s]:
            if len(facts) < N:
                facts.append({**f, "id": f"z{len(facts):04d}"})
    rng.shuffle(facts)
    return facts


def residual(facts, scorer, slot_theta, duplicate_theta, per_entry=False):
    ledger = Ledger(
        NoStore(),
        scorer,
        THETA_ROUTE,
        slot_theta,
        duplicate_theta or slot_theta,
        keep_history=False,  # SS104 as run: an update drops the entry it replaces
    )
    holder, actions = {}, {"insert": 0, "duplicate": 0, "update": 0}
    for f in facts:
        if per_entry:
            ledger.entries[f["id"]] = f
            ledger.phrasings[f["id"]] = []
            ledger._router = None
            holder[f["id"]] = f["id"]
            actions["insert"] += 1
            continue
        event = ledger.commit(f)
        actions[event["action"]] += 1
        if event["action"] == "update":
            holder = {
                k: (event["entry"] if v == event["superseded"] else v)
                for k, v in holder.items()
            }
        holder[f["id"]] = event["entry"]
    removed = facts[:REMOVED]
    for fid in sorted({holder[f["id"]] for f in removed}):
        if per_entry:
            del ledger.entries[fid], ledger.phrasings[fid]
            ledger._router = None
        else:
            ledger.revoke(fid)
    router, active = ledger.router
    rows = []
    for f in removed:
        j, score = router.route(f["probe"])
        same = j is not None and normalize(active[j]["subject"]) == normalize(
            f["subject"]
        )
        rows.append(
            {
                "fact_id": f["id"],
                "residual": same,
                "routed": j is not None,
                "score": round(score, 4),
            }
        )
    values = np.asarray([r["residual"] for r in rows], dtype=float)
    rng = np.random.default_rng(SEED)
    boots = rng.choice(values, size=(BOOT, values.size), replace=True).mean(axis=1)
    by_id = {f["id"]: f for f in facts}
    different = sum(
        normalize(by_id[f["id"]]["answer"])
        != normalize(by_id[holder[f["id"]]]["answer"])
        for f in facts
    )
    return {
        "actions": actions,
        "merged_with_different_answer": different,
        "entries_removed": len({holder[f["id"]] for f in removed}),
        "residual_share": round(float(values.mean()), 4),
        "upper_95_one_sided": round(float(np.percentile(boots, 95)), 4),
        "rows": rows,
    }


def main() -> None:
    out = CONFIRM / "slot_fresh.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    raw_cf = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    raw_zs = json.loads((CONFIRM / "raw/zsre.json").read_text())
    facts = fresh_facts(raw_cf, raw_zs)
    scorer = make_scorer("relenc", device="cpu")
    scorer.prefetch([t for f in facts for t in (f["probe"], f["paraphrase"])])
    report = {
        "prereg": "SS104",
        "n": len(facts),
        "subjects": len({f["subject"] for f in facts}),
        "per_entry": residual(facts, scorer, THETA_SLOT, None, per_entry=True),
        "slots_ss96": residual(facts, scorer, THETA_SLOT, None),
        "asymmetric": residual(facts, scorer, THETA_SLOT, THETA_ROUTE),
    }
    report["holds"] = report["asymmetric"]["upper_95_one_sided"] <= LIMIT
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(
        json.dumps(
            {
                k: (
                    {x: y for x, y in v.items() if x != "rows"}
                    if isinstance(v, dict)
                    else v
                )
                for k, v in report.items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
