"""SS143: does `Ledger` build the store SS136 tested?

.venv/bin/python experiments/sandbox/live_learning/confirm/ledger_parity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

from adapter_only import CONFIRM, MODEL, draw_facts  # noqa: E402

from relic.ledger import DeltaStore, Ledger  # noqa: E402
from relic.router import make_scorer  # noqa: E402

SEED = 136


def pick(saved, keys) -> list[str]:
    """Four first-attempt facts per dataset, and two facts per rung the run stored at."""
    first = saved["first_rung"]
    plain = [k for k in keys if k not in first and k not in saved["no_rung"]]
    out = [k for k in plain if k.startswith("cf:")][:4]
    out += [k for k in plain if k.startswith("zs:")][:4]
    for rung in ("R1", "L2", "L4", "L8"):
        out += sorted(k for k, r in first.items() if r == rung)[:2]
    return out


def main() -> None:
    out = CONFIRM / "ledger_parity.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    saved = json.loads((CONFIRM / "fresh_ladder.json").read_text())
    seen = {f["subject"] for fs in draw_facts(150)[0].values() for f in fs}
    sets = draw_facts(400, seed=SEED, exclude=seen)[0]
    facts = {f"{ds}:{f['id']}": f for ds, fs in sets.items() for f in fs}
    chosen = pick(saved, list(facts))

    # the run served each fact from a store of 799, so the route is read off the saved row
    responses = {}
    for r in saved["served"]:
        if r["reading"] in ("edit", "paraphrase") and r["route"]["ladder"] == r["id"]:
            responses[(r["id"], r["reading"])] = r["response"]["ladder"]

    ledger = Ledger(
        DeltaStore(MODEL, steps=16, capacity=1), make_scorer("relenc"), 0.7501, 0.8985
    )
    rows = []
    for key in chosen:
        f = facts[key]
        rung = saved["first_rung"].get(key, "A1")
        expected = (saved["rows"]["A1"] if rung == "A1" else saved["rungs"][rung])[key][
            "d_heldout"
        ]
        rewrites = ledger._rewrites(f)
        event = ledger.write(f)
        row = {
            "key": key,
            "rewrites_match": rewrites == saved["rewrites"][f["id"]],
            "rung": event.get("rung"),
            "rung_expected": rung,
            "d": event.get("kld"),
            "d_expected": expected,
            "answers": {},
        }
        for reading, field in (("edit", "probe"), ("paraphrase", "paraphrase")):
            want = responses.get((f["id"], reading))
            if want is not None:
                got = ledger.ask(f[field])
                row["answers"][reading] = {
                    "got": got["answer"],
                    "want": want,
                    "entry": got["entry"] == f["id"],
                }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False)[:300], flush=True)

    answers = [a for r in rows for a in r["answers"].values()]
    d_gap = max(abs(r["d"] - r["d_expected"]) for r in rows)
    rule = {
        "rewrites": all(r["rewrites_match"] for r in rows),
        "rung": all(r["rung"] == r["rung_expected"] for r in rows),
        "d_gap": round(d_gap, 5),
        "d_ok": d_gap <= 0.01,
        "answers_same": round(
            sum(a["got"] == a["want"] for a in answers) / len(answers), 4
        ),
        "routed_to_own_entry": all(a["entry"] for a in answers),
        "n_facts": len(rows),
        "n_answers": len(answers),
    }
    rule["holds"] = (
        rule["rewrites"]
        and rule["rung"]
        and rule["d_ok"]
        and rule["answers_same"] >= 0.95
        and rule["routed_to_own_entry"]
    )
    out.write_text(
        json.dumps(
            {"prereg": "SS143", "rule": rule, "facts": rows},
            ensure_ascii=False,
            indent=1,
        )
    )
    print(json.dumps(rule, indent=1))
    assert rule["holds"], "the ledger does not build the store SS136 tested"


if __name__ == "__main__":
    main()
