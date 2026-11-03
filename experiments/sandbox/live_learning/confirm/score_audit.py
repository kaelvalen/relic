"""SS97: audit of the answer score on saved outputs (no model run).

.venv/bin/python experiments/sandbox/live_learning/confirm/score_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from locality_eval import match_raw  # noqa: E402

from relic.metrics import hit, word_hit  # noqa: E402

CONFIRM = Path("results/live_learning/confirm")
ARMS = ("adapter", "retrieval", "both")
LIMIT = 0.02


def rate(xs) -> float | None:
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def audit(rows, facts, old=None):
    out = {}
    for kind in ("probe", "paraphrase"):
        for arm in ARMS:
            sub, word, strict, both = [], [], [], []
            for row, fact in zip(rows, facts):
                resp = row[kind]["responses"][arm]
                sub.append(hit(resp, fact["answer"]))
                word.append(word_hit(resp, fact["answer"]))
                if old is not None:
                    o = old[fact["id"]]
                    old_there = word_hit(resp, o)
                    strict.append(word[-1] and not old_there)
                    if sub[-1]:
                        both.append(old_there)
            out[f"{kind}_{arm}"] = {
                "substring": rate(sub),
                "word": rate(word),
                "strict": rate(strict) if old is not None else None,
                "hits_with_old_answer": rate(both) if old is not None else None,
            }
    return out


def short_targets(rows, facts):
    idx = [i for i, f in enumerate(facts) if len(f["answer"]) <= 2]
    out = {"n": len(idx), "targets": sorted({facts[i]["answer"] for i in idx})}
    for kind in ("probe", "paraphrase"):
        resp = [(rows[i][kind]["responses"]["both"], facts[i]["answer"]) for i in idx]
        out[kind] = {
            "substring": rate(hit(r, a) for r, a in resp),
            "word": rate(word_hit(r, a) for r, a in resp),
        }
    return out


def main() -> None:
    raw = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    report = {"prereg": "SS97", "limit": LIMIT, "datasets": {}}
    for dataset, tag, facts_file in (
        ("counterfact", "cf", "external_counterfact_n1000.json"),
        ("zsre", "zsre", "external_zsre_n1000.json"),
    ):
        facts = json.loads((CONFIRM / facts_file).read_text())["facts"]
        saved = json.loads((CONFIRM / f"{tag}_n1000_v8_eval.json").read_text())
        rows = saved["examples"]["serving"]
        assert [r["fact_id"] for r in rows] == [f["id"] for f in facts]
        old = None
        if dataset == "counterfact":
            old = {
                f["id"]: r["requested_rewrite"]["target_true"]["str"]
                for f, r in zip(facts, match_raw(facts, raw))
            }
        report["datasets"][dataset] = {
            "arms": audit(rows, facts, old),
            "short_targets": short_targets(rows, facts),
            "removal_gone": {
                "substring": rate(r["gone"] for r in saved["examples"]["removal"]),
                "word": rate(
                    not word_hit(r["response"], f["answer"])
                    for r, f in zip(saved["examples"]["removal"], facts)
                ),
            },
        }
    drops = []
    for dataset, d in report["datasets"].items():
        for kind in ("probe", "paraphrase"):
            a = d["arms"][f"{kind}_both"]
            for name in ("word", "strict"):
                if a[name] is not None:
                    drops.append(
                        (round(a["substring"] - a[name], 4), dataset, kind, name)
                    )
    report["largest_drops_both_arm"] = sorted(drops, reverse=True)[:6]
    report["stricter_score_becomes_primary"] = drops != [] and max(drops)[0] > LIMIT
    out = CONFIRM / "score_audit.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
