"""SS106: a three-tier gate, on saved outputs (U1, U2) and with new note answers (L).

.venv/bin/python experiments/sandbox/live_learning/confirm/tier_gate.py
.venv/bin/python experiments/sandbox/live_learning/confirm/tier_gate.py --serve
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from routed_eval import MANIFEST_PATH  # noqa: E402
from router_v4_calibrate import build_dev  # noqa: E402
from wrong_route import paired  # noqa: E402

from relic.ledger import DeltaStore  # noqa: E402
from relic.metrics import hit, word_hit  # noqa: E402
from relic.router import (  # noqa: E402
    RouterV7,
    find_subject,
    key_text,
    make_scorer,
    relation_text,
)
from relic.serving import base_answer, rag_system, trim  # noqa: E402

THETA = 0.7501
CONFIRM = Path("results/live_learning/confirm")
FILES = {
    "counterfact": ("external_counterfact_n1000.json", "cf"),
    "zsre": ("external_zsre_n1000.json", "zsre"),
}


def dev_threshold(scorer, dev, fpr):
    """The threshold with the given false-positive rate on CounterFact dev negatives."""
    neg = []
    for f in dev:
        if f["relation_id"] is None:
            continue
        span = find_subject(f["negative"], f["subject"])
        if span is not None:
            key = key_text(f["probe"], f["subject"])
            neg.append(scorer.sims(relation_text(f["negative"], span), [key])[0])
    return float(np.quantile(neg, 1 - fpr, method="higher"))


def tier(score, policy):
    """adapter and note, note alone, or base."""
    hi, lo = policy
    if score is None or score < lo:
        return "base"
    return "both" if score >= hi else "note"


def rows_of(dataset, tag, facts, router, extra_note):
    """One record per query: score and the base, note and both responses."""
    saved = json.loads((CONFIRM / f"{tag}_n1000_v8_eval.json").read_text())
    out = {"probe": [], "paraphrase": []}
    for r, f in zip(saved["examples"]["serving"], facts):
        for kind in out:
            s = r[kind]
            score = s["route_score"] if s["route_fact_id"] is not None else None
            resp = s["responses"]
            # an abstained query saved the base answer under every arm
            base = resp["adapter"] if score is None else None
            _, raw = router.route(f[kind])
            out[kind].append(
                {
                    "query": f[kind],
                    "answer": f["answer"],
                    "score": raw if score is None else score,
                    "base": base,
                    "note": resp["retrieval"],
                    "both": resp["both"],
                    "abstained": score is None,
                }
            )
    for kind, rows in saved["examples"]["locality"].items():
        out[kind] = []
        for r in rows:
            _, raw = router.route(r["query"])
            out[kind].append(
                {
                    "query": r["query"],
                    "score": raw,
                    "base": r["base"],
                    "note": r["note"]["response"],
                    "both": r["both"]["response"],
                    "note_leak": r["note"]["leak"],
                    "both_leak": r["both"]["leak"],
                    "abstained": not r["routed"],
                }
            )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="note answers for the band L")
    args = ap.parse_args(argv)
    out_path = CONFIRM / ("tier_gate_served.json" if args.serve else "tier_gate.json")
    if out_path.exists():
        ap.error(f"refusing to overwrite {out_path}")
    raw_cf = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    raw_zs = json.loads((CONFIRM / "raw/zsre.json").read_text())
    data = {
        d: json.loads((CONFIRM / f).read_text())["facts"] for d, (f, _) in FILES.items()
    }
    used = {f["subject"] for fs in data.values() for f in fs}
    dev = build_dev(raw_cf, raw_zs, used)
    scorer = make_scorer("relenc", device="cuda" if args.serve else "cpu")
    t2, t1 = dev_threshold(scorer, dev, 0.02), dev_threshold(scorer, dev, 0.01)
    t10 = dev_threshold(scorer, dev, 0.10)
    policies = {
        "current": (THETA, THETA),
        "U2": (t2, THETA),
        "U1": (t1, THETA),
    }
    if args.serve:
        policies["L"] = (THETA, t10)
    report = {
        "prereg": "SS106",
        "thresholds": {"t2": round(t2, 4), "t1": round(t1, 4), "t10": round(t10, 4)},
        "readings": {},
    }
    store = None
    if args.serve:
        manifest = json.loads(MANIFEST_PATH.read_bytes())
        store = DeltaStore(manifest["models"]["base"]["id"], steps=16)
        store.deltas = {}
        store.materialize()

    natural = json.loads((CONFIRM / "natural_e3.json").read_text())["rows"][
        "default_ss76"
    ]
    cf = data["counterfact"]
    cf_router = RouterV7(cf, scorer, THETA)
    index = {f["id"]: i for i, f in enumerate(cf)}

    def lower_band_note(q, router, facts):
        """The note answer of the best candidate, for a query below the routing threshold."""
        cand, sims = router.score(q)
        if not cand:
            return None
        j = cand[int(np.argmax(sims))]
        resp = base_answer(store, q, system=rag_system("instruct", facts, j))
        trim()
        return resp

    for dataset, (_, tag) in FILES.items():
        facts = data[dataset]
        router = (
            cf_router if dataset == "counterfact" else RouterV7(facts, scorer, THETA)
        )
        groups = rows_of(dataset, tag, facts, router, None)
        if args.serve:
            for rows in groups.values():
                for r in rows:
                    if r["abstained"] and r["score"] >= t10:
                        r["note"] = lower_band_note(r["query"], router, facts)
        for kind in ("probe", "paraphrase"):
            rows = groups[kind]
            for r in rows:
                if r["base"] is None:
                    r["base"] = r["note"] if r["abstained"] else ""
            current = None
            for name, policy in policies.items():
                sub, word = [], []
                for r in rows:
                    t = (
                        tier(r["score"], policy)
                        if not r["abstained"]
                        else ("note" if name == "L" and r["score"] >= t10 else "base")
                    )
                    resp = r[t] if t != "base" else (r["base"] or r["both"])
                    sub.append(hit(resp, r["answer"]))
                    word.append(word_hit(resp, r["answer"]))
                if name == "current":
                    current = (sub, word)
                report["readings"].setdefault(f"{dataset}_{kind}", {})[name] = {
                    "substring": round(float(np.mean(sub)), 4),
                    "word": round(float(np.mean(word)), 4),
                    "note_tier": round(
                        float(
                            np.mean(
                                [
                                    (not r["abstained"])
                                    and tier(r["score"], policy) == "note"
                                    for r in rows
                                ]
                            )
                        ),
                        4,
                    ),
                    "minus_current": paired(sub, current[0]),
                }
        for kind in ("E3_other_relation", "E2_neighborhood"):
            rows = groups.get(kind)
            if not rows:
                continue
            current = None
            for name, policy in policies.items():
                leak = []
                for r in rows:
                    if r["abstained"]:
                        if name == "L" and r["score"] >= t10:
                            answers = [f["answer"] for f in facts]
                            leak.append(
                                any(
                                    hit(r["note"], w) and not hit(r["base"], w)
                                    for w in answers
                                )
                            )
                        else:
                            leak.append(False)
                    else:
                        t = tier(r["score"], policy)
                        leak.append(r["both_leak"] if t == "both" else r["note_leak"])
                if name == "current":
                    current = leak
                report["readings"].setdefault(f"{dataset}_{kind}", {})[name] = {
                    "leak": round(float(np.mean(leak)), 4),
                    "minus_current": paired(leak, current),
                }

    for kind in ("near", "uniform"):
        rows = [r for r in natural if r["kind"] == kind]
        if not rows:
            continue
        current = None
        for name, policy in policies.items():
            leak = []
            for r in rows:
                if not r["routed"]:
                    if name == "L" and r["score"] >= t10:
                        resp = lower_band_note(r["query"], cf_router, cf)
                        j = index[cf[cf_router.score(r["query"])[0][0]]["id"]]
                        leak.append(
                            hit(resp, cf[j]["answer"])
                            and not hit(r["base"], cf[j]["answer"])
                        )
                    else:
                        leak.append(False)
                else:
                    t = tier(r["score"], policy)
                    leak.append(r[t]["leak_target"])
            if name == "current":
                current = leak
            report["readings"].setdefault(f"natural_{kind}", {})[name] = {
                "leak": round(float(np.mean(leak)), 4),
                "minus_current": paired(leak, current),
            }

    near = report["readings"]["natural_near"]
    margins = {}
    for name in policies:
        if name == "current":
            continue
        d = near[name]["minus_current"]
        upper = d["diff"] + (d["diff"] - d["lower_95_one_sided"])
        keys = [f"{ds}_{k}" for ds in FILES for k in ("probe", "paraphrase")]
        margins[name] = {
            "i_leak_falls": d["diff"] <= -0.03 and upper < 0,
            "ii_noninferior": all(
                report["readings"][k][name]["minus_current"]["lower_95_one_sided"]
                >= -0.01
                for k in keys
            ),
        }
    report["rule"] = margins
    report["adopted"] = [n for n, m in margins.items() if all(m.values())]
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
