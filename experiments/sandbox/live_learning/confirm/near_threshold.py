"""SS138: per-relation thresholds on the fresh facts of SS136, as served (no language model).

.venv/bin/python experiments/sandbox/live_learning/confirm/near_threshold.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

from adapter_only import CONFIRM, draw_facts, lower_upper  # noqa: E402
from adapter_size import KAPPA  # noqa: E402
from locality_eval import match_raw  # noqa: E402
from wrong_route import paired  # noqa: E402

from relic.metrics import word_hit  # noqa: E402
from relic.router import RouterV7, make_scorer  # noqa: E402

SEED = 136
THETA = 0.7501
MIN_PER_RELATION = 20
STORES = ("current", "ladder")


def auroc(pos, neg) -> float:
    """Probability that a positive scores above a negative, ties counting half."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(np.mean((pos > neg) + 0.5 * (pos == neg)))


def chosen(router, q):
    cand, sims = router.score(q)
    if not cand:
        return None, 0.0
    k = int(np.argmax(sims))
    return cand[k], float(sims[k])


def calibrate(scorer) -> tuple[dict, int]:
    """The recipe of SS107 on every natural question of the earlier runs."""
    raw_cf = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    old = json.loads((CONFIRM / "external_counterfact_n1000.json").read_text())["facts"]
    for f, r in zip(old, match_raw(old, raw_cf)):
        f["relation"] = r["requested_rewrite"]["relation_id"]
    natural = json.loads((CONFIRM / "natural_e3.json").read_text())["rows"][
        "default_ss76"
    ]
    seen = draw_facts(150)[0]["cf"]
    questions = json.loads((CONFIRM / "served_adapter_questions.json").read_text())
    scored = []
    for facts, queries in (
        (old, [r["query"] for r in natural]),
        (seen, [q["query"] for q in questions if q["subject_found"]]),
    ):
        router = RouterV7(facts, scorer, THETA)
        for q in queries:
            j, s = chosen(router, q)
            if j is not None:
                scored.append((facts[j]["relation"], s))
    out = {}
    for rel in sorted({r for r, _ in scored}):
        values = [s for r, s in scored if r == rel]
        if len(values) >= MIN_PER_RELATION:
            out[rel] = max(THETA, float(np.quantile(values, 0.95, method="higher")))
    return out, len(scored)


def read_arm(rows, facts, kept, scorer, tau, store) -> list[dict]:
    """Each saved row under this arm's thresholds; a query below them gets the base answer."""
    router = RouterV7(kept, scorer, THETA, relation_tau=tau)
    out = []
    for r in rows:
        j, _ = router.route(r["q"])
        fid = None if j is None else kept[j]["id"]
        assert fid in (None, r["route"][store]), "a raised threshold moved a route"
        text = r["base"] if fid is None else r["response"][store]
        answer = facts[r["id"]]["answer"]
        got = word_hit(text, answer)
        out.append(
            {
                "reading": r["reading"],
                "id": r["id"],
                "kind": r.get("kind"),
                "routed": fid is not None,
                "score": got
                if r["reading"] in ("edit", "paraphrase")
                else got and not word_hit(r["base"], answer),
            }
        )
    return out


def summarize(read, ref) -> dict:
    groups = {
        "edit_cf": lambda x: x["reading"] == "edit" and x["id"].startswith("cf"),
        "paraphrase_cf": lambda x: (
            x["reading"] == "paraphrase" and x["id"].startswith("cf")
        ),
        "paraphrase_zs": lambda x: (
            x["reading"] == "paraphrase" and x["id"].startswith("zs")
        ),
        "e3": lambda x: x["reading"] == "e3",
        "natural": lambda x: x["reading"] == "natural",
        "near": lambda x: x["reading"] == "natural" and x["kind"] == "near",
        "uniform": lambda x: x["reading"] == "natural" and x["kind"] == "uniform",
    }
    out = {}
    for name, keep in groups.items():
        a = [x for x in read if keep(x)]
        b = [x for x in ref if keep(x)]
        out[name] = {
            "n": len(a),
            "rate": round(float(np.mean([x["score"] for x in a])), 4),
            "routed": round(float(np.mean([x["routed"] for x in a])), 4),
            "minus_global": paired([x["score"] for x in a], [x["score"] for x in b]),
        }
    return out


def decide(s) -> dict:
    near = s["near"]["minus_global"]
    rule = {
        "i_near": near["diff"] <= -0.05 and lower_upper(near)[1] < 0,
        "ii_paraphrase": s["paraphrase_cf"]["minus_global"]["lower_95_one_sided"]
        >= -0.02,
        "iii_edit": s["edit_cf"]["minus_global"]["lower_95_one_sided"] >= -0.01,
    }
    rule["holds"] = all(rule.values())
    return rule


def diagnostic(rows, facts, kept, scorer) -> dict:
    router = RouterV7(kept, scorer, THETA)
    para, near, by_rel = [], [], {}
    for r in rows:
        if not r["id"].startswith("cf"):
            continue
        j, s = chosen(router, r["q"])
        if j is None:
            continue
        rel = facts[r["id"]]["relation"]
        if r["reading"] == "paraphrase" and kept[j]["id"] == r["id"]:
            para.append(s)
            by_rel.setdefault(rel, ([], []))[0].append(s)
        elif r["reading"] == "natural" and r["kind"] == "near":
            near.append(s)
            by_rel.setdefault(rel, ([], []))[1].append(s)
    # the highest threshold that keeps paraphrase routing within 2 points of global
    tau = max(THETA, float(np.quantile(para, 0.02 + np.mean(np.asarray(para) < THETA))))
    n_near = sum(1 for r in rows if r["reading"] == "natural" and r["kind"] == "near")
    return {
        "pooled": round(auroc(para, near), 4),
        "n_paraphrase": len(para),
        "n_near_with_candidate": len(near),
        "threshold_at_2_points": round(tau, 4),
        "near_routed_at_it": round(sum(s >= tau for s in near) / n_near, 4),
        "near_routed_global": round(sum(s >= THETA for s in near) / n_near, 4),
        "per_relation": {
            rel: {"auroc": round(auroc(p, n), 3), "n_para": len(p), "n_near": len(n)}
            for rel, (p, n) in sorted(by_rel.items())
            if len(p) >= 5 and len(n) >= 5
        },
    }


def main() -> None:
    out = CONFIRM / "near_threshold.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    saved = json.loads((CONFIRM / "fresh_ladder.json").read_text())
    seen = {f["subject"] for fs in draw_facts(150)[0].values() for f in fs}
    sets = draw_facts(400, seed=SEED, exclude=seen)[0]
    facts = {f["id"]: f for fs in sets.values() for f in fs}
    key = {f["id"]: f"{ds}:{f['id']}" for ds, fs in sets.items() for f in fs}
    rows = saved["served"]

    def held(i, store):
        d0 = saved["rows"]["A0"][key[i]]["d_heldout"] <= KAPPA
        d1 = saved["rows"]["A1"][key[i]]["d_heldout"] <= KAPPA
        return d0 if store == "current" else d1 or key[i] in saved["first_rung"] or d0

    scorer = make_scorer("relenc", device="cpu")
    tau_all, n_calib = calibrate(scorer)
    arms = {
        "global": {},
        "R107": json.loads((CONFIRM / "tau_calib.json").read_text())["per_relation"],
        "Rall": tau_all,
    }
    report = {
        "prereg": "SS138",
        "n_calibration_Rall": n_calib,
        "thresholds": {
            a: {k: round(v, 4) for k, v in t.items()} for a, t in arms.items()
        },
        "stores": {},
    }
    for store in STORES:
        kept = [f for i, f in facts.items() if held(i, store)]
        assert len(kept) == saved["store_size"][store]
        reads = {
            a: read_arm(rows, facts, kept, scorer, t, store) for a, t in arms.items()
        }
        # the global arm has to reproduce the saved run exactly
        assert all(
            x["score"] == r["score"][store]
            and x["routed"] == (r["route"][store] is not None)
            for x, r in zip(reads["global"], rows)
        ), "the global arm differs from the saved run"
        report["stores"][store] = {
            a: summarize(reads[a], reads["global"]) for a in arms
        }
    current = report["stores"]["current"]
    report["rule"] = {a: decide(current[a]) for a in ("R107", "Rall")}
    passing = [a for a in ("R107", "Rall") if report["rule"][a]["holds"]]
    report["adopted"] = min(
        passing, key=lambda a: current[a]["near"]["rate"], default=None
    )
    kept = [f for i, f in facts.items() if held(i, "current")]
    report["auroc"] = diagnostic(rows, facts, kept, scorer)
    out.write_text(json.dumps(report, indent=1))
    for a in arms:
        print(
            a, json.dumps({k: (v["rate"], v["routed"]) for k, v in current[a].items()})
        )
        print(
            " ",
            json.dumps(
                {
                    k: current[a][k]["minus_global"]
                    for k in ("near", "paraphrase_cf", "edit_cf")
                }
            ),
        )
    print("thresholds", json.dumps(report["thresholds"]), "n", n_calib)
    print("rule", json.dumps(report["rule"]), "adopted", report["adopted"])
    print("auroc", json.dumps(report["auroc"]))


if __name__ == "__main__":
    main()
