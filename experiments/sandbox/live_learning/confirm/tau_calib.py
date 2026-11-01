"""SS107: threshold calibration outside the calibrated phrasings (no language model).

.venv/bin/python experiments/sandbox/live_learning/confirm/tau_calib.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from locality_eval import match_raw, other_relation_prompts  # noqa: E402
from wrong_route import paired  # noqa: E402

from relic.router import RouterV7, make_scorer  # noqa: E402

SEED = 107
THETA = 0.7501
MIN_PER_RELATION = 20
CONFIRM = Path("results/live_learning/confirm")


def decide(router, facts, q, limit_of):
    """Routed entry index under a threshold that may depend on the entry."""
    cand, sims = router.score(q)
    if not cand:
        return None, 0.0
    k = int(np.argmax(sims))
    j, s = cand[k], sims[k]
    return (j if s >= limit_of(facts[j]) else None), s


def main() -> None:
    out = CONFIRM / "tau_calib.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    raw_cf = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    cf = json.loads((CONFIRM / "external_counterfact_n1000.json").read_text())["facts"]
    zs = json.loads((CONFIRM / "external_zsre_n1000.json").read_text())["facts"]
    records = match_raw(cf, raw_cf)
    relation = {
        f["id"]: r["requested_rewrite"]["relation_id"] for f, r in zip(cf, records)
    }
    natural = json.loads((CONFIRM / "natural_e3.json").read_text())["rows"][
        "default_ss76"
    ]

    subjects = sorted({r["fact_id"] for r in natural})
    random.Random(SEED).shuffle(subjects)
    calib_ids = set(subjects[: len(subjects) // 2])
    calib = [r for r in natural if r["fact_id"] in calib_ids]
    test = [r for r in natural if r["fact_id"] not in calib_ids]

    scorer = make_scorer("relenc", device="cpu")
    cf_router = RouterV7(cf, scorer, THETA)
    zs_router = RouterV7(zs, scorer, THETA)

    scored = []
    for r in calib:
        cand, sims = cf_router.score(r["query"])
        if cand:
            k = int(np.argmax(sims))
            scored.append((relation[cf[cand[k]]["id"]], sims[k]))
    pooled = np.asarray([s for _, s in scored])
    theta_n = float(np.quantile(pooled, 0.95, method="higher"))
    per_relation = {}
    for rel in sorted({r for r, _ in scored}):
        values = [s for r, s in scored if r == rel]
        if len(values) >= MIN_PER_RELATION:
            per_relation[rel] = max(
                THETA, float(np.quantile(values, 0.95, method="higher"))
            )

    def limit(arm, fact):
        if arm == "global":
            return THETA
        if arm == "N":
            return theta_n
        return per_relation.get(relation.get(fact["id"]), THETA)

    arms = ("global", "N", "R")
    readings = {}
    for kind in ("near", "uniform"):
        rows = [r for r in test if r["kind"] == kind]
        routed = {
            a: [
                decide(cf_router, cf, r["query"], lambda f, a=a: limit(a, f))[0]
                is not None
                for r in rows
            ]
            for a in arms
        }
        readings[f"natural_{kind}_routed"] = {
            a: {
                "n": len(rows),
                "rate": round(float(np.mean(routed[a])), 4),
                "minus_global": paired(routed[a], routed["global"]),
            }
            for a in arms
        }
        # automatic leak with both served, on the routed test questions
        readings[f"natural_{kind}_auto_leak"] = {
            a: round(
                float(
                    np.mean(
                        [
                            v and r["both"]["leak_target"]
                            for v, r in zip(routed[a], rows)
                        ]
                    )
                ),
                4,
            )
            for a in arms
        }

    # routing of the stores' own questions
    for name, router, facts in (
        ("counterfact", cf_router, cf),
        ("zsre", zs_router, zs),
    ):
        for key in ("probe", "paraphrase"):
            correct = {
                a: [
                    decide(router, facts, f[key], lambda g, a=a: limit(a, g))[0] == i
                    for i, f in enumerate(facts)
                ]
                for a in arms
            }
            readings[f"{name}_{key}_route_correct"] = {
                a: {
                    "rate": round(float(np.mean(correct[a])), 4),
                    "minus_global": paired(correct[a], correct["global"]),
                }
                for a in arms
            }
    e3 = [p["query"] for p in other_relation_prompts(records, raw_cf)]
    readings["E3_routed"] = {
        a: round(
            float(
                np.mean(
                    [
                        decide(cf_router, cf, q, lambda g, a=a: limit(a, g))[0]
                        is not None
                        for q in e3
                    ]
                )
            ),
            4,
        )
        for a in arms
    }

    near = readings["natural_near_routed"]
    rule = {}
    for a in ("N", "R"):
        d = near[a]["minus_global"]
        upper = d["diff"] + (d["diff"] - d["lower_95_one_sided"])
        rule[a] = {
            "i_near_falls": d["diff"] <= -0.10 and upper < 0,
            "ii_paraphrase": all(
                readings[f"{s}_paraphrase_route_correct"][a]["minus_global"][
                    "lower_95_one_sided"
                ]
                >= -0.02
                for s in ("counterfact", "zsre")
            ),
        }
    report = {
        "prereg": "SS107",
        "n_calibration": len(calib),
        "n_test": len(test),
        "theta_natural": round(theta_n, 4),
        "per_relation": {k: round(v, 4) for k, v in per_relation.items()},
        "readings": readings,
        "rule": rule,
        "adopted": [a for a, m in rule.items() if all(m.values())],
    }
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
