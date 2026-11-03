"""SS110: routing with encoders trained on mined hard negatives (no language model).

.venv/bin/python experiments/sandbox/live_learning/confirm/hardneg_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from locality_eval import match_raw, other_relation_prompts  # noqa: E402
from router_v4_calibrate import build_dev  # noqa: E402
from router_v7_select import dev_scores, threshold_fpr  # noqa: E402
from wrong_route import paired  # noqa: E402

from relic.router import BiScorer, RouterV7, make_scorer  # noqa: E402

CONFIRM = Path("results/live_learning/confirm")
ENCODERS = {
    "baseline": None,
    "hardA": CONFIRM / "relation_encoder_hardA",
    "hardB": CONFIRM / "relation_encoder_hardB",
}


def main() -> None:
    out = CONFIRM / "hardneg_eval.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    raw_cf = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    raw_zs = json.loads((CONFIRM / "raw/zsre.json").read_text())
    cf = json.loads((CONFIRM / "external_counterfact_n1000.json").read_text())["facts"]
    zs = json.loads((CONFIRM / "external_zsre_n1000.json").read_text())["facts"]
    used = {f["subject"] for f in cf + zs}
    dev = build_dev(raw_cf, raw_zs, used)
    records = match_raw(cf, raw_cf)
    e3 = [p["query"] for p in other_relation_prompts(records, raw_cf)]
    natural = json.loads((CONFIRM / "natural_e3_queries.json").read_text())["queries"]

    cols, thetas = {}, {}
    for name, path in ENCODERS.items():
        scorer = make_scorer("relenc") if path is None else BiScorer(str(path))
        _, neg = dev_scores(RouterV7(dev, scorer, -np.inf), dev)
        theta = threshold_fpr([s for p, s in neg if p == "counterfact"])
        thetas[name] = round(float(theta), 4)
        cf_router, zs_router = RouterV7(cf, scorer, theta), RouterV7(zs, scorer, theta)
        col = {}
        for ds, router, facts in (("cf", cf_router, cf), ("zsre", zs_router, zs)):
            for key in ("probe", "paraphrase"):
                col[f"{ds}_{key}"] = [
                    router.route(f[key])[0] == i for i, f in enumerate(facts)
                ]
        col["e3"] = [cf_router.route(q)[0] is not None for q in e3]
        for kind in ("near", "uniform"):
            col[f"natural_{kind}"] = [
                cf_router.route(q["query"])[0] is not None
                for q in natural
                if q["kind"] == kind
            ]
        cols[name] = col
        print(
            name,
            theta,
            {k: round(float(np.mean(v)), 4) for k, v in col.items()},
            flush=True,
        )

    base = cols["baseline"]
    readings = {
        name: {
            k: {
                "rate": round(float(np.mean(v)), 4),
                "minus_baseline": paired(v, base[k]),
            }
            for k, v in col.items()
        }
        for name, col in cols.items()
    }

    def upper(d):
        return d["diff"] + (d["diff"] - d["lower_95_one_sided"])

    rule = {}
    for name in ("hardA", "hardB"):
        r = readings[name]
        near = r["natural_near"]["minus_baseline"]
        rule[name] = {
            "i_near_falls": near["diff"] <= -0.05 and upper(near) < 0,
            "ii_paraphrase": all(
                r[k]["minus_baseline"]["lower_95_one_sided"] >= -0.02
                for k in ("cf_paraphrase", "zsre_paraphrase")
            ),
            "iii_e3": upper(r["e3"]["minus_baseline"]) <= 0.01,
        }
    report = {
        "prereg": "SS110",
        "thetas": thetas,
        "readings": readings,
        "rule": rule,
        "adopted": [n for n, m in rule.items() if all(m.values())],
    }
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("thetas", "rule", "adopted")}, indent=1))


if __name__ == "__main__":
    main()
