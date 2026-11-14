"""SS154: the router's decision in Turkish and English on BMIKE-53 CounterFact (prereg entry 154).

python xling_route.py --smoke   # 40 facts, two encoders
python xling_route.py           # the pinned reading -> xling_route.json
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from relic.router import RELATION_TAU, BiScorer, RouterV7

CONFIRM = Path("results/live_learning/confirm")
DATA = CONFIRM / "raw/bmike_counterfact_tr.json"
SHA = "6e024ff5e7a3c798d86bce215c18783379025d73c8c0e0cb84301cddcf0a0eb9"
ENCODERS = {
    "relenc": str(CONFIRM / "relation_encoder_mpnet"),
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "mmpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "relenc-m": str(CONFIRM / "relation_encoder_mmpnet"),
    "relenc-b": str(CONFIRM / "relation_encoder_bil"),  # SS158
}
PAIRS = (("relenc-m", "relenc"), ("relenc-b", "relenc"), ("relenc-b", "relenc-m"))
THETA = 0.7501
SEED, BOOT = 154, 10000


def load():
    import hashlib

    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == SHA
    items = json.loads(DATA.read_text())
    raw = {
        r["case_id"]: r["requested_rewrite"]
        for r in json.loads((CONFIRM / "raw/counterfact.json").read_text())
    }
    facts = []
    for it in items:
        rw = raw[it["en"]["case_id"]]
        facts.append(
            {
                "subject": rw["subject"],
                "probe": rw["prompt"].replace("{}", rw["subject"]),
                "relation": rw["relation_id"],
                "en": it["en"],
                "tr": it["tr"],
            }
        )
    return facts


def e3(facts) -> list:
    """For fact i: the src of the next fact j of another relation, with i's subject in j's place."""
    out = []
    n = len(facts)
    for i, f in enumerate(facts):
        for d in range(1, n):
            g = facts[(i + d) % n]
            s = g["en"]["subject"]
            if (
                g["relation"] != f["relation"]
                and s in g["en"]["src"]
                and g["tr"]["subject"] in g["tr"]["src"]
            ):
                out.append(
                    {
                        "en": g["en"]["src"].replace(s, f["subject"]),
                        "tr": g["tr"]["src"].replace(g["tr"]["subject"], f["subject"]),
                        "from": (i + d) % n,
                    }
                )
                break
    assert len(out) == n
    return out


def read(router, facts, negs) -> dict:
    """Per fact: own-entry score and routed index for positives; top score and routing for E3."""
    rows = []
    for i, (f, ng) in enumerate(zip(facts, negs)):
        row = {}
        for name, q in (
            ("en_reph", f["en"]["rephrase"]),
            ("tr_src", f["tr"]["src"]),
            ("tr_reph", f["tr"]["rephrase"]),
        ):
            cand, sims = router.score(q)
            j, _ = router.route(q)
            row[name] = {
                "score": float(sims[cand.index(i)]) if i in cand else None,
                "right": j == i,
            }
        for lang in ("en", "tr"):
            cand, sims = router.score(ng[lang])
            j, _ = router.route(ng[lang])
            row[f"e3_{lang}"] = {
                "score": float(max(sims)) if cand else None,
                "routed": j is not None,
            }
        rows.append(row)
    return rows


def auc(pos, neg) -> float:
    """Mann-Whitney AUROC, ties counting half."""
    x = np.concatenate([pos, neg])
    order = x.argsort(kind="mergesort")
    ranks = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):  # average ranks over ties
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


POS = {"en": ("en_reph",), "tr": ("tr_src", "tr_reph")}


def scores(rows, idx, lang):
    pos = [rows[k][p]["score"] for k in idx for p in POS[lang]]
    neg = [rows[k][f"e3_{lang}"]["score"] for k in idx]
    return (
        np.array([s for s in pos if s is not None]),
        np.array([s for s in neg if s is not None]),
    )


def boot(rows_by_enc, facts, keep):
    rng = np.random.default_rng(SEED)
    base = np.array(keep)
    draws = [base[rng.integers(0, len(base), len(base))] for _ in range(BOOT)]
    out = {}
    for lang in POS:
        point = {e: auc(*scores(r, base, lang)) for e, r in rows_by_enc.items()}
        dist = {
            e: np.array([auc(*scores(r, d, lang)) for d in draws])
            for e, r in rows_by_enc.items()
        }
        out[lang] = {
            e: {
                "auroc": round(point[e], 4),
                "ci95": [
                    round(float(np.percentile(dist[e], q)), 4) for q in (2.5, 97.5)
                ],
            }
            for e in rows_by_enc
        }
        for x, y in PAIRS:
            if x in dist and y in dist:
                d = dist[x] - dist[y]
                out[lang][f"{x}_minus_{y}"] = {
                    "diff": round(point[x] - point[y], 4),
                    "ci95": [round(float(np.percentile(d, q)), 4) for q in (2.5, 97.5)],
                }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--encoders", default="relenc,mpnet,mmpnet,relenc-m")
    ap.add_argument("--prereg", default="SS154")
    ap.add_argument("--out", default=str(CONFIRM / "xling_route.json"))
    args = ap.parse_args()
    facts = load()
    encoders = {k: ENCODERS[k] for k in args.encoders.split(",")}
    if args.smoke:
        facts = facts[:40]
        encoders = {k: encoders[k] for k in ("relenc", "mmpnet")}
    global BOOT
    BOOT = 200 if args.smoke else BOOT
    negs = e3(facts)
    heldout = set(
        json.loads((CONFIRM / "relation_encoder_mpnet/ss76_meta.json").read_text())[
            "heldout_relations"
        ]
    )

    rows_by_enc = {}
    for name, path in encoders.items():
        scorer = BiScorer(path)
        tau_rel = RELATION_TAU if name == "relenc" else None
        router = RouterV7(facts, scorer, THETA, relation_tau=tau_rel)
        rows_by_enc[name] = read(router, facts, negs)
        print(name, "read", flush=True)
        del router, scorer
        gc.collect()
        torch.cuda.empty_cache()

    every = list(range(len(facts)))
    held = [k for k in every if facts[k]["relation"] in heldout]
    r = rows_by_enc["relenc"]

    def share(key, field):
        return round(float(np.mean([x[key][field] for x in r])), 4)

    report = {
        "prereg": args.prereg,
        "n_facts": len(facts),
        "n_heldout_facts": len(held),
        "found": {
            e: {k: int(sum(x[k]["score"] is not None for x in rows)) for k in rows[0]}
            for e, rows in rows_by_enc.items()
        },
        "auroc": boot(rows_by_enc, facts, every),
        "auroc_heldout": boot(rows_by_enc, facts, held) if len(held) > 1 else None,
        "relenc_operating_point": {
            "right_en_reph": share("en_reph", "right"),
            "right_tr_src": share("tr_src", "right"),
            "right_tr_reph": share("tr_reph", "right"),
            "e3_routed_en": share("e3_en", "routed"),
            "e3_routed_tr": share("e3_tr", "routed"),
        },
        "rows": {"e3_queries": negs, **rows_by_enc},
    }
    a = report["auroc"]
    report["rule"] = {
        x: {
            "i_tr": a["tr"][f"{x}_minus_relenc"]["ci95"][0] > 0,
            "ii_en": a["en"][f"{x}_minus_relenc"]["ci95"][0] >= -0.02,
        }
        for x in ("relenc-m", "relenc-b")
        if x in rows_by_enc
    }
    out = Path(args.out)
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    assert args.smoke or not out.exists(), f"refusing to overwrite {out}"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(
        json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1),
        flush=True,
    )


if __name__ == "__main__":
    main()
