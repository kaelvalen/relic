"""SS96: write-time slots on the N=1000 stores (eval-only).

.venv/bin/python experiments/sandbox/live_learning/confirm/slot_eval.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from locality_eval import match_raw  # noqa: E402
from routed_eval import (  # noqa: E402
    MANIFEST_PATH,
    REPO_ROOT,
    _metric,
    _repo_path,
    load_verified_checkpoint,
    verify_manifest_inputs,
)
from router_v4_calibrate import _templates_by_relation, build_dev  # noqa: E402

from relic.ledger import DeltaStore, Ledger  # noqa: E402
from relic.metrics import hit  # noqa: E402
from relic.router import (  # noqa: E402
    find_subject,
    key_text,
    make_scorer,  # noqa: E402
    relation_text,
)
from relic.serving import Server, serve_arms, trim  # noqa: E402

SEED = 96
BOOT = 10000
CONFIRM = Path("results/live_learning/confirm")
FILES = {
    "counterfact": ("cf_n1000.ckpt", "external_counterfact_n1000.json", "cf"),
    "zsre": ("zsre_n1000.ckpt", "external_zsre_n1000.json", "zsre"),
}
THETA_ROUTE = 0.7501


def slot_threshold(scorer, dev):
    """The lowest threshold at which at most 1% of other-relation dev pairs reach it."""
    pos, neg = [], []
    for f in dev:
        key = key_text(f["probe"], f["subject"])
        span = find_subject(f["negative"], f["subject"])
        if span is not None:
            neg.append(scorer.sims(relation_text(f["negative"], span), [key])[0])
        for p in f["paraphrases"]:
            span = find_subject(p, f["subject"])
            if span is not None:
                pos.append(scorer.sims(relation_text(p, span), [key])[0])
    pos, neg = np.asarray(pos), np.asarray(neg)
    theta = float(np.quantile(neg, 0.99, method="higher"))
    return {
        "theta": round(theta, 4),
        "dev_fpr": round(float((neg >= theta).mean()), 4),
        "dev_tpr": round(float((pos >= theta).mean()), 4),
        "n_pos": int(pos.size),
        "n_neg": int(neg.size),
    }


def template_pairs(scorer, raw_cf, theta):
    """Slot decisions over CounterFact templates with the subject masked."""
    by_relation = {
        r: sorted(set(ts)) for r, ts in _templates_by_relation(raw_cf).items()
    }
    # one key per relation-template; the placeholder stands where the subject would be
    keys = {r: [t.replace("{}", "X") for t in ts] for r, ts in by_relation.items()}
    scorer.prefetch([k for ks in keys.values() for k in ks])
    across, within, worst = [], [], []
    for r1, r2 in itertools.combinations(sorted(keys), 2):
        best = 0.0
        for a in keys[r1]:
            sims = scorer.sims(a, keys[r2])
            across.extend(s >= theta for s in sims)
            best = max(best, *sims)
        worst.append((round(best, 4), r1, r2))
    for r in sorted(keys):
        for a, b in itertools.combinations(keys[r], 2):
            within.append(scorer.sims(a, [b])[0] >= theta)
    worst.sort(reverse=True)
    return {
        "across_share": round(float(np.mean(across)), 4),
        "across_n": len(across),
        "within_share": round(float(np.mean(within)), 4) if within else None,
        "within_n": len(within),
        "worst_relation_pairs": worst[:10],
    }


def lower_one_sided(values, seed=SEED):
    """Lower end of the one-sided 95% bootstrap interval of a mean."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(BOOT, arr.size), replace=True).mean(axis=1)
    return round(float(np.percentile(boots, 5)), 4)


def evaluate(dataset, store, scorer, theta_slot, raw_cf, duplicate_theta=None):
    ckpt, facts_file, tag = FILES[dataset]
    manifest = json.loads(MANIFEST_PATH.read_bytes())
    verified = verify_manifest_inputs(
        manifest,
        _repo_path(CONFIRM / ckpt, REPO_ROOT),
        _repo_path(CONFIRM / facts_file, REPO_ROOT),
        manifest["models"]["base"]["id"],
        repo_root=REPO_ROOT,
    )
    import torch

    facts = verified["facts"]
    checkpoint, _ = load_verified_checkpoint(
        verified["checkpoint_path"],
        verified["checkpoint_sha256"],
        lambda source: torch.load(source, map_location="cpu", weights_only=False),
    )
    saved = json.loads((CONFIRM / f"{tag}_n1000_v8_eval.json").read_text())
    saved_rows = {r["fact_id"]: r for r in saved["examples"]["serving"]}

    store.deltas, store.keys, store.notes = {}, {}, {}
    ledger = Ledger(store, scorer, THETA_ROUTE, theta_slot, duplicate_theta)
    holder, events = {}, []
    for f in facts:
        event = ledger.commit(f, checkpoint["deltas"][f["id"]])
        # an update drops the entry it replaces; its facts now live in the new one
        if event["action"] == "update":
            for k, v in holder.items():
                if v == event["superseded"]:
                    holder[k] = event["entry"]
        holder[f["id"]] = event["entry"]
        events.append({"fact_id": f["id"], **event})
    del checkpoint
    trim()
    actions = {
        a: sum(e["action"] == a for e in events)
        for a in ("insert", "duplicate", "update")
    }
    print(dataset, actions, f"entries {len(ledger.entries)}", flush=True)

    merged = []
    relation_of = {}
    if dataset == "counterfact":
        records = match_raw(facts, raw_cf)
        relation_of = {
            f["id"]: r["requested_rewrite"]["relation_id"]
            for f, r in zip(facts, records)
        }
    by_id = {f["id"]: f for f in facts}
    for f in facts:
        if holder[f["id"]] != f["id"]:
            e = by_id[holder[f["id"]]]
            merged.append(
                {
                    "fact_id": f["id"],
                    "entry": e["id"],
                    "probe": f["probe"],
                    "entry_probe": e["probe"],
                    "answer": f["answer"],
                    "entry_answer": e["answer"],
                    "relations": [relation_of.get(f["id"]), relation_of.get(e["id"])],
                }
            )
    across = sum(
        1
        for m in merged
        if m["relations"][0] is not None and m["relations"][0] != m["relations"][1]
    )

    # serving on the original questions, through the ledger's router
    router, active = ledger.router
    server = Server(store)
    rows, reserved = [], 0
    for f in facts:
        row = {"fact_id": f["id"], "entry": holder[f["id"]]}
        for kind in ("probe", "paraphrase"):
            j, _ = router.route(f[kind])
            route = None if j is None else active[j]["id"]
            old = saved_rows[f["id"]][kind]
            if route == old["route_fact_id"]:
                adapter, both = old["adapter_hit"], old["both_hit"]
            else:
                reserved += 1
                _, _, arms = serve_arms(server, router, active, f[kind], None)
                adapter = hit(arms["adapter"], f["answer"])
                both = hit(arms["both"], f["answer"])
            row[kind] = {
                "route": route,
                "adapter_hit": adapter,
                "both_hit": both,
                "old_adapter_hit": old["adapter_hit"],
                "old_both_hit": old["both_hit"],
            }
        rows.append(row)
        trim()
    print(dataset, f"re-served {reserved} queries", flush=True)

    def paired(kind, arm):
        new = np.asarray([r[kind][f"{arm}_hit"] for r in rows], dtype=float)
        old = np.asarray([r[kind][f"old_{arm}_hit"] for r in rows], dtype=float)
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(rows), size=(BOOT, len(rows)))
        boots = (new - old)[idx].mean(axis=1)
        return {
            "new": round(float(new.mean()), 4),
            "old": round(float(old.mean()), 4),
            "diff": round(float((new - old).mean()), 4),
            "lower_95_one_sided": round(float(np.percentile(boots, 5)), 4),
        }

    serving = {
        f"{kind}_{arm}": paired(kind, arm)
        for kind in ("probe", "paraphrase")
        for arm in ("adapter", "both")
    }

    # removal of the first 100 original facts, by the entry that holds each
    removed = facts[:100]
    removed_entries = {holder[f["id"]] for f in removed}
    pre = {f["id"]: server.base(f["probe"]) for f in removed}
    for fid in sorted(removed_entries):
        ledger.revoke(fid)
    router, active = ledger.router
    rem = []
    for f in removed:
        j, _ = router.route(f["probe"])
        resp = (
            pre[f["id"]] if j is None else server.adapter(active[j]["id"], f["probe"])
        )
        rem.append(
            {
                "fact_id": f["id"],
                "routed_to": None if j is None else active[j]["id"],
                "gone": not hit(resp, f["answer"]),
                "return": hit(resp, f["answer"]) == hit(pre[f["id"]], f["answer"]),
                "base_hit": hit(pre[f["id"]], f["answer"]),
                "response": resp,
            }
        )
    kept = [f for f in facts if holder[f["id"]] not in removed_entries]
    retention = []
    for f in kept:
        j, _ = router.route(f["probe"])
        resp = (
            server.base(f["probe"])
            if j is None
            else server.adapter(active[j]["id"], f["probe"])
        )
        retention.append(hit(resp, f["answer"]))
    trim()
    return {
        "actions": actions,
        "entries": len(ledger.entries) + len(removed_entries),
        "merged": merged,
        "merged_across_relation_ids": across,
        "requeried": reserved,
        "serving": serving,
        "removal": {
            "entries_removed": len(removed_entries),
            "gone": _metric([r["gone"] for r in rem]),
            "return": _metric([r["return"] for r in rem]),
            "return_lower_95_one_sided": lower_one_sided([r["return"] for r in rem]),
            "base_hit": _metric([r["base_hit"] for r in rem]),
            "retention": _metric(retention),
            "old": saved["removal"]["return"]["rate"],
            "examples": rem,
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CONFIRM / "slot_eval.json"))
    ap.add_argument("--duplicate-theta", type=float, default=None, help="SS104")
    ap.add_argument("--prereg", default="SS96")
    args = ap.parse_args(argv)
    if Path(args.out).exists():
        ap.error(f"refusing to overwrite {args.out}")
    raw_cf = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    raw_zs = json.loads((CONFIRM / "raw/zsre.json").read_text())
    test_subjects = {
        f["subject"]
        for _, facts_file, _ in FILES.values()
        for f in json.loads((CONFIRM / facts_file).read_text())["facts"]
    }
    dev = build_dev(raw_cf, raw_zs, test_subjects)
    scorer = make_scorer("relenc")
    threshold = slot_threshold(scorer, dev)
    print("slot threshold", threshold, flush=True)
    pairs = template_pairs(scorer, raw_cf, threshold["theta"])
    print("template pairs", pairs, flush=True)

    manifest = json.loads(MANIFEST_PATH.read_bytes())
    store = DeltaStore(manifest["models"]["base"]["id"], steps=16)
    report = {
        "prereg": args.prereg,
        "duplicate_theta": args.duplicate_theta,
        "theta_route": THETA_ROUTE,
        "slot_threshold": threshold,
        "template_pairs": pairs,
        "datasets": {
            d: evaluate(
                d, store, scorer, threshold["theta"], raw_cf, args.duplicate_theta
            )
            for d in FILES
        },
    }
    cf = report["datasets"]["counterfact"]
    zs = report["datasets"]["zsre"]
    report["rule"] = {
        "i_zsre_return": zs["removal"]["return_lower_95_one_sided"] >= 0.95,
        "ii_cf_cross_relation_merges": cf["merged_across_relation_ids"]
        <= (2 if args.duplicate_theta else 1),
        "iii_template_share": pairs["across_share"] <= 0.02,
        "iv_noninferior": all(
            d["serving"][k]["lower_95_one_sided"] >= -0.01
            for d in (cf, zs)
            for k in ("probe_both", "paraphrase_both")
        ),
    }
    report["adopted"] = all(report["rule"].values())
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report["rule"]), "adopted", report["adopted"], flush=True)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
