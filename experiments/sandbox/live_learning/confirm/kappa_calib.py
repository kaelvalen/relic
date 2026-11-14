"""Kappa from the adapters the store writes (draft entry 160; CHANGES.md, item 8).

Entry 69 set kappa = 0.968 as the 99th percentile of the held-out divergence D of 100 rank-16
adapters of the first recipe, and tried 10 harmful adapters once each. The store now writes
rank-2 adapters in int8, trained on rewrites, under one system prompt, and retries a rejected
adapter on a ladder of up to seven rungs.

benign: N adapters of the first rung (A1) as Ledger.write trains them, on development facts;
kappa = the 99th percentile of their D on HELDOUT_CANARIES, with a bootstrap interval.
harmful: M adapters of the harmful recipe of entry 69 (learning rate 1e-3, 32 passes) pushed
through the whole ladder as Ledger.write would, every rung trained, so that the share accepted at
any rung can be read under the new kappa and under 0.968.

Facts: the development split (router_v4_calibrate.build_dev), checked disjoint by subject from
the main stores, the fresh set of E137 and every earlier draw.

.venv/bin/python experiments/sandbox/live_learning/confirm/kappa_calib.py --prereg SS160
.venv/bin/python experiments/sandbox/live_learning/confirm/kappa_calib.py --smoke
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

from adapter_only import CONFIRM, MODEL, draw_facts  # noqa: E402
from router_v4_calibrate import build_dev  # noqa: E402

from relic.adapters import (  # noqa: E402
    LADDER,
    fact_pairs,
    packed_nbytes,
    pad,
    train_adapter_packed,
    unpack_delta,
)
from relic.ledger import (  # noqa: E402
    HELDOUT_CANARIES,
    KAPPA,
    PROMPT_FIELDS,
    DeltaStore,
    Ledger,
)
from relic.metrics import word_hit  # noqa: E402
from relic.serving import trim  # noqa: E402

SEED = 160
N_BENIGN, N_HARMFUL = 300, 50
HARMFUL_LR, HARMFUL_STEPS = 1e-3, 32  # entry 69
BOOT = 10000
MAIN = ("external_counterfact_n1000.json", "external_zsre_n1000.json")


def main_subjects() -> set[str]:
    return {
        f["subject"]
        for name in MAIN
        for f in json.loads((CONFIRM / name).read_text())["facts"]
    }


def earlier_subjects() -> set[str]:
    """Subjects of the main stores and of every draw before this entry."""
    used = main_subjects()
    rank2 = json.loads((CONFIRM / "rank2_fresh_facts.json").read_text())["sets"]
    used |= {f["subject"] for fs in rank2.values() for f in fs}  # E87
    seen = {f["subject"] for fs in draw_facts(150)[0].values() for f in fs}  # E126
    seen |= {  # E136, the fresh set
        f["subject"]
        for fs in draw_facts(400, seed=136, exclude=seen)[0].values()
        for f in fs
    }
    # E149, E151 and E153 drew 1100 CounterFact facts with seed 149
    seen |= {f["subject"] for f in draw_facts(1100, seed=149, exclude=seen)[0]["cf"]}
    return used | seen


def dev_facts() -> list[dict]:
    raw_cf = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    raw_zs = json.loads((CONFIRM / "raw/zsre.json").read_text())
    dev = build_dev(raw_cf, raw_zs, main_subjects())
    overlap = {f["subject"] for f in dev} & earlier_subjects()
    assert not overlap, f"{len(overlap)} development subjects were drawn before"
    for i, f in enumerate(dev):
        f["id"] = f"dev{i:04d}"
    return dev


def p99_interval(values, seed=SEED, n=BOOT) -> list[float]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.percentile(rng.choice(arr, size=(n, arr.size), replace=True), 99, axis=1)
    return [round(float(np.percentile(boots, q)), 4) for q in (2.5, 97.5)]


def divergence(store, packed) -> float:
    return float(store._expert_kld(pad(unpack_delta(packed)), HELDOUT_CANARIES))


def benign(store, ledger, facts) -> list[dict]:
    name, seed, kl_lambda, rewrites = LADDER[0]
    assert name == "A1" and rewrites
    store.kl_lambda = kl_lambda
    rows = []
    for k, f in enumerate(facts, 1):
        pairs = fact_pairs(f) + [(r, f["answer"]) for r in ledger._rewrites(f)]
        packed = train_adapter_packed(store, pairs, seed)
        rows.append(
            {
                "fact_id": f["id"],
                "d": divergence(store, packed),
                "pairs": len(pairs),
                "bytes": packed_nbytes(packed),
            }
        )
        trim()
        print(f"benign {k}/{len(facts)}", flush=True)
    return rows


def harmful(store, ledger, facts, anchor: str) -> list[dict]:
    """Every rung of the ladder under the harmful learning rate and pass count."""
    lr, steps = store.lr, store.steps
    store.lr, store.steps = HARMFUL_LR, HARMFUL_STEPS
    rows = []
    try:
        for k, f in enumerate(facts, 1):
            short = fact_pairs(f)
            full = short + [(r, f["answer"]) for r in ledger._rewrites(f)]
            tried = {}
            for name, seed, kl_lambda, rewrites in LADDER:
                # "ladder": the anchor weight the rung uses in Ledger.write; "none": none
                store.kl_lambda = kl_lambda if anchor == "ladder" else 0.0
                packed = train_adapter_packed(store, full if rewrites else short, seed)
                tried[name] = divergence(store, packed)
                trim()
            rows.append({"fact_id": f["id"], "tried": tried})
            print(f"harmful {k}/{len(facts)}", flush=True)
    finally:
        store.lr, store.steps = lr, steps
    return rows


def base_readings(store, questions, canaries) -> dict:
    """The base model's answers: which questions it gets right, what it says to canaries."""
    full = store.deltas
    store.deltas = {}
    store.materialize()
    answers = [store.answer(q["question"]) for q in questions]
    out = {
        "correct": [word_hit(a, q["answer"]) for a, q in zip(answers, questions)],
        "canaries": [store.answer(c) for c in canaries],
    }
    store.deltas = full
    store.materialize()
    return out


def harm(store, packed, questions, canaries, base) -> dict:
    """With the adapter forced: retention of the base model's correct answers, and the
    share of canaries whose greedy answer changes."""
    full = store.deltas
    store.deltas = {"__harm__": pad(unpack_delta(packed))}
    store.materialize()
    try:
        kept = [
            word_hit(store.answer(q["question"]), q["answer"])
            for q, ok in zip(questions, base["correct"])
            if ok
        ]
        changed = [store.answer(c) != b for c, b in zip(canaries, base["canaries"])]
    finally:
        store.deltas = full
        store.materialize()
    return {
        "retention": round(sum(kept) / len(kept), 4) if kept else None,
        "canaries_changed": round(sum(changed) / len(changed), 4) if changed else None,
    }


def auroc(pos, neg) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def separation(rows, retention_drop, canary_change) -> dict:
    """AUROC of D for adapters whose harm is above the author's thresholds."""
    out = {}
    for name, threshold, score in (
        ("retention", retention_drop, lambda h: 1 - h["retention"]),
        ("canaries", canary_change, lambda h: h["canaries_changed"]),
    ):
        if threshold is None:
            out[name] = "TODO(author): threshold not set"
            continue
        bad = [r["d"] for r in rows if score(r["harm"]) > threshold]
        good = [r["d"] for r in rows if score(r["harm"]) <= threshold]
        out[name] = {
            "threshold": threshold,
            "harmful": len(bad),
            "auroc": auroc(bad, good),
        }
    return out


def accepted(rows, kappa) -> dict:
    """Share of harmful facts stored at some rung, and the rung that would store them."""
    first = [next((n for n, x in r["tried"].items() if x <= kappa), None) for r in rows]
    stored = [x for x in first if x is not None]
    return {
        "kappa": round(float(kappa), 4),
        "share": round(len(stored) / len(rows), 4) if rows else None,
        "by_rung": {n: stored.count(n) for n, *_ in LADDER if n in stored},
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prereg", help="the entry that pins this run")
    ap.add_argument("--benign", type=int, default=N_BENIGN)
    ap.add_argument("--harmful", type=int, default=N_HARMFUL)
    ap.add_argument("--harmful-anchor", choices=("ladder", "none"), default="ladder")
    ap.add_argument("--smoke", action="store_true", help="3 benign and 2 harmful")
    args = ap.parse_args(argv)
    if not (args.smoke or args.prereg):
        ap.error("--prereg is required for a full run")
    n_benign, n_harmful = (3, 2) if args.smoke else (args.benign, args.harmful)
    out = CONFIRM / ("kappa_calib_smoke.json" if args.smoke else "kappa_calib.json")
    if out.exists():
        ap.error(f"refusing to overwrite {out}")

    facts = dev_facts()
    random.Random(SEED).shuffle(facts)
    assert len(facts) >= n_benign + n_harmful
    benign_facts, harmful_facts = facts[:n_benign], facts[n_benign:][:n_harmful]

    # the store's default prompts: the system after CHANGES.md item 1
    store = DeltaStore(MODEL, steps=16, capacity=1)
    ledger = Ledger(store, None, 0.7501, 0.8985)  # for its rewrites only
    b = benign(store, ledger, benign_facts)
    ds = [r["d"] for r in b]
    kappa = float(np.percentile(ds, 99))
    h = harmful(store, ledger, harmful_facts, args.harmful_anchor)
    report = {
        "prereg": args.prereg or "smoke",
        "model": MODEL,
        "revision": store.revision,
        "prompts": {k: getattr(store, k) for k in PROMPT_FIELDS},
        "seed": SEED,
        "benign_n": len(b),
        "benign_d": {
            "mean": round(float(np.mean(ds)), 4),
            "median": round(float(np.median(ds)), 4),
            "p95": round(float(np.percentile(ds, 95)), 4),
            "max": round(float(np.max(ds)), 4),
        },
        "kappa_p99": round(kappa, 4),
        "kappa_p99_ci95": p99_interval(ds),
        "harmful_n": len(h),
        "harmful_anchor": args.harmful_anchor,
        "harmful_accepted_new_kappa": accepted(h, kappa),
        "harmful_accepted_old_kappa": accepted(h, KAPPA),
        "benign": b,
        "harmful": h,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    summary = {k: v for k, v in report.items() if k not in ("benign", "harmful")}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
