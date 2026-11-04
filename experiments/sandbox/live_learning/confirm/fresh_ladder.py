"""SS136: a store with no note and a retry ladder, on fresh facts, served.

.venv/bin/python experiments/sandbox/live_learning/confirm/fresh_ladder.py [--smoke]
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

import torch  # noqa: E402
from adapter_only import CONFIRM, MODEL, draw_facts, write_rewrites  # noqa: E402
from adapter_size import KAPPA  # noqa: E402
from rank2_confirm import evaluate  # noqa: E402
from retry_dose import ARMS, ROUND2, train_arm  # noqa: E402
from served_adapter import (  # noqa: E402
    build_queries,
    compare,
    decide,
    matches_forced,
    serve,
    write_questions,
)

from relic.adapters import R2  # noqa: E402
from relic.ledger import CANARIES, DeltaStore  # noqa: E402
from relic.metrics import word_hit  # noqa: E402

SEED = 136
N_PER = 400
RUNGS = {
    "R1": ROUND2["R1"],
    "L2": ROUND2["L2"],
    "L4": ARMS["L4"],
    "L8": ROUND2["L8"],
    "L16": ARMS["L16"],
}
BASE_SPEC = (16, 1.0, False, 126)
STORES = ("current", "a0_alone", "fallback", "ladder", "gate_off")


def store_arms(d0: float, d1: float, first_rung: str | None) -> dict:
    """{store: the adapter it holds for a fact}; above kappa a write commits nothing."""
    a0 = "A0" if d0 <= KAPPA else None
    a1 = "A1" if d1 <= KAPPA else None
    return {
        "current": a0,
        "a0_alone": a0,
        "fallback": a1 or a0,
        "ladder": a1 or first_rung or a0,
        "gate_off": "A1",
    }


def train_resumable(store, spec, items, rewrites, path, arm="A1") -> dict:
    """Saves every 25 facts, as the GPU is shared and this runs for hours."""
    done = (
        torch.load(path, map_location="cpu", weights_only=False)
        if path.exists()
        else {}
    )
    todo = [it for it in items if it[0] not in done]
    for s in range(0, len(todo), 25):
        done.update(train_arm(store, spec, todo[s : s + 25], rewrites, arm=arm))
        tmp = path.with_suffix(".tmp")
        torch.save(done, tmp)
        tmp.replace(path)
        print(f"{path.stem}: {len(done)}/{len(items)}", flush=True)
    return done


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="4 facts and 4 questions")
    args = ap.parse_args(argv)
    n_per = 2 if args.smoke else N_PER
    stem = "fresh_ladder_smoke" if args.smoke else "fresh_ladder"
    out = CONFIRM / f"{stem}.json"
    report = json.loads(out.read_text()) if out.exists() else {"rows": {}, "rungs": {}}
    report["prereg"] = "SS136"

    def save():
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1))

    seen, _, _ = draw_facts(150)
    seen = {f["subject"] for fs in seen.values() for f in fs}
    sets, cf_records, raw_cf = draw_facts(n_per, seed=SEED, exclude=seen)
    assert not {f["subject"] for fs in sets.values() for f in fs} & seen
    facts = {f["id"]: f for fs in sets.values() for f in fs}
    key = {f["id"]: f"{ds}:{f['id']}" for ds, fs in sets.items() for f in fs}
    items = {
        f"{ds}:{f['id']}": (f"{ds}:{f['id']}", f, fs, i)
        for ds, fs in sets.items()
        for i, f in enumerate(fs)
    }
    natural = write_questions(sets["cf"], CONFIRM / f"{stem}_questions.json")
    queries = build_queries(sets, facts, cf_records, raw_cf, natural)

    store = DeltaStore(MODEL, steps=16)
    store.kl_prompts = list(CANARIES)
    if "rewrites" not in report:
        report["rewrites"] = {}
        for fs in sets.values():
            report["rewrites"].update(write_rewrites(store.tok, store.model, fs))
        save()
    rewrites = report["rewrites"]
    store.model.to("cpu")  # only the tokenizer and the loss helpers are used
    torch.cuda.empty_cache()

    damage = json.loads((CONFIRM / "canary_eval.json").read_text())["damage"]
    damage = damage[:4] if args.smoke else damage
    answers = [d["answer"] for d in damage]

    deltas = {}
    for arm in ("A0", "A1"):
        path = CONFIRM / f"{stem}_{arm}.pt"
        deltas[arm] = train_resumable(
            store, BASE_SPEC, list(items.values()), rewrites, path, arm
        )
        if arm not in report["rows"]:
            res = evaluate(
                MODEL, store.tok, R2, deltas[arm], list(items.values()), [], []
            )
            report["rows"][arm] = res["rows"]
            save()

    # as at write time: a rung is trained only while every attempt so far is above kappa
    first = {}
    left = [k for k in items if report["rows"]["A1"][k]["d_heldout"] > KAPPA]
    report["rejected_by_A1"] = list(left)
    for rung, spec in RUNGS.items():
        if not left:
            break
        path = CONFIRM / f"{stem}_{rung}.pt"
        batch = [items[k] for k in left]
        deltas[rung] = train_resumable(store, spec, batch, rewrites, path)
        if rung not in report["rungs"]:
            res = evaluate(MODEL, store.tok, R2, deltas[rung], batch, [], [])
            report["rungs"][rung] = res["rows"]
            save()
        rows = report["rungs"][rung]
        first.update({k: rung for k in left if rows[k]["d_heldout"] <= KAPPA})
        left = [k for k in left if k not in first]
    report["first_rung"], report["no_rung"] = first, left

    arms = {
        i: store_arms(
            report["rows"]["A0"][key[i]]["d_heldout"],
            report["rows"]["A1"][key[i]]["d_heldout"],
            first.get(key[i]),
        )
        for i in facts
    }
    rows, sizes, n_base, n_adapter = serve(queries, facts, key, arms, deltas, STORES)
    gc.collect()
    torch.cuda.empty_cache()
    summary = {s: compare(rows, s) for s in STORES}
    report.update(
        theta=0.7501,
        kappa=KAPPA,
        store_size=sizes,
        generated={"base": n_base, "adapter": n_adapter},
        summary=summary,
        rule=decide(summary, "ladder"),
        rule_fallback=decide(summary, "fallback"),
        served=rows,
        store_share={
            arm: round(sum(a["ladder"] == arm for a in arms.values()) / len(arms), 4)
            for arm in ("A1", *RUNGS, "A0")
        },
        matches_forced=matches_forced(rows, report["rows"]["A1"], key),
    )
    save()

    held = {key[i]: a["ladder"] for i, a in arms.items() if a["ladder"]}
    ladder = {k: deltas[a][k] for k, a in held.items()}
    rng = random.Random(SEED)
    draw = [rng.choice(sorted(held)) for _ in damage]
    report["wrong_route"] = {}
    for name, d in (("A0", deltas["A0"]), ("A1", deltas["A1"]), ("ladder", ladder)):
        wrong = evaluate(MODEL, store.tok, R2, d, [], damage, draw)["wrong_route"]
        report["wrong_route"][name] = round(
            float(
                np.mean([word_hit(w["response"], a) for w, a in zip(wrong, answers)])
            ),
            4,
        )
    save()
    shown = (
        "edit",
        "paraphrase",
        "paraphrase_cf",
        "paraphrase_zs",
        "e3",
        "natural",
        "routed_other_fact",
    )
    for s in STORES:
        print(s, json.dumps({k: summary[s][k] for k in shown}), flush=True)
    print("minus current", json.dumps(summary["ladder"]["minus_current"]), flush=True)
    print(
        "rule",
        json.dumps(report["rule"]),
        "fallback",
        json.dumps(report["rule_fallback"]),
        flush=True,
    )
    print(
        "rejected",
        len(report["rejected_by_A1"]),
        "share",
        report["store_share"],
        flush=True,
    )
    print("wrong route", report["wrong_route"], "sizes", sizes, flush=True)
    print("matches", report["matches_forced"], flush=True)
    assert report["matches_forced"] >= 0.99, "served and forced outputs differ"


if __name__ == "__main__":
    main()
