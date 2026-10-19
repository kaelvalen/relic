"""SS87: rank-2 LoRA+ in int8 against the stored recipe on fresh adapter facts.

2000 facts (1000 CounterFact, 1000 zsRE) outside the N=1000 stores and the SS66 dev
split; the facts whose note fails on the edit prompt (SS68 tier rule) get a rank-16
baseline adapter (ref16, the stored recipe) and a rank-2 LoRA+ adapter evaluated in int8.

    .venv/bin/python experiments/sandbox/live_learning/confirm/rank2_fresh.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

import torch  # noqa: E402
from adapter_size import KAPPA, train_one  # noqa: E402
from external import adapt_counterfact, adapt_zsre  # noqa: E402
from facts import pairs as fact_pairs  # noqa: E402
from harness import chat, load_model  # noqa: E402
from pilot import CANARIES, PilotStore, hit, rag_system, trim  # noqa: E402
from rank2_confirm import R2, R16, READS, evaluate, int8, summarize  # noqa: E402
from router_v4_calibrate import build_dev  # noqa: E402

SEED = 87
N_PER = 1000
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CONFIRM = Path("results/live_learning/confirm")
ARMS = {
    "ref16": (R16, 1),
    "r2_lp": (R2, 8),
}  # arm -> (layout, B learning-rate multiple)


def draw_facts():
    raw_cf = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    raw_zs = json.loads((CONFIRM / "raw/zsre.json").read_text())
    test = {
        f["subject"]
        for name in ("external_counterfact_n1000.json", "external_zsre_n1000.json")
        for f in json.loads((CONFIRM / name).read_text())["facts"]
    }
    excluded = test | {f["subject"] for f in build_dev(raw_cf, raw_zs, test)}
    rng = random.Random(SEED)
    out = {}
    for ds, raw, adapt in (
        ("cf", raw_cf, adapt_counterfact),
        ("zs", raw_zs, adapt_zsre),
    ):
        pool = [
            f for f in (adapt(r) for r in raw) if f and f["subject"] not in excluded
        ]
        rng.shuffle(pool)
        facts, seen = [], set()
        for f in pool:
            if len(facts) == N_PER:
                break
            if f["subject"] in seen:
                continue
            seen.add(f["subject"])
            f["id"] = f"{ds}{len(facts):04d}"
            facts.append(f)
        out[ds] = facts
    return out


def tier_check(sets):
    tok, model = load_model(MODEL)
    tiers = {}
    for ds, facts in sets.items():
        for i, f in enumerate(facts):
            resp = chat(
                tok,
                model,
                [
                    {"role": "system", "content": rag_system("instruct", facts, i)},
                    {"role": "user", "content": f["probe"]},
                ],
                max_new_tokens=64,
            )
            tiers[f"{ds}:{f['id']}"] = "note" if hit(resp, f["answer"]) else "adapter"
        print(f"tier check {ds}: done", flush=True)
    del model
    torch.cuda.empty_cache()
    return tiers


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CONFIRM / "rank2_fresh.json"))
    args = ap.parse_args(argv)
    out_path = Path(args.out)
    facts_path = out_path.with_name(f"{out_path.stem}_facts.json")
    if facts_path.exists():
        saved = json.loads(facts_path.read_text())
        sets, tiers = saved["sets"], saved["tiers"]
    else:
        sets = draw_facts()
        tiers = tier_check(sets)
        facts_path.write_text(
            json.dumps(
                {"prereg": "SS87", "sets": sets, "tiers": tiers},
                ensure_ascii=False,
                indent=1,
            )
        )
    items = [
        (f"{ds}:{f['id']}", f, facts, i)
        for ds, facts in sets.items()
        for i, f in enumerate(facts)
        if tiers[f"{ds}:{f['id']}"] == "adapter"
    ]
    keys = [k for k, *_ in items]
    n_all = sum(len(v) for v in sets.values())
    print(f"adapter facts: {len(items)} of {n_all}", flush=True)

    report = json.loads(out_path.read_text()) if out_path.exists() else {"arms": {}}
    report.update(
        {"prereg": "SS87", "kappa": KAPPA, "n_drawn": n_all, "n_adapter": len(items)}
    )
    damage = json.loads((CONFIRM / "canary_eval.json").read_text())["damage"]
    rng = random.Random(SEED)
    draw = [rng.choice(keys) for _ in damage]

    store = PilotStore(MODEL, steps=16)
    store.kl_prompts = list(CANARIES)
    store.model.to("cpu")  # only the tokenizer and the loss helpers are used
    torch.cuda.empty_cache()
    deltas = {}
    for arm, (spec, b_mult) in ARMS.items():
        path = out_path.with_name(f"{out_path.stem}_{arm}.pt")
        if path.exists():
            deltas[arm] = torch.load(path, map_location="cpu", weights_only=False)
            continue
        t0, d = time.time(), {}
        for n, (key, f, _, _) in enumerate(items):
            d[key] = train_one(store, spec, fact_pairs(f), b_mult=b_mult, seed=SEED)
            trim()
            if (n + 1) % 50 == 0:
                print(
                    f"{arm}: trained {n + 1}/{len(items)} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
        torch.save(d, path)
        deltas[arm] = d
        report.setdefault("train_seconds", {})[arm] = round(time.time() - t0, 1)
    evals = {"ref16": (R16, deltas["ref16"]), "r2_lp_int8": (R2, int8(deltas["r2_lp"]))}
    for arm, (spec, d) in evals.items():
        if arm in report["arms"]:
            continue
        print("evaluating", arm, flush=True)
        report["arms"][arm] = evaluate(MODEL, store.tok, spec, d, items, damage, draw)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))

    ref = report["arms"]["ref16"]
    subsets = {
        "pooled": keys,
        "cf": [k for k in keys if k.startswith("cf:")],
        "zs": [k for k in keys if k.startswith("zs:")],
    }
    summary = {}
    for arm, res in report["arms"].items():
        summary[arm] = {
            name: summarize(res, ks, None if arm == "ref16" else ref)
            for name, ks in subsets.items()
        }
        summary[arm]["wrong_route_retained"] = round(
            float(np.mean([w["retained"] for w in res["wrong_route"]])), 4
        )
    r, c = summary["ref16"]["pooled"], summary["r2_lp_int8"]["pooled"]
    para_ci = c["minus_ref"]["paraphrase_note"]["ci95"]
    rule = {
        "i_efficacy_note": c["probe_note"] >= r["probe_note"] - 0.02,
        "i_paraphrase_note": c["paraphrase_note"] >= r["paraphrase_note"] - 0.03,
        "i_paraphrase_note_ci": para_ci[0] >= -0.06,
        "ii_above_kappa": c["above_kappa"] <= r["above_kappa"] + 0.02,
        "ii_d_mean": c["d_heldout_mean"] <= r["d_heldout_mean"] + 0.05,
        "iii_wrong_route": summary["r2_lp_int8"]["wrong_route_retained"]
        >= summary["ref16"]["wrong_route_retained"] - 0.03,
    }
    rule["passes"] = all(rule.values())
    report.update(
        {
            "summary": summary,
            "rule": rule,
            "P2_paraphrase_alone_gap": c["paraphrase"] - r["paraphrase"],
            "P3_adapter_share": len(items) / n_all,
            "draw": draw,
        }
    )
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    for arm in summary:
        p = summary[arm]["pooled"]
        print(
            arm,
            json.dumps(
                {k: p[k] for k in ("n", *READS, "d_heldout_mean", "above_kappa")}
            ),
            "wrong-route",
            summary[arm]["wrong_route_retained"],
            flush=True,
        )
    print("rule", json.dumps(rule), flush=True)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
