"""SS132: what brings a rejected A1 adapter under kappa, on SS126 facts (development).

.venv/bin/python experiments/sandbox/live_learning/confirm/retry_dose.py [--smoke]
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

import torch  # noqa: E402
from adapter_only import (  # noqa: E402
    CONFIRM,
    MODEL,
    arm_pairs,
    draw_facts,
    e3_readings,
)
from adapter_size import KAPPA  # noqa: E402
from locality_eval import other_relation_prompts  # noqa: E402
from rank2_confirm import evaluate, int8  # noqa: E402

from relic.adapters import R2, train_one  # noqa: E402
from relic.ledger import CANARIES, DeltaStore  # noqa: E402
from relic.metrics import word_hit  # noqa: E402
from relic.serving import SYSTEM, trim  # noqa: E402

SEED = 132
N_OTHERS = 31
ARMS = {  # name: (passes, kl_lambda, train under the serving system prompt)
    "D8": (8, 1.0, False),
    "D4": (4, 1.0, False),
    "L4": (16, 4.0, False),
    "L16": (16, 16.0, False),
    "S": (16, 1.0, True),
}


def train_arm(store, spec, items, rewrites):
    passes, kl_lambda, system = spec
    store.steps, store.kl_lambda = passes, kl_lambda
    apply = store.tok.apply_chat_template
    if system:  # the pair step adds no system message; the anchors bring their own

        def with_system(msgs, **kw):
            head = [] if msgs[0]["role"] == "system" else [SYS_MSG]
            return apply([*head, *msgs], **kw)

        store.tok.apply_chat_template = with_system
    try:
        trained = {}
        for key, f, _, _ in items:
            pairs = arm_pairs("A1", f, rewrites[f["id"]])
            trained[key] = train_one(store, R2, pairs, b_mult=8, seed=126)
            trim()
    finally:
        store.tok.apply_chat_template = apply
    return int8(trained)


SYS_MSG = {"role": "system", "content": SYSTEM}


def group_summary(rows, e3, keys, answers) -> dict:
    d = [rows[k]["d_heldout"] for k in keys]
    words = {
        r: [word_hit(rows[k]["outputs"][r], answers[k]) for k in keys]
        for r in ("probe", "paraphrase")
    }
    cf = [k for k in keys if k.startswith("cf:")]
    return {
        "n": len(keys),
        "below_kappa": int(sum(x <= KAPPA for x in d)),
        "d_mean": round(float(np.mean(d)), 4),
        "edit": round(float(np.mean(words["probe"])), 4),
        "paraphrase": round(float(np.mean(words["paraphrase"])), 4),
        "paraphrase_and_below": round(
            float(np.mean([w and x <= KAPPA for w, x in zip(words["paraphrase"], d)])),
            4,
        ),
        "e3": round(float(np.mean([e3[k]["leak"] for k in cf])), 4) if cf else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 rejected and 2 other facts")
    args = ap.parse_args(argv)
    stem = "retry_dose_smoke" if args.smoke else "retry_dose"
    out = CONFIRM / f"{stem}.json"
    report = json.loads(out.read_text()) if out.exists() else {"arms": {}}
    report["prereg"] = "SS132"
    saved = json.loads((CONFIRM / "adapter_only.json").read_text())
    sets, cf_records, raw_cf = draw_facts(150)
    e3 = {
        f["id"]: p["query"]
        for f, p in zip(sets["cf"], other_relation_prompts(cf_records, raw_cf))
    }
    all_items = {
        f"{ds}:{f['id']}": (f"{ds}:{f['id']}", f, facts, i)
        for ds, facts in sets.items()
        for i, f in enumerate(facts)
    }
    a1 = saved["arms"]["A1"]
    rejected = sorted(k for k in all_items if a1["rows"][k]["d_heldout"] > KAPPA)
    others = sorted(k for k in all_items if k not in rejected)
    others = random.Random(SEED).sample(others, N_OTHERS)
    if args.smoke:
        rejected, others = rejected[:2], others[:2]
    keys = rejected + others
    report["groups"] = {"rejected": rejected, "others": others}
    items = [all_items[k] for k in keys]
    answers = {k: all_items[k][1]["answer"] for k in keys}
    groups = {"rejected": rejected, "others": others}
    report["reference_A1"] = {
        g: group_summary(a1["rows"], a1["e3"], ks, answers) for g, ks in groups.items()
    }

    damage = json.loads((CONFIRM / "canary_eval.json").read_text())["damage"]
    damage = damage[:4] if args.smoke else damage
    rng = random.Random(SEED)
    draw = [rng.choice(keys) for _ in damage]
    report["damage_answers"] = [d["answer"] for d in damage]
    store = DeltaStore(MODEL, steps=16)
    store.kl_prompts = list(CANARIES)
    store.model.to("cpu")  # only the tokenizer and the loss helpers are used
    torch.cuda.empty_cache()

    for name, spec in ARMS.items():
        if name in report["arms"]:
            continue
        deltas = train_arm(store, spec, items, saved["rewrites"])
        res = evaluate(MODEL, store.tok, R2, deltas, items, damage, draw)
        cf_items = [it for it in items if it[0].startswith("cf:")]
        res["e3"] = e3_readings(store.tok, deltas, cf_items, e3)
        res["summary"] = {
            g: group_summary(res["rows"], res["e3"], ks, answers)
            for g, ks in groups.items()
        }
        res["wrong_route_word"] = round(
            float(
                np.mean(
                    [
                        word_hit(w["response"], a)
                        for w, a in zip(res["wrong_route"], report["damage_answers"])
                    ]
                )
            ),
            4,
        )
        report["arms"][name] = res
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
        print(name, json.dumps(res["summary"]), res["wrong_route_word"], flush=True)
    print("reference A1", json.dumps(report["reference_A1"]), flush=True)


if __name__ == "__main__":
    main()
