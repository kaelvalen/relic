"""SS126: the adapter alone, trained on phrasings the base model writes at write time.

.venv/bin/python experiments/sandbox/live_learning/confirm/adapter_only.py [--smoke]
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
from adapter_size import KAPPA  # noqa: E402
from external import adapt_counterfact, adapt_zsre  # noqa: E402
from gen_keys import SYSTEM as REWRITE_SYSTEM  # noqa: E402
from gen_keys import USER, parse  # noqa: E402
from locality_eval import other_relation_prompts  # noqa: E402
from rank2_confirm import evaluate, int8  # noqa: E402
from router_v4_calibrate import build_dev  # noqa: E402
from wrong_route import paired  # noqa: E402

from relic.adapters import (  # noqa: E402
    R2,
    fact_pairs,
    make_variant,
    set_delta,
    train_one,
)
from relic.ledger import CANARIES, LEGACY_PROMPTS, DeltaStore, use_prompts  # noqa: E402
from relic.metrics import word_hit  # noqa: E402
from relic.model import chat, load_model  # noqa: E402
from relic.serving import SYSTEM, trim  # noqa: E402

SEED = 126
N_PER = 150
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CONFIRM = Path("results/live_learning/confirm")
ARMS = ("A0", "A1", "A2")
READS = ("probe", "paraphrase", "paraphrase_note")


def draw_facts(n_per, seed=SEED, exclude=frozenset()):
    """Facts no earlier decision saw, with the CounterFact records for the E3 prompts."""
    raw_cf = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    raw_zs = json.loads((CONFIRM / "raw/zsre.json").read_text())
    used = {
        f["subject"]
        for name in ("external_counterfact_n1000.json", "external_zsre_n1000.json")
        for f in json.loads((CONFIRM / name).read_text())["facts"]
    }
    used |= {f["subject"] for f in build_dev(raw_cf, raw_zs, used)}
    fresh = json.loads((CONFIRM / "rank2_fresh_facts.json").read_text())["sets"]
    used |= {f["subject"] for facts in fresh.values() for f in facts}
    used |= set(exclude)
    rng = random.Random(seed)
    sets, records = {}, {}
    for ds, raw, adapt in (
        ("cf", raw_cf, adapt_counterfact),
        ("zs", raw_zs, adapt_zsre),
    ):
        pool = [(adapt(r), r) for r in raw]
        pool = [(f, r) for f, r in pool if f and f["subject"] not in used]
        rng.shuffle(pool)
        facts, kept, seen = [], [], set()
        for f, r in pool:
            if len(facts) == n_per:
                break
            if f["subject"] in seen or not f.get("paraphrase"):
                continue
            seen.add(f["subject"])
            f["id"] = f"{ds}{len(facts):04d}"
            facts.append(f)
            kept.append(r)
        sets[ds], records[ds] = facts, kept
    assert not {f["subject"] for fs in sets.values() for f in fs} & used
    return sets, records["cf"], raw_cf


def write_rewrites(tok, model, facts) -> dict:
    """Only the subject and the edit prompt go in, as when a fact is written."""
    out = {}
    for f in facts:
        text = chat(
            tok,
            model,
            [
                {"role": "system", "content": REWRITE_SYSTEM},
                {
                    "role": "user",
                    "content": USER.format(subject=f["subject"], probe=f["probe"]),
                },
            ],
            max_new_tokens=96,
        )
        out[f["id"]] = parse(text, f["probe"], f["subject"])
    return out


def arm_pairs(arm: str, f: dict, rewrites: list[str]) -> list[tuple[str, str]]:
    extra = [(r, f["answer"]) for r in rewrites]
    if arm == "A0":
        return fact_pairs(f)
    if arm == "A1":
        return fact_pairs(f) + extra
    return [(f["probe"], f["answer"])] + extra


def e3_readings(tok, deltas, cf_items, e3) -> dict:
    base = load_model(MODEL)[1]
    m = make_variant(base, R2)
    m.eval()

    def ask(q, model=m):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
        return chat(tok, model, msgs, max_new_tokens=64)

    out = {}
    for key, f, _, _ in cf_items:
        q = e3[f["id"]]
        with m.disable_adapter():
            before = ask(q)
        set_delta(m, deltas[key])
        after = ask(q)
        out[key] = {
            "leak": word_hit(after, f["answer"]) and not word_hit(before, f["answer"]),
            "base": before,
            "response": after,
        }
        trim()
    del m, base
    torch.cuda.empty_cache()
    return out


def word_rows(res, items) -> dict:
    rows = {}
    for key, f, _, _ in items:
        outputs = res["rows"][key]["outputs"]
        rows[key] = {r: word_hit(outputs[r], f["answer"]) for r in READS}
    return rows


def lower_upper(d):
    return d["lower_95_one_sided"], 2 * d["diff"] - d["lower_95_one_sided"]


def summarize(report, items):
    keys = {
        "pooled": [k for k, *_ in items],
        "cf": [k for k, *_ in items if k.startswith("cf:")],
        "zs": [k for k, *_ in items if k.startswith("zs:")],
    }
    words = {arm: word_rows(report["arms"][arm], items) for arm in ARMS}
    summary = {}
    for arm in ARMS:
        res = report["arms"][arm]
        s = {}
        for name, ks in keys.items():
            d = [res["rows"][k]["d_heldout"] for k in ks]
            s[name] = {
                **{
                    r: round(float(np.mean([words[arm][k][r] for k in ks])), 4)
                    for r in READS
                },
                "d_heldout_mean": round(float(np.mean(d)), 4),
                "above_kappa": round(float(np.mean([x > KAPPA for x in d])), 4),
            }
            s[name]["alone_minus_A0_note"] = paired(
                [words[arm][k]["paraphrase"] for k in ks],
                [words["A0"][k]["paraphrase_note"] for k in ks],
            )
            if arm != "A0":
                s[name]["minus_A0"] = {
                    r: paired(
                        [words[arm][k][r] for k in ks], [words["A0"][k][r] for k in ks]
                    )
                    for r in ("probe", "paraphrase")
                }
        cf = keys["cf"]
        leak = [res["e3"][k]["leak"] for k in cf]
        s["e3_leak"] = round(float(np.mean(leak)), 4)
        if arm != "A0":
            s["e3_minus_A0"] = paired(
                leak, [report["arms"]["A0"]["e3"][k]["leak"] for k in cf]
            )
        s["wrong_route_retained"] = round(
            float(np.mean([w["retained"] for w in res["wrong_route"]])), 4
        )
        s["wrong_route_word"] = round(
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
        summary[arm] = s
    return summary


def decide(summary):
    a0, rule = summary["A0"], {}
    for arm in ("A1", "A2"):
        s, p = summary[arm], summary[arm]["pooled"]
        para = p["minus_A0"]["paraphrase"]
        _, e3_upper = lower_upper(s["e3_minus_A0"])
        rule[arm] = {
            "i_paraphrase": para["diff"] >= 0.05 and para["ci95"][0] > 0,
            "ii_edit": p["minus_A0"]["probe"]["diff"] >= -0.02,
            "iii_above_kappa": p["above_kappa"] <= a0["pooled"]["above_kappa"] + 0.02,
            "iii_d_mean": p["d_heldout_mean"] <= a0["pooled"]["d_heldout_mean"] + 0.05,
            "iv_e3": e3_upper <= 0.02,
            "v_wrong_route": s["wrong_route_word"] >= a0["wrong_route_word"] - 0.03,
        }
    passing = [a for a in rule if all(rule[a].values())]
    adopted = max(
        passing, key=lambda a: summary[a]["pooled"]["paraphrase"], default=None
    )
    best = summary[adopted or "A0"]
    note_free = {
        name: lower_upper(best[name]["alone_minus_A0_note"])[0] >= -0.03
        for name in ("pooled", "cf", "zs")
    }
    note_free["holds"] = note_free["pooled"] and note_free["cf"]
    return {"arms": rule, "adopted": adopted, "note_free": note_free}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="4 facts and 4 questions")
    args = ap.parse_args(argv)
    n_per = 2 if args.smoke else N_PER
    stem = "adapter_only_smoke" if args.smoke else "adapter_only"
    out = CONFIRM / f"{stem}.json"
    report = json.loads(out.read_text()) if out.exists() else {"arms": {}}
    report["prereg"] = "SS126"

    sets, cf_records, raw_cf = draw_facts(n_per)
    e3 = {
        f["id"]: p["query"]
        for f, p in zip(sets["cf"], other_relation_prompts(cf_records, raw_cf))
    }
    items = [
        (f"{ds}:{f['id']}", f, facts, i)
        for ds, facts in sets.items()
        for i, f in enumerate(facts)
    ]
    keys = [k for k, *_ in items]
    damage = json.loads((CONFIRM / "canary_eval.json").read_text())["damage"]
    if args.smoke:
        damage = damage[:4]
    rng = random.Random(SEED)
    draw = [rng.choice(keys) for _ in damage]
    report["damage_answers"] = [d["answer"] for d in damage]

    store = DeltaStore(MODEL, steps=16)
    use_prompts(store, LEGACY_PROMPTS)
    store.kl_prompts = list(CANARIES)
    if "rewrites" not in report:
        report["rewrites"] = {}
        for facts in sets.values():
            report["rewrites"].update(write_rewrites(store.tok, store.model, facts))
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    counts = [len(report["rewrites"][f["id"]]) for _, f, _, _ in items]
    report["rewrites_per_fact"] = {str(n): counts.count(n) for n in sorted(set(counts))}
    print("rewrites per fact", report["rewrites_per_fact"], flush=True)
    store.model.to("cpu")  # only the tokenizer and the loss helpers are used
    torch.cuda.empty_cache()

    for arm in ARMS:
        if arm in report["arms"]:
            continue
        path = CONFIRM / f"{stem}_{arm}.pt"
        if path.exists():
            deltas = torch.load(path, map_location="cpu", weights_only=False)
        else:
            t0, trained = time.time(), {}
            for n, (key, f, _, _) in enumerate(items):
                pairs = arm_pairs(arm, f, report["rewrites"][f["id"]])
                trained[key] = train_one(store, R2, pairs, b_mult=8, seed=SEED)
                trim()
                if (n + 1) % 25 == 0:
                    print(
                        f"{arm}: trained {n + 1}/{len(items)} ({time.time() - t0:.0f}s)",
                        flush=True,
                    )
            deltas = int8(trained)
            torch.save(deltas, path)
            report.setdefault("train_seconds", {})[arm] = round(time.time() - t0, 1)
        res = evaluate(MODEL, store.tok, R2, deltas, items, damage, draw)
        cf_items = [it for it in items if it[0].startswith("cf:")]
        res["e3"] = e3_readings(store.tok, deltas, cf_items, e3)
        report["arms"][arm] = res
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
        print("evaluated", arm, flush=True)

    report["summary"] = summarize(report, items)
    report["rule"] = decide(report["summary"])
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    for arm in ARMS:
        print(arm, json.dumps(report["summary"][arm]["pooled"])[:400], flush=True)
    print(json.dumps(report["rule"], indent=1), flush=True)


if __name__ == "__main__":
    main()
