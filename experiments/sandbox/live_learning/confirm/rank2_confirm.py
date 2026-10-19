"""SS84: rank-2 adapters on the adapter facts of the N=1000 stores.

The 368 facts that carry an adapter in the tiered store (SS68 tiers: 162 CounterFact,
206 zsRE) are served with the stored rank-16 adapters (ref), rank-2 adapters trained with
the baseline recipe (r2_base) and with LoRA+ (r2_lp), and the rank-2 adapters stored in
int8. Readings against ref; wrong-route retention on the SS69 zsRE dev questions.

    .venv/bin/python experiments/sandbox/live_learning/confirm/rank2_confirm.py
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
from adapter_size import (  # noqa: E402
    KAPPA,
    make_variant,
    mean_d,
    round_to,
    set_delta,
    train_one,
)
from canary_eval import HELDOUT  # noqa: E402
from facts import pairs as fact_pairs  # noqa: E402
from harness import chat, load_model  # noqa: E402
from locality_eval import paired_diff_ci  # noqa: E402
from pilot import CANARIES, SYSTEM, PilotStore, hit, rag_system, trim  # noqa: E402
from routed_eval import (  # noqa: E402
    MANIFEST_PATH,
    REPO_ROOT,
    _repo_path,
    load_verified_checkpoint,
    verify_manifest_inputs,
)

SEED = 84
BOOT = 10000
CONFIRM = Path("results/live_learning/confirm")
DATASETS = {
    "cf": ("cf_n1000.ckpt", "external_counterfact_n1000.json", "cf_n1000_v4_eval.json"),
    "zs": ("zsre_n1000.ckpt", "external_zsre_n1000.json", "zsre_n1000_v4_eval.json"),
}
QV = ["q_proj", "v_proj"]
R16 = dict(r=16, targets=QV, layers=None, fa=False)
R2 = dict(r=2, targets=QV, layers=None, fa=False)
TRAINED = {"r2_base": 1, "r2_lp": 8}  # arm -> B learning-rate multiple


def load_tier_facts():
    """[(key, fact, facts of its dataset, index)] and the stored deltas of these facts."""
    manifest = json.loads(MANIFEST_PATH.read_bytes())
    model_id = manifest["models"]["base"]["id"]
    items, ref = [], {}
    for ds, (ckpt, facts_file, v4_file) in DATASETS.items():
        verified = verify_manifest_inputs(
            manifest,
            _repo_path(str(CONFIRM / ckpt), REPO_ROOT),
            _repo_path(str(CONFIRM / facts_file), REPO_ROOT),
            model_id,
            repo_root=REPO_ROOT,
        )
        facts = verified["facts"]
        rows = json.loads((CONFIRM / v4_file).read_text())["examples"]["serving"]
        tiers = {r["fact_id"]: r["tier"] for r in rows}
        checkpoint, _ = load_verified_checkpoint(
            verified["checkpoint_path"],
            verified["checkpoint_sha256"],
            lambda source: torch.load(source, map_location="cpu", weights_only=False),
        )
        for i, f in enumerate(facts):
            if tiers[f["id"]] == "adapter":
                key = f"{ds}:{f['id']}"
                items.append((key, f, facts, i))
                ref[key] = checkpoint["deltas"][f["id"]]
        del checkpoint
        trim()
    return model_id, items, ref


def int8(deltas):
    return {
        k: {
            n: {"A": round_to(ab["A"], "int8"), "B": round_to(ab["B"], "int8")}
            for n, ab in d.items()
        }
        for k, d in deltas.items()
    }


def evaluate(model_id, tok, spec, deltas, items, damage, draw):
    base = load_model(model_id)[1]
    m = make_variant(base, spec)
    m.eval()

    def ask(q, system, model=m):
        return chat(
            tok,
            model,
            [{"role": "system", "content": system}, {"role": "user", "content": q}],
            max_new_tokens=64,
        )

    with torch.no_grad(), m.disable_adapter():
        base_held = [store_logits(tok, m, q) for q in HELDOUT]
    rows = {}
    for n, (key, f, facts, i) in enumerate(items):
        set_delta(m, deltas[key])
        note = rag_system("instruct", facts, i)
        out = {
            "probe": ask(f["probe"], SYSTEM),
            "probe_note": ask(f["probe"], note),
            "paraphrase": ask(f["paraphrase"], SYSTEM),
            "paraphrase_note": ask(f["paraphrase"], note),
        }
        with torch.no_grad():
            cur = [store_logits(tok, m, q) for q in HELDOUT]
        rows[key] = {
            "outputs": out,
            **{k: hit(v, f["answer"]) for k, v in out.items()},
            "d_heldout": mean_d(base_held, cur),
        }
        trim()
        if (n + 1) % 50 == 0:
            print(f"  eval {n + 1}/{len(items)}", flush=True)
    wrong = []
    for d, key in zip(damage, draw):
        set_delta(m, deltas[key])
        resp = ask(d["question"], SYSTEM)
        wrong.append(
            {
                "question": d["question"],
                "adapter": key,
                "retained": hit(resp, d["answer"]),
                "response": resp,
            }
        )
    del m, base
    torch.cuda.empty_cache()
    return {"rows": rows, "wrong_route": wrong}


def store_logits(tok, model, q):
    """DeltaStore._logits without a store (the same system prompt and chat template)."""
    ids = tok.apply_chat_template(
        [
            {"role": "system", "content": "Kısa ve net cevap ver."},
            {"role": "user", "content": q},
        ],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(model.device)
    return model(input_ids=ids).logits[0, -1].float()


READS = ("probe", "probe_note", "paraphrase", "paraphrase_note")


def summarize(res, keys, ref=None):
    rows = res["rows"]
    dh = [rows[k]["d_heldout"] for k in keys]
    out = {
        "n": len(keys),
        **{r: round(float(np.mean([rows[k][r] for k in keys])), 4) for r in READS},
        "d_heldout_mean": round(float(np.mean(dh)), 4),
        "d_heldout_max": round(float(np.max(dh)), 4),
        "above_kappa": round(float(np.mean([x > KAPPA for x in dh])), 4),
    }
    if ref is not None:
        out["minus_ref"] = {
            r: paired_diff_ci(
                [rows[k][r] for k in keys], [ref["rows"][k][r] for k in keys], n=BOOT
            )
            for r in READS
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CONFIRM / "rank2_confirm.json"))
    ap.add_argument("--limit", type=int, default=None, help="smoke: first n facts")
    args = ap.parse_args(argv)
    out_path = Path(args.out)
    report = json.loads(out_path.read_text()) if out_path.exists() else {"arms": {}}
    report.update({"prereg": "SS84", "kappa": KAPPA})

    model_id, items, ref_deltas = load_tier_facts()
    if args.limit:
        items = items[: args.limit // 2] + items[-(args.limit - args.limit // 2) :]
    keys = [k for k, *_ in items]
    print(f"adapter facts: {len(items)}", flush=True)
    damage = json.loads((CONFIRM / "canary_eval.json").read_text())["damage"]
    if args.limit:
        damage = damage[: args.limit]
    rng = random.Random(SEED)
    draw = [rng.choice(keys) for _ in damage]

    store = PilotStore(model_id, steps=16)
    store.kl_prompts = list(CANARIES)
    store.model.to("cpu")  # only the tokenizer and the loss helpers are used
    torch.cuda.empty_cache()

    arms = {"ref": (R16, ref_deltas)}
    for arm, b_mult in TRAINED.items():
        path = out_path.with_name(f"{out_path.stem}_{arm}.pt")
        if path.exists():
            deltas = torch.load(path, map_location="cpu", weights_only=False)
        else:
            t0, deltas = time.time(), {}
            for n, (key, f, _, _) in enumerate(items):
                deltas[key] = train_one(
                    store, R2, fact_pairs(f), b_mult=b_mult, seed=SEED
                )
                trim()
                if (n + 1) % 50 == 0:
                    print(
                        f"{arm}: trained {n + 1}/{len(items)} ({time.time() - t0:.0f}s)",
                        flush=True,
                    )
            torch.save(deltas, path)
            report.setdefault("train_seconds", {})[arm] = round(time.time() - t0, 1)
        arms[arm] = (R2, deltas)
        arms[f"{arm}_int8"] = (R2, int8(deltas))

    for arm, (spec, deltas) in arms.items():
        if arm in report["arms"]:
            print(arm, "done, skipped", flush=True)
            continue
        print("evaluating", arm, flush=True)
        report["arms"][arm] = evaluate(
            model_id, store.tok, spec, deltas, items, damage, draw
        )
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))

    ref = report["arms"]["ref"]
    subsets = {
        "pooled": keys,
        "cf": [k for k in keys if k.startswith("cf:")],
        "zs": [k for k in keys if k.startswith("zs:")],
    }
    summary = {}
    for arm, res in report["arms"].items():
        summary[arm] = {
            name: summarize(res, ks, None if arm == "ref" else ref)
            for name, ks in subsets.items()
        }
        kept = [w["retained"] for w in res["wrong_route"]]
        summary[arm]["wrong_route_retained"] = round(float(np.mean(kept)), 4)
        if arm.endswith("_int8"):
            fp = report["arms"][arm.removesuffix("_int8")]["rows"]
            same = [
                res["rows"][k]["outputs"][r] == fp[k]["outputs"][r]
                for k in keys
                for r in READS
            ]
            summary[arm]["identical_to_fp32"] = round(float(np.mean(same)), 4)
        print(
            arm,
            json.dumps(
                {k: v for k, v in summary[arm]["pooled"].items() if k != "minus_ref"}
            ),
            "wrong-route",
            summary[arm]["wrong_route_retained"],
            flush=True,
        )
    report["summary"] = summary
    report["draw"] = draw
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("saved:", out_path)


if __name__ == "__main__":
    main()
