"""SS81: adapter size - storage precision (A, eval only) and update form (B).

A: the first 200 adapters of the CounterFact N=1000 store, rounded to bf16, fp16 or int8
and cast back to fp32 before serving; readings against the fp32 reference.
B: per-fact adapters on the SS44 set with smaller LoRA layouts (rank, targets, layers,
a shared frozen A), trained with the per-fact LoRA+ recipe.

    .venv/bin/python experiments/sandbox/live_learning/confirm/adapter_size.py precision
    .venv/bin/python experiments/sandbox/live_learning/confirm/adapter_size.py sweep
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

import torch  # noqa: E402
from canary_eval import HELDOUT  # noqa: E402
from harness import chat, load_model  # noqa: E402
from ledger import sym_kl  # noqa: E402
from pilot import CANARIES, SYSTEM, PilotStore, hit, rag_system, trim  # noqa: E402
from routed_eval import (  # noqa: E402
    MANIFEST_PATH,
    REPO_ROOT,
    _repo_path,
    load_verified_checkpoint,
    verify_manifest_inputs,
)

SEED = 81
KAPPA = 0.968  # SS69 held-out kappa (paper \chkKappa)
CONFIRM = Path("results/live_learning/confirm")
FORMATS = ["fp32", "bf16", "fp16", "int8"]
MID = list(range(8, 16))
VARIANTS = {
    "r16_qv_all": dict(r=16, targets=["q_proj", "v_proj"], layers=None, fa=False),
    "r4_qv_all": dict(r=4, targets=["q_proj", "v_proj"], layers=None, fa=False),
    "r2_qv_all": dict(r=2, targets=["q_proj", "v_proj"], layers=None, fa=False),
    "r1_qv_all": dict(r=1, targets=["q_proj", "v_proj"], layers=None, fa=False),
    "r4_v_all": dict(r=4, targets=["v_proj"], layers=None, fa=False),
    "r1_v_all": dict(r=1, targets=["v_proj"], layers=None, fa=False),
    "r4_qv_mid": dict(r=4, targets=["q_proj", "v_proj"], layers=MID, fa=False),
    "r1_qv_mid": dict(r=1, targets=["q_proj", "v_proj"], layers=MID, fa=False),
    "fa_r4_qv_all": dict(r=4, targets=["q_proj", "v_proj"], layers=None, fa=True),
    "fa_r1_qv_all": dict(r=1, targets=["q_proj", "v_proj"], layers=None, fa=True),
}


# -- A: storage precision ---------------------------------------------------
def _int8_rows(t: torch.Tensor) -> torch.Tensor:
    """Symmetric int8 with one fp16 absmax scale per row, dequantized to fp32."""
    scale = (t.abs().amax(dim=1, keepdim=True) / 127.0).half().float()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (t / scale).round().clamp(-127, 127) * scale


def round_to(t: torch.Tensor, fmt: str) -> torch.Tensor:
    t = t.float()
    if fmt == "fp32":
        return t
    if fmt == "bf16":
        return t.bfloat16().float()
    if fmt == "fp16":
        return t.half().float()
    if fmt == "int8":
        return _int8_rows(t)
    raise ValueError(fmt)


def stored_bytes(delta: dict, fmt: str, b_only: bool = False) -> int:
    width = {"fp32": 4, "bf16": 2, "fp16": 2, "int8": 1}[fmt]
    total = 0
    for ab in delta.values():
        for k in ("B",) if b_only else ("A", "B"):
            t = ab[k]
            total += t.numel() * width
            if fmt == "int8":
                total += t.shape[0] * 2
    return total


def logits_on(store, model, prompts):
    with torch.no_grad():
        return [store._logits(model, q) for q in prompts]


def mean_d(base, cur) -> float:
    return float(np.mean([sym_kl(b, a) for b, a in zip(base, cur)]))


def precision(args) -> None:
    manifest = json.loads(MANIFEST_PATH.read_bytes())
    model_id = manifest["models"]["base"]["id"]
    verified = verify_manifest_inputs(
        manifest,
        _repo_path(args.ckpt, REPO_ROOT),
        _repo_path(args.facts, REPO_ROOT),
        model_id,
        repo_root=REPO_ROOT,
    )
    facts = verified["facts"][: args.n]
    checkpoint, _ = load_verified_checkpoint(
        verified["checkpoint_path"],
        verified["checkpoint_sha256"],
        lambda source: torch.load(source, map_location="cpu", weights_only=False),
    )
    deltas = {f["id"]: checkpoint["deltas"][f["id"]] for f in facts}
    del checkpoint
    trim()

    torch.manual_seed(SEED)
    store = PilotStore(model_id, steps=16)
    slot = next(p for n, p in store.model.named_parameters() if "lora_A" in n)
    store.deltas = {}
    store.materialize()
    base_held = logits_on(store, store.model, HELDOUT)

    rows = []
    for i, f in enumerate(facts):
        row = {"fact_id": f["id"]}
        for fmt in FORMATS:
            d = {
                n: {"A": round_to(ab["A"], fmt), "B": round_to(ab["B"], fmt)}
                for n, ab in deltas[f["id"]].items()
            }
            store.deltas = {f["id"]: d}
            store.materialize()
            note = rag_system("instruct", facts, i)
            out = {
                "probe": store.answer(f["probe"], system=SYSTEM),
                "paraphrase": store.answer(f["paraphrase"], system=SYSTEM),
                "paraphrase_note": store.answer(f["paraphrase"], system=note),
            }
            row[fmt] = {
                "outputs": out,
                "efficacy": hit(out["probe"], f["answer"]),
                "paraphrase": hit(out["paraphrase"], f["answer"]),
                "paraphrase_note": hit(out["paraphrase_note"], f["answer"]),
                "d_heldout": mean_d(base_held, logits_on(store, store.model, HELDOUT)),
            }
        store.deltas = {}
        store.materialize()
        rows.append(row)
        trim()
        if (i + 1) % 20 == 0:
            print(f"precision {i + 1}/{len(facts)}", flush=True)

    ref = {r["fact_id"]: r["fp32"] for r in rows}
    summary = {}
    example = deltas[facts[0]["id"]]
    for fmt in FORMATS:
        same = [
            r[fmt]["outputs"][k] == ref[r["fact_id"]]["outputs"][k]
            for r in rows
            for k in ("probe", "paraphrase", "paraphrase_note")
        ]
        dd = [abs(r[fmt]["d_heldout"] - ref[r["fact_id"]]["d_heldout"]) for r in rows]
        dh = [r[fmt]["d_heldout"] for r in rows]
        summary[fmt] = {
            "bytes_per_fact": stored_bytes(example, fmt),
            "identical_outputs": round(float(np.mean(same)), 4),
            "efficacy": round(float(np.mean([r[fmt]["efficacy"] for r in rows])), 4),
            "paraphrase": round(
                float(np.mean([r[fmt]["paraphrase"] for r in rows])), 4
            ),
            "paraphrase_note": round(
                float(np.mean([r[fmt]["paraphrase_note"] for r in rows])), 4
            ),
            "d_heldout_mean": round(float(np.mean(dh)), 4),
            "d_heldout_max": round(float(np.max(dh)), 4),
            "above_kappa": int(sum(x > KAPPA for x in dh)),
            "abs_d_change_max": round(float(np.max(dd)), 5),
        }
        print(fmt, json.dumps(summary[fmt]), flush=True)
    report = {
        "prereg": "SS81-A",
        "ckpt": args.ckpt,
        "n": len(facts),
        "slot_dtype": str(slot.dtype),
        "kappa": KAPPA,
        "summary": summary,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("saved:", args.out)


# -- B: update form ---------------------------------------------------------
def make_variant(base, spec, seed=SEED):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    kw = dict(
        r=spec["r"],
        lora_alpha=2 * spec["r"],
        lora_dropout=0.0,
        target_modules=spec["targets"],
        task_type="CAUSAL_LM",
    )
    if spec["layers"] is not None:
        kw["layers_to_transform"] = spec["layers"]
    m = get_peft_model(base, LoraConfig(**kw))
    if spec["fa"]:
        for n, p in m.named_parameters():
            if "lora_A" in n:
                p.requires_grad_(False)
    return m


def get_delta(m) -> dict:
    params = dict(m.named_parameters())
    return {
        n: {
            "A": p.detach().float().cpu().clone(),
            "B": params[n.replace("lora_A", "lora_B")].detach().float().cpu().clone(),
        }
        for n, p in params.items()
        if "lora_A" in n
    }


def set_delta(m, delta=None) -> None:
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_A" in n:
                key, k = n, "A"
            elif "lora_B" in n:
                key, k = n.replace("lora_B", "lora_A"), "B"
            else:
                continue
            if delta is None:
                p.zero_()
            else:
                p.copy_(delta[key][k].to(p.device, p.dtype))


def train_one(store, spec, pairs, b_mult=8, seed=SEED):
    """b_mult: B learning rate as a multiple of A's (8 = LoRA+, 1 = baseline)."""
    base = load_model(store.model_name)[1]
    m = make_variant(base, spec, seed)
    a = [p for n, p in m.named_parameters() if "lora_A" in n and p.requires_grad]
    b = [p for n, p in m.named_parameters() if "lora_B" in n and p.requires_grad]
    groups = [{"params": b, "lr": b_mult * store.lr}]
    if a:
        groups.insert(0, {"params": a, "lr": store.lr})
    opt = torch.optim.AdamW(groups)
    m.eval()
    with torch.no_grad(), m.disable_adapter():
        store._base_logits = [store._logits(m, q) for q in store.kl_prompts]
    m.train()
    for _ in range(store.steps):
        for user, assistant in pairs:
            store._pair_step(m, opt, user, assistant)
    store._base_logits = None
    delta = get_delta(m)
    del m, base
    torch.cuda.empty_cache()
    return delta


def eval_variant(store, spec, deltas, facts) -> dict:
    base = load_model(store.model_name)[1]
    m = make_variant(base, spec)
    m.eval()
    with torch.no_grad(), m.disable_adapter():
        base_held = logits_on(store, m, HELDOUT)
        base_in = logits_on(store, m, CANARIES)
    rows = []
    for i, f in enumerate(facts):
        set_delta(m, deltas[f["id"]])

        def ask(q, system, model=m):
            return chat(
                store.tok,
                model,
                [{"role": "system", "content": system}, {"role": "user", "content": q}],
                max_new_tokens=24,
            )

        rows.append(
            {
                "fact_id": f["id"],
                "efficacy": hit(ask(f["probe"], SYSTEM), f["answer"]),
                "paraphrase": hit(ask(f["paraphrase"], SYSTEM), f["answer"]),
                "paraphrase_note": hit(
                    ask(f["paraphrase"], rag_system("instruct", facts, i)), f["answer"]
                ),
                "d_heldout": mean_d(base_held, logits_on(store, m, HELDOUT)),
                "d_in_sample": mean_d(base_in, logits_on(store, m, CANARIES)),
            }
        )
        trim()
    del m, base
    torch.cuda.empty_cache()
    dh = [r["d_heldout"] for r in rows]
    di = [r["d_in_sample"] for r in rows]
    return {
        "efficacy": round(float(np.mean([r["efficacy"] for r in rows])), 3),
        "paraphrase": round(float(np.mean([r["paraphrase"] for r in rows])), 3),
        "paraphrase_note": round(
            float(np.mean([r["paraphrase_note"] for r in rows])), 3
        ),
        "d_heldout_mean": round(float(np.mean(dh)), 4),
        "d_heldout_max": round(float(np.max(dh)), 4),
        "above_kappa": round(float(np.mean([x > KAPPA for x in dh])), 3),
        "d_in_sample_mean": round(float(np.mean(di)), 4),
        "d_in_sample_max": round(float(np.max(di)), 4),
        "rows": rows,
    }


def shared_a_identical(deltas) -> bool:
    first = next(iter(deltas.values()))
    return all(
        torch.equal(d[n]["A"], first[n]["A"]) for d in deltas.values() for n in first
    )


def sweep(args) -> None:
    data = json.loads(Path(args.facts).read_text())
    facts = data["facts"] if isinstance(data, dict) else data
    store = PilotStore(args.model, steps=16)
    store.kl_prompts = list(CANARIES)
    store.model.to("cpu")  # only the tokenizer and the loss helpers are used
    torch.cuda.empty_cache()
    out = Path(args.out)
    report = json.loads(out.read_text()) if out.exists() else {"variants": {}}
    report.update({"prereg": "SS81-B", "facts": args.facts, "kappa": KAPPA})
    for name in args.variants.split(","):
        if name in report["variants"]:
            print(name, "done, skipped", flush=True)
            continue
        spec = VARIANTS[name]
        t0 = time.time()
        deltas = {}
        for f in facts:
            deltas[f["id"]] = train_one(
                store,
                spec,
                [(f["teach"], f"Not aldım: {f['teach']}"), (f["probe"], f["answer"])],
            )
            trim()
        train_s = time.time() - t0
        res = {"spec": spec, "train_seconds": round(train_s, 1)}
        if spec["fa"]:
            res["shared_a_identical"] = shared_a_identical(deltas)
        example = next(iter(deltas.values()))
        res["bytes_per_fact_bf16"] = stored_bytes(example, "bf16", b_only=spec["fa"])
        res["bytes_per_fact_fp32"] = stored_bytes(example, "fp32", b_only=spec["fa"])
        res.update(eval_variant(store, spec, deltas, facts))
        report["variants"][name] = res
        print(
            name,
            json.dumps({k: v for k, v in res.items() if k != "rows"}),
            flush=True,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("saved:", out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("precision")
    a.add_argument("--ckpt", default=str(CONFIRM / "cf_n1000.ckpt"))
    a.add_argument("--facts", default=str(CONFIRM / "external_counterfact_n1000.json"))
    a.add_argument("--n", type=int, default=200)
    a.add_argument("--out", default=str(CONFIRM / "adapter_size_precision.json"))
    b = sub.add_parser("sweep")
    b.add_argument("--facts", default=str(CONFIRM / "external_counterfact_n50.json"))
    b.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    b.add_argument("--variants", default=",".join(VARIANTS))
    b.add_argument("--out", default=str(CONFIRM / "adapter_size_sweep.json"))
    args = ap.parse_args(argv)
    if args.cmd == "precision":
        if Path(args.out).exists():
            ap.error(f"refusing to overwrite {args.out}")
        precision(args)
    else:
        sweep(args)


if __name__ == "__main__":
    main()
