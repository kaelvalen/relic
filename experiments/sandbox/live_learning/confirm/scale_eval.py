"""SS98: how many facts need an adapter as the base model grows.

For one base model and dataset: (1) serve each edit prompt and paraphrase of the N=1000
store with the note of the routed entry; routes are those of the default router, saved in
the SS76 evaluation, since the router does not depend on the base model. A fact needs an
adapter if its note does not answer its edit prompt. (2) Train the adopted adapter for
those facts and serve the queries routed to them with adapter and note.

One resident rank-2 model is reset per fact rather than reloaded, which a 4-bit base
would make slow. The Qwen2.5-1.5B bf16 arm reuses the SS94 adapters.

    .venv/bin/python experiments/sandbox/live_learning/confirm/scale_eval.py --arm q05
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from adapter_size import stored_bytes  # noqa: E402
from canary_eval import HELDOUT  # noqa: E402

from relic.adapters import (  # noqa: E402
    R2,
    fact_pairs,
    get_delta,
    int8_delta,
    make_variant,
    set_delta,
)
from relic.ledger import CANARIES, DeltaStore  # noqa: E402
from relic.metrics import ci, hit, sym_kl, word_hit  # noqa: E402
from relic.model import chat, load_model  # noqa: E402
from relic.serving import SYSTEM, rag_system, trim  # noqa: E402

SEED, RETRY_SEED = 98, 980
GATE = 2.0
KAPPA = 0.968  # SS69, measured on Qwen2.5-1.5B; reported as a reference on other models
CONFIRM = Path("results/live_learning/confirm")
OUT = CONFIRM / "scale"
FILES = {
    "counterfact": ("external_counterfact_n1000.json", "cf"),
    "zsre": ("external_zsre_n1000.json", "zsre"),
}
ARMS = {
    "q05": ("Qwen/Qwen2.5-0.5B-Instruct", False),
    "q15": ("Qwen/Qwen2.5-1.5B-Instruct", False),
    "q15q": ("Qwen/Qwen2.5-1.5B-Instruct", True),
    "q3q": ("Qwen/Qwen2.5-3B-Instruct", True),
    "q7q": ("Qwen/Qwen2.5-7B-Instruct", True),
    "smol3q": ("HuggingFaceTB/SmolLM3-3B", True),
}
# entry 117: the two layers 4-bit leaves in bf16 (2.2 GB) stay on CPU, so 7B fits a shared GPU
CPU_EDGES = {
    "model.embed_tokens": "cpu",
    "model.layers": 0,
    "model.norm": 0,
    "model.rotary_emb": 0,
    "lm_head": "cpu",
}
DEVICE_MAPS = {"q7q": CPU_EDGES}


class Arm:
    def __init__(self, name):
        self.name = name
        self.model_id, four_bit = ARMS[name]
        self.tok, base = load_model(
            self.model_id, load_in_4bit=four_bit, device_map=DEVICE_MAPS.get(name)
        )
        self.revision = getattr(base.config, "_commit_hash", None)
        self.m = make_variant(base, R2, SEED)
        self.m.eval()
        # _logits and _pair_step use only the tokenizer and the loss settings
        self.store = DeltaStore.__new__(DeltaStore)
        self.store.tok = self.tok
        self.store.lr, self.store.steps, self.store.kl_lambda = 3e-4, 16, 1.0
        self.store.kl_prompts = list(CANARIES)
        self.store._base_logits = None
        self.thinks = "smollm" in self.model_id.lower()
        with torch.no_grad(), self.m.disable_adapter():
            self.base_in = [self.store._logits(self.m, q) for q in CANARIES]
            self.base_held = [self.store._logits(self.m, q) for q in HELDOUT]

    def ask(self, q, system, adapter=False):
        if self.thinks:
            system = f"/no_think {system}"
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
        if adapter:
            out = chat(self.tok, self.m, msgs, max_new_tokens=24)
        else:
            with self.m.disable_adapter():
                out = chat(self.tok, self.m, msgs, max_new_tokens=24)
        return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()

    def train(self, pairs, seed):
        torch.manual_seed(seed)
        for module in self.m.modules():
            if hasattr(module, "reset_lora_parameters"):
                module.reset_lora_parameters("default", True)
        params = dict(self.m.named_parameters())
        a = [p for n, p in params.items() if "lora_A" in n]
        b = [p for n, p in params.items() if "lora_B" in n]
        opt = torch.optim.AdamW(
            [
                {"params": a, "lr": self.store.lr},
                {"params": b, "lr": 8 * self.store.lr},
            ]
        )
        self.m.eval()
        with torch.no_grad(), self.m.disable_adapter():
            self.store._base_logits = [self.store._logits(self.m, q) for q in CANARIES]
        self.m.train()
        for _ in range(self.store.steps):
            for user, assistant in pairs:
                self.store._pair_step(self.m, opt, user, assistant)
        self.m.eval()
        self.store._base_logits = None
        return int8_delta(get_delta(self.m))

    def divergence(self):
        with torch.no_grad():
            cur_in = [self.store._logits(self.m, q) for q in CANARIES]
            cur_held = [self.store._logits(self.m, q) for q in HELDOUT]
        return (
            float(np.mean([sym_kl(b, c) for b, c in zip(self.base_in, cur_in)])),
            float(np.mean([sym_kl(b, c) for b, c in zip(self.base_held, cur_held)])),
        )


def notes_phase(arm, dataset, facts, routes, path):
    done = json.loads(path.read_text()) if path.exists() else {}
    t0 = time.time()
    by_id = {f["id"]: i for i, f in enumerate(facts)}
    for n, f in enumerate(facts):
        if f["id"] in done:
            continue
        row = {"base": arm.ask(f["probe"], SYSTEM)}
        for kind in ("probe", "paraphrase"):
            route = routes[f["id"]][kind]
            row[f"note_{kind}"] = (
                arm.ask(f[kind], SYSTEM)
                if route is None
                else arm.ask(f[kind], rag_system("instruct", facts, by_id[route]))
            )
        done[f["id"]] = row
        if (n + 1) % 100 == 0:
            path.write_text(json.dumps(done, ensure_ascii=False))
            trim()
            print(
                f"{arm.name} {dataset} notes {n + 1}/{len(facts)} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    path.write_text(json.dumps(done, ensure_ascii=False))
    return done


def saved_adapters(tag):
    """SS94 rank-2 adapters of the 1.5B bf16 arm, int8, in the rank-2 layout."""
    return torch.load(
        CONFIRM / f"store_r2_{tag}_train.pt", map_location="cpu", weights_only=False
    )["deltas"]


def adapters_phase(arm, dataset, tag, facts, routes, notes, path):
    ckpt = path.with_suffix(".pt")
    state = (
        torch.load(ckpt, map_location="cpu", weights_only=False)
        if ckpt.exists()
        else {"deltas": {}, "rows": {}}
    )
    need = [f for f in facts if not word_hit(notes[f["id"]]["note_probe"], f["answer"])]
    reuse = saved_adapters(tag) if arm.name == "q15" else None
    t0 = time.time()
    for n, f in enumerate(need):
        fid = f["id"]
        if fid in state["rows"]:
            continue
        t = time.time()
        pairs = fact_pairs(f)
        delta = reuse[fid] if reuse is not None else arm.train(pairs, SEED)
        set_delta(arm.m, delta)
        d_in, d_held = arm.divergence()
        retrained = False
        if not d_in <= GATE and reuse is None:
            delta = arm.train(pairs, RETRY_SEED)
            set_delta(arm.m, delta)
            d_in, d_held = arm.divergence()
            retrained = True
        committed = d_in <= GATE
        state["rows"][fid] = {
            "d_in": d_in,
            "d_held": d_held,
            "retrained": retrained,
            "committed": committed,
            "seconds": round(time.time() - t, 2),
        }
        if committed:
            state["deltas"][fid] = delta
        if (n + 1) % 20 == 0:
            torch.save(state, ckpt)
            trim()
            print(
                f"{arm.name} {dataset} adapters {n + 1}/{len(need)} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    torch.save(state, ckpt)

    # serve the queries routed to an entry that holds an adapter
    served = state.setdefault("served", {})
    by_id = {f["id"]: i for i, f in enumerate(facts)}
    current = None
    for f in facts:
        for kind in ("probe", "paraphrase"):
            route = routes[f["id"]][kind]
            key = f"{f['id']}:{kind}"
            if route not in state["deltas"] or key in served:
                continue
            if route != current:
                set_delta(arm.m, state["deltas"][route])
                current = route
            note = rag_system("instruct", facts, by_id[route])
            served[key] = {
                "both": arm.ask(f[kind], note, adapter=True),
                "adapter": arm.ask(f[kind], SYSTEM, adapter=True),
            }
    torch.save(state, ckpt)
    return state


def metric(values):
    values = [bool(v) for v in values]
    return {
        "n": len(values),
        "rate": round(float(np.mean(values)), 4),
        "ci95": ci(values),
    }


def read(facts, routes, notes, state):
    need = [f for f in facts if not word_hit(notes[f["id"]]["note_probe"], f["answer"])]
    need_sub = [f for f in facts if not hit(notes[f["id"]]["note_probe"], f["answer"])]
    deltas, served = state["deltas"], state["served"]

    def tiered(f, kind, score):
        key = f"{f['id']}:{kind}"
        if routes[f["id"]][kind] in deltas and key in served:
            return score(served[key]["both"], f["answer"])
        return score(notes[f["id"]][f"note_{kind}"], f["answer"])

    rows = list(state["rows"].values())
    held = np.asarray([r["d_held"] for r in rows]) if rows else np.zeros(0)
    sizes = [stored_bytes(d, "int8") for d in deltas.values()]
    return {
        "n": len(facts),
        "needs_adapter": metric(
            [not word_hit(notes[f["id"]]["note_probe"], f["answer"]) for f in facts]
        ),
        "needs_adapter_substring": round(len(need_sub) / len(facts), 4),
        "base_word": metric(
            [word_hit(notes[f["id"]]["base"], f["answer"]) for f in facts]
        ),
        "note_probe_word": metric(
            [word_hit(notes[f["id"]]["note_probe"], f["answer"]) for f in facts]
        ),
        "note_paraphrase_word": metric(
            [word_hit(notes[f["id"]]["note_paraphrase"], f["answer"]) for f in facts]
        ),
        "tiered_probe_word": metric([tiered(f, "probe", word_hit) for f in facts]),
        "tiered_paraphrase_word": metric(
            [tiered(f, "paraphrase", word_hit) for f in facts]
        ),
        "tiered_probe_substring": metric([tiered(f, "probe", hit) for f in facts]),
        "tiered_paraphrase_substring": metric(
            [tiered(f, "paraphrase", hit) for f in facts]
        ),
        "adapters_trained": len(need),
        "adapters_committed": len(deltas),
        "retrained": sum(r["retrained"] for r in rows),
        "train_seconds_mean": (
            round(float(np.mean([r["seconds"] for r in rows])), 2) if rows else None
        ),
        "heldout_d_mean": round(float(held.mean()), 4) if rows else None,
        "heldout_d_p99": round(float(np.percentile(held, 99)), 4) if rows else None,
        "above_kappa_share": round(float((held > KAPPA).mean()), 4) if rows else None,
        "adapter_bytes": int(sizes[0]) if sizes else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument(
        "--smoke", action="store_true", help="3 facts per dataset, discarded"
    )
    args = ap.parse_args(argv)
    out = OUT / "smoke" if args.smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    arm = Arm(args.arm)
    report = {
        "prereg": "SS98",
        "arm": args.arm,
        "model": arm.model_id,
        "revision": arm.revision,
        "four_bit": ARMS[args.arm][1],
        "memory_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "datasets": {},
    }
    for dataset, (facts_file, tag) in FILES.items():
        facts = json.loads((CONFIRM / facts_file).read_text())["facts"]
        saved = json.loads((CONFIRM / f"{tag}_n1000_v8_eval.json").read_text())
        routes = {
            r["fact_id"]: {k: r[k]["route_fact_id"] for k in ("probe", "paraphrase")}
            for r in saved["examples"]["serving"]
        }
        if args.smoke:
            facts = facts[:3]
            kept = {f["id"] for f in facts}
            routes = {
                fid: {k: (r if r in kept else None) for k, r in by_kind.items()}
                for fid, by_kind in routes.items()
            }
        path = out / f"{args.arm}_{tag}"
        notes = notes_phase(
            arm, dataset, facts, routes, path.with_suffix(".notes.json")
        )
        if args.arm == "q15" and not args.smoke:
            same = sum(
                notes[r["fact_id"]][f"note_{k}"] == r[k]["responses"]["retrieval"]
                for r in saved["examples"]["serving"]
                for k in ("probe", "paraphrase")
            )
            report.setdefault("reproduced_saved_notes", {})[dataset] = same
            print(f"{dataset}: {same}/2000 notes equal the saved outputs", flush=True)
        state = adapters_phase(arm, dataset, tag, facts, routes, notes, path)
        report["datasets"][dataset] = read(facts, routes, notes, state)
        print(args.arm, dataset, json.dumps(report["datasets"][dataset]), flush=True)
    report["memory_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    (out / f"{args.arm}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1)
    )
    print("saved", out / f"{args.arm}.json", flush=True)


if __name__ == "__main__":
    main()
