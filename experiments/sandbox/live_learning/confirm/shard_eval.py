"""SS112: shard adapters with exact removal, a pilot on 96 CounterFact facts.

.venv/bin/python experiments/sandbox/live_learning/confirm/shard_eval.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from adapter_size import stored_bytes  # noqa: E402
from canary_eval import HELDOUT  # noqa: E402
from locality_eval import match_raw, other_relation_prompts  # noqa: E402
from wrong_route import paired  # noqa: E402

from relic.adapters import (  # noqa: E402
    R2,
    fact_pairs,
    get_delta,
    int8_delta,
    make_variant,
    set_delta,
)
from relic.ledger import CANARIES, DeltaStore  # noqa: E402
from relic.metrics import hit, sym_kl  # noqa: E402
from relic.model import chat, load_model  # noqa: E402
from relic.serving import SYSTEM, rag_system, trim  # noqa: E402

SEED = 112
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
KAPPA = 0.968
PER_RELATION, RELATIONS = 32, 3
ARMS = [
    (8, 2, "relation"),
    (8, 4, "relation"),
    (8, 16, "relation"),
    (16, 4, "relation"),
    (16, 16, "relation"),
    (32, 4, "relation"),
    (32, 16, "relation"),
    (16, 16, "random"),
]
REMOVAL = {(8, 16, "relation"), (32, 16, "relation")}
CONFIRM = Path("results/live_learning/confirm")


class Bench:
    """One resident model per rank; `_logits` and `_pair_step` need only the tokenizer."""

    def __init__(self):
        self.rank = None
        self.m = None
        self.store = DeltaStore.__new__(DeltaStore)
        self.store.lr, self.store.steps, self.store.kl_lambda = 3e-4, 16, 1.0
        self.store.kl_prompts = list(CANARIES)
        self.store._base_logits = None

    def use_rank(self, r):
        if r == self.rank:
            return
        self.m = None
        torch.cuda.empty_cache()
        self.store.tok, base = load_model(MODEL)
        spec = {**R2, "r": r}
        self.m = make_variant(base, spec, SEED)
        self.m.eval()
        self.rank = r
        with torch.no_grad(), self.m.disable_adapter():
            self.base_in = [self.store._logits(self.m, q) for q in CANARIES]
            self.base_held = [self.store._logits(self.m, q) for q in HELDOUT]

    def ask(self, q, system, adapter=True):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
        if adapter:
            return chat(self.store.tok, self.m, msgs, max_new_tokens=24)
        with self.m.disable_adapter():
            return chat(self.store.tok, self.m, msgs, max_new_tokens=24)

    def train(self, facts, seed=SEED):
        torch.manual_seed(seed)
        for module in self.m.modules():
            if hasattr(module, "reset_lora_parameters"):
                module.reset_lora_parameters("default", True)
        params = [p for n, p in self.m.named_parameters() if "lora_" in n]
        opt = torch.optim.AdamW(params, lr=self.store.lr)
        pairs = [p for f in facts for p in fact_pairs(f)]
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

    def heldout_d(self):
        with torch.no_grad():
            cur = [self.store._logits(self.m, q) for q in HELDOUT]
        return float(np.mean([sym_kl(b, c) for b, c in zip(self.base_held, cur)]))


def draw(facts, raw_cf):
    """32 facts from each of the three most frequent relations, relation of each fact."""
    records = match_raw(facts, raw_cf)
    relation = {
        f["id"]: r["requested_rewrite"]["relation_id"] for f, r in zip(facts, records)
    }
    e3 = {
        f["id"]: p["query"]
        for f, p in zip(facts, other_relation_prompts(records, raw_cf))
    }
    top = [r for r, _ in Counter(relation.values()).most_common(RELATIONS)]
    rng = random.Random(SEED)
    chosen = []
    for r in top:
        pool = [f for f in facts if relation[f["id"]] == r]
        chosen.extend(rng.sample(pool, PER_RELATION))
    return chosen, relation, e3, top


def shards_of(chosen, k, grouping):
    order = list(chosen)  # already grouped by relation
    if grouping == "random":
        random.Random(SEED).shuffle(order)
    return [order[i : i + k] for i in range(0, len(order), k)]


def serve_shard(bench, shard, facts, index, e3, base_e3):
    rows = []
    answers = [f["answer"] for f in shard]
    d_held = bench.heldout_d()
    for f in shard:
        note = rag_system("instruct", facts, index[f["id"]])
        probe = bench.ask(f["probe"], SYSTEM)
        probe_note = bench.ask(f["probe"], note)
        para_note = bench.ask(f["paraphrase"], note)
        e3_alone = bench.ask(e3[f["id"]], SYSTEM)
        e3_note = bench.ask(e3[f["id"]], note)
        base = base_e3[f["id"]]
        others = [a for a in answers if a.lower() != f["answer"].lower()]
        rows.append(
            {
                "fact_id": f["id"],
                "probe": hit(probe, f["answer"]),
                "probe_note": hit(probe_note, f["answer"]),
                "paraphrase_note": hit(para_note, f["answer"]),
                "cross_talk": not hit(probe, f["answer"])
                and any(hit(probe, a) for a in others),
                "e3_leak": any(hit(e3_alone, a) and not hit(base, a) for a in answers),
                "e3_note_leak": any(
                    hit(e3_note, a) and not hit(base, a) for a in answers
                ),
                "d_heldout": d_held,
            }
        )
    trim()
    return rows


def summarize(rows, ref=None):
    keys = [
        "probe",
        "probe_note",
        "paraphrase_note",
        "cross_talk",
        "e3_leak",
        "e3_note_leak",
    ]
    out = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}
    d = [r["d_heldout"] for r in rows]
    out["d_heldout_mean"] = round(float(np.mean(d)), 4)
    out["above_kappa"] = round(float(np.mean([x > KAPPA for x in d])), 4)
    if ref is not None:
        by_id = {r["fact_id"]: r for r in rows}
        for k in ("paraphrase_note", "e3_leak"):
            out[f"{k}_minus_ref"] = paired(
                [by_id[r["fact_id"]][k] for r in ref], [r[k] for r in ref]
            )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CONFIRM / "shard_eval.json"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    report = (
        json.loads(out.read_text()) if out.exists() else {"prereg": "SS112", "arms": {}}
    )
    raw_cf = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    facts = json.loads((CONFIRM / "external_counterfact_n1000.json").read_text())[
        "facts"
    ]
    index = {f["id"]: i for i, f in enumerate(facts)}
    chosen, relation, e3, top = draw(facts, raw_cf)
    report["relations"] = top
    bench = Bench()

    bench.use_rank(2)
    base_e3 = {f["id"]: bench.ask(e3[f["id"]], SYSTEM, adapter=False) for f in chosen}
    saved = torch.load(
        CONFIRM / "store_r2_cf_train.pt", map_location="cpu", weights_only=False
    )["deltas"]
    ref_rows = []
    for f in chosen:
        set_delta(bench.m, saved[f["id"]])
        ref_rows += serve_shard(bench, [f], facts, index, e3, base_e3)
    report["reference"] = {
        **summarize(ref_rows),
        "bytes_per_fact": stored_bytes(saved[chosen[0]["id"]], "int8"),
    }
    print("reference", json.dumps(report["reference"]), flush=True)

    for k, r, grouping in ARMS:
        name = f"{k}/{r}/{grouping}"
        if name in report["arms"]:
            continue
        bench.use_rank(r)
        rows, seconds, size = [], [], None
        for shard in shards_of(chosen, k, grouping):
            t = time.time()
            delta = bench.train(shard)
            seconds.append(round(time.time() - t, 1))
            size = stored_bytes(delta, "int8")
            set_delta(bench.m, delta)
            rows += serve_shard(bench, shard, facts, index, e3, base_e3)
        res = {
            "k": k,
            "rank": r,
            "grouping": grouping,
            "bytes_per_fact": size // k,
            "train_seconds_mean": round(float(np.mean(seconds)), 1),
            **summarize(rows, ref_rows),
        }
        if (k, r, grouping) in REMOVAL:
            shard = shards_of(chosen, k, grouping)[0]
            removed, rest = shard[0], shard[1:]
            t = time.time()
            first = bench.train(rest)
            secs = round(time.time() - t, 1)
            second = bench.train(rest)
            same = all(
                torch.equal(first[n]["A"], second[n]["A"])
                and torch.equal(first[n]["B"], second[n]["B"])
                for n in first
            )
            drift = max(
                float((first[n][x] - second[n][x]).abs().max())
                for n in first
                for x in "AB"
            )
            set_delta(bench.m, first)
            res["removal"] = {
                "seconds": secs,
                "bitwise_equal": same,
                "max_abs_difference": drift,
                "removed_answer_gone": not hit(
                    bench.ask(removed["probe"], SYSTEM), removed["answer"]
                ),
            }
        report["arms"][name] = res
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
        print(name, json.dumps(res), flush=True)

    ref_bytes = report["reference"]["bytes_per_fact"]
    rule = {}
    for name, a in report["arms"].items():
        leak = a["e3_leak_minus_ref"]
        rule[name] = {
            "i_efficacy": a["probe_note"] >= 0.95,
            "ii_paraphrase": a["paraphrase_note_minus_ref"]["lower_95_one_sided"]
            >= -0.03,
            "iii_leak": leak["diff"] + (leak["diff"] - leak["lower_95_one_sided"])
            <= 0.02,
            "iv_kappa": a["above_kappa"] <= 0.10,
            "v_bytes": a["bytes_per_fact"] <= ref_bytes / 4,
        }
    report["rule"] = rule
    report["viable"] = [n for n, m in rule.items() if all(m.values())]
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({"rule": rule, "viable": report["viable"]}, indent=1))


if __name__ == "__main__":
    main()
