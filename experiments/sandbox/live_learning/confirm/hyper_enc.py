"""SS151: the hypernetwork of SS149 with the answer's identity in its input (prereg entry 151).

    python hyper_enc.py --smoke   # a few steps and facts, end to end
    python hyper_enc.py           # the pinned run -> hyper_enc.json

Ea: last-layer mean over the teach sentence + mean input embedding of the answer's tokens.
Eb: the same mean + mean layer-14 state over the answer's tokens. Then Hk and Rk on the
better input: k passes of today's training from the generated adapter or from scratch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter_only import CONFIRM, MODEL, draw_facts  # noqa: E402
from hyper_spike import (  # noqa: E402
    N_UNSEEN,
    SEED,
    STEPS,
    Hooks,
    make_delta,
    raw_delta,
    read,
    selfcheck,
    summary,
    train,
)

from relic.adapters import fact_pairs, train_adapter  # noqa: E402
from relic.ledger import KAPPA, DeltaStore  # noqa: E402

POOL = 1000
INPUTS = {"Ea": 0, "Eb": 14}  # hidden_states index read at the answer's tokens
KS = (2, 4)


def answer_positions(tok, fact) -> list[int]:
    """Tokens of the teach sentence whose span starts at or after the end of the edit prompt."""
    enc = tok(fact["teach"], return_offsets_mapping=True)
    pos = [k for k, (s, _) in enumerate(enc["offset_mapping"]) if s >= len(fact["probe"])]
    assert pos, f"no answer token in {fact['teach']!r}"
    return pos


@torch.no_grad()
def embed(store, facts, layer: int) -> torch.Tensor:
    store.deltas = {}
    store.materialize()
    out = []
    for f in facts:
        ids = store.tok(f["teach"], return_tensors="pt").input_ids.cuda()
        hs = store.model(input_ids=ids, output_hidden_states=True).hidden_states
        ans = hs[layer][0, answer_positions(store.tok, f)].float().mean(0)
        out.append(torch.cat([unit(hs[-1][0].float().mean(0)), unit(ans)]).cpu())
    return torch.stack(out)


def unit(v: torch.Tensor) -> torch.Tensor:
    """RMS 1: input embeddings are about 130 times smaller than last-layer states."""
    return v / v.norm() * v.numel() ** 0.5


def passes(store, unseen, k: int, inits=None) -> dict:
    """k passes of today's training per fact, from `inits` or from a fresh LoRA."""
    store.steps, rows, secs = k, [], []
    for n, f in enumerate(unseen):
        t0 = time.time()
        init = None if inits is None else inits[n]
        delta = train_adapter(store, fact_pairs(f), seed=126, init=init)
        secs.append(time.time() - t0)
        rows.append(read(store, delta, f, True))
    store.steps = 16
    return {
        "unseen_edit": summary(rows, "edit"),
        "unseen_para": summary(rows, "para"),
        "under_kappa": round(float(np.mean([r["kld"] <= KAPPA for r in rows])), 4),
        "seconds_per_adapter": round(float(np.median(secs)), 2),
        "rows_unseen": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    store = DeltaStore(MODEL, steps=16, capacity=1)
    store.kl_lambda = 1.0
    hooks = Hooks(store)
    selfcheck(store, hooks)

    seen = {f["subject"] for fs in draw_facts(150)[0].values() for f in fs}
    seen |= {f["subject"] for fs in draw_facts(400, seed=136, exclude=seen)[0].values() for f in fs}
    cf = draw_facts(POOL + N_UNSEEN, seed=SEED, exclude=seen)[0]["cf"]
    pool, unseen = cf[:POOL], cf[POOL:]
    steps, n_read, ks = STEPS, 100, KS
    if args.smoke:
        pool, unseen, steps, n_read, ks = pool[:16], unseen[:4], 20, 4, (1,)

    report = {"prereg": "SS151", "model": MODEL, "pool": len(pool), "steps": steps, "arms": {}}
    hypers, embs = {}, {}
    for name, layer in INPUTS.items():
        emb_pool, emb_unseen = embed(store, pool, layer), embed(store, unseen, layer)
        log, t0 = [], time.time()
        hyper = train(store, hooks, pool, emb_pool, steps, log=log)
        train_s = time.time() - t0
        hyper.eval()
        if not args.smoke:
            torch.save(hyper.state_dict(), CONFIRM / f"hyper_enc_{name}.pt")
        t1 = time.time()
        deltas = [make_delta(hyper, e) for e in emb_unseen]
        make_s = (time.time() - t1) / len(deltas)
        rows_u = [read(store, d, f, True) for d, f in zip(deltas, unseen)]
        rows_t = [
            read(store, make_delta(hyper, emb_pool[k]), pool[k], False) for k in range(n_read)
        ]
        report["arms"][f"H0-{name}"] = {
            "train_edit": summary(rows_t, "edit"),
            "unseen_edit": summary(rows_u, "edit"),
            "unseen_para": summary(rows_u, "para"),
            "under_kappa": round(float(np.mean([r["kld"] <= KAPPA for r in rows_u])), 4),
            "seconds_per_adapter": round(make_s, 4),
            "train_seconds": round(train_s, 1),
            "log": log,
            "rows_unseen": rows_u,
            "rows_train": rows_t,
        }
        hypers[name], embs[name] = hyper, emb_unseen

    rate = {n: report["arms"][f"H0-{n}"]["unseen_edit"]["rate"] for n in INPUTS}
    best = "Ea" if rate["Ea"] >= rate["Eb"] else "Eb"
    report["best_input"] = best
    inits = [raw_delta(hypers[best], e) for e in embs[best]]
    del hypers
    torch.cuda.empty_cache()
    for k in ks:
        report["arms"][f"H{k}-{best}"] = passes(store, unseen, k, inits)
        report["arms"][f"R{k}"] = passes(store, unseen, k)

    out = CONFIRM / ("hyper_enc_smoke.json" if args.smoke else "hyper_enc.json")
    out.write_text(json.dumps(report, indent=1))
    brief = {
        a: {k: v for k, v in r.items() if not k.startswith(("rows", "log"))}
        for a, r in report["arms"].items()
    }
    print(json.dumps(brief, indent=1), flush=True)


if __name__ == "__main__":
    main()
