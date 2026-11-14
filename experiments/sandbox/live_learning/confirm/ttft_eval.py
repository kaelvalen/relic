"""Time to first token through the ledger (draft entry 163; CHANGES.md, item 10).

A pinned query set: the edit prompts of K stored entries, which the router sends to an entry,
and K MMLU and ARC prompts of entry 101, which it sends nowhere. Each query is timed in two
conditions:

  base   the base model alone: prefill and the first generated token;
  store  Ledger serving: routing, the adapter swap (Serve's materialize), prefill and the
         first generated token. A query the router sends nowhere has no swap.

The first token is the greedy argmax of the prefill's last logits, the token `generate` would
emit first. Every timer is bracketed by torch.cuda.synchronize. W warmup queries are run and
discarded, then each query is repeated R times in alternating condition order.

.venv/bin/python experiments/sandbox/live_learning/confirm/ttft_eval.py \\
    --ledger results/live_learning/confirm/notless_ledger_cf_current.ledger --prereg SS163
.venv/bin/python experiments/sandbox/live_learning/confirm/ttft_eval.py --ledger PATH --smoke
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
from adapter_only import CONFIRM, MODEL  # noqa: E402
from general_eval import prompt  # noqa: E402

from relic.ledger import DeltaStore, Ledger, chat_messages  # noqa: E402
from relic.router import make_scorer  # noqa: E402

SEED = 163
K, WARMUP, REPEATS = 100, 10, 5
# total_s is the time to first token: route + swap + first token. swap_back_s, the return
# to the full set after the token, is timed on its own and is not part of it.
PARTS = ("route_s", "swap_s", "first_token_s", "swap_back_s", "total_s")


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def query_set(ledger, k: int) -> list[dict]:
    """K routed edit prompts and K never-routed MMLU/ARC prompts, in a seeded order."""
    rng = random.Random(SEED)
    ids = sorted(ledger.entries)
    routed = [
        {"kind": "routed", "q": ledger.entries[i]["probe"], "entry": i}
        for i in rng.sample(ids, min(k, len(ids)))
    ]
    rows = json.loads((CONFIRM / "general_eval.json").read_text())["rows"]
    general = [
        {"kind": "never_routed", "q": prompt(r), "entry": None}
        for r in rng.sample(rows, min(k, len(rows)))
    ]
    return routed + general


@torch.no_grad()
def first_token(store, q: str) -> int:
    ids = store.tok.apply_chat_template(
        chat_messages(store.system, {"role": "user", "content": q}),
        add_generation_prompt=True,
        return_tensors="pt",
    )
    ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(store.model.device)
    return int(store.model(input_ids=ids).logits[0, -1].argmax())


def time_base(store, q: str) -> dict:
    sync()
    t0 = time.perf_counter()
    token = first_token(store, q)
    sync()
    t1 = time.perf_counter()
    return {
        "route_s": 0.0,
        "swap_s": 0.0,
        "first_token_s": t1 - t0,
        "swap_back_s": 0.0,
        "token": token,
    }


def time_store(ledger, q: str) -> dict:
    store = ledger.store
    router, facts = ledger.router
    sync()
    t0 = time.perf_counter()
    j, _ = router.route(q)
    fid = None if j is None else facts[j]["id"]
    t1 = time.perf_counter()
    full = store.deltas
    if fid is not None and fid in full:
        store.deltas = {fid: full[fid]}
        store.materialize()
    else:
        fid = None
    sync()
    t2 = time.perf_counter()
    token = first_token(store, q)
    sync()
    t3 = time.perf_counter()
    if fid is not None:  # back to the full set, as Serve leaves it
        store.deltas = full
        store.materialize()
    sync()
    t4 = time.perf_counter()
    return {
        "route_s": t1 - t0,
        "swap_s": t2 - t1,
        "first_token_s": t3 - t2,
        "swap_back_s": t4 - t3,
        "token": token,
        "entry": fid,
    }


def load_ledger(path, store, scorer) -> Ledger:
    """The ledger with the settings and prompts its file records; the hash is checked."""
    return Ledger.from_file(path, store, scorer)


def summary(rows: list[dict]) -> dict:
    out = {}
    for part in PARTS:
        xs = np.asarray([r[part] for r in rows], dtype=float)
        q1, med, q3 = np.percentile(xs, [25, 50, 75])
        out[part] = {
            "median": round(float(med), 6),
            "iqr": [round(float(q1), 6), round(float(q3), 6)],
        }
    out["n"] = len(rows)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", required=True, help="a file written by Ledger.save")
    ap.add_argument("--prereg", help="the entry that pins this run")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--smoke", action="store_true", help="3 queries of each kind, once")
    args = ap.parse_args(argv)
    if not (args.smoke or args.prereg):
        ap.error("--prereg is required for a full run")
    k, warmup, repeats = (
        (3, 1, 1) if args.smoke else (args.k, args.warmup, args.repeats)
    )
    out = CONFIRM / ("ttft_eval_smoke.json" if args.smoke else "ttft_eval.json")
    if out.exists():
        ap.error(f"refusing to overwrite {out}")

    store = DeltaStore(MODEL, steps=16, capacity=1)
    # the settings and prompts come from the file; a changed file is refused
    ledger = load_ledger(args.ledger, store, make_scorer("relenc"))
    queries = query_set(ledger, k)
    _ = ledger.router  # built once, as a serving process holds it

    for item in queries[:warmup]:
        time_base(store, item["q"])
        time_store(ledger, item["q"])

    rows = []
    for rep in range(repeats):
        for n, item in enumerate(queries):
            # alternate which condition runs first, so neither always gets a warm cache
            order = ("base", "store") if (rep + n) % 2 == 0 else ("store", "base")
            for cond in order:
                with ledger.lock:
                    if cond == "base":
                        full = store.deltas
                        store.deltas = {}
                        store.materialize()
                        r = time_base(store, item["q"])
                        store.deltas = full
                        store.materialize()
                    else:
                        r = time_store(ledger, item["q"])
                r.update(cond=cond, kind=item["kind"], repeat=rep, query=n)
                r["total_s"] = r["route_s"] + r["swap_s"] + r["first_token_s"]
                rows.append(r)
        print(f"repeat {rep + 1}/{repeats}", flush=True)

    by = {
        f"{cond}/{kind}": summary(
            [r for r in rows if r["cond"] == cond and r["kind"] == kind]
        )
        for cond in ("base", "store")
        for kind in ("routed", "never_routed")
    }
    routed_as_pinned = np.mean(
        [
            r["entry"] is not None
            for r in rows
            if r["cond"] == "store" and r["kind"] == "routed"
        ]
    )
    never = [r for r in rows if r["cond"] == "store" and r["kind"] == "never_routed"]
    report = {
        "prereg": args.prereg or "smoke",
        "ledger": str(args.ledger),
        "state_hash": ledger.state_hash(),
        "revision": store.revision,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "k": k,
        "warmup": warmup,
        "repeats": repeats,
        "routed_share_of_routed_queries": round(float(routed_as_pinned), 4),
        "never_routed_that_routed": sum(r["entry"] is not None for r in never),
        "summary": by,
        "rows": rows,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(by, indent=1))


if __name__ == "__main__":
    main()
