"""Exploratory entry point of the transactional store (slice-1).

The store itself is relic.ledger.DeltaStore. The slice-1 records anchor on the capability
prompts of their stream, so this module keeps that default and the smoke run.

Smoke:  python ledger.py --facts 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stream import CAPABILITY, FACTS  # noqa: E402

from relic.ledger import CAP, R  # noqa: E402, F401
from relic.ledger import DeltaStore as _DeltaStore  # noqa: E402
from relic.metrics import sym_kl  # noqa: E402, F401


class DeltaStore(_DeltaStore):
    def __init__(self, model_name: str, **kwargs):
        kwargs.setdefault("kl_prompts", [q for _, q, _ in CAPABILITY[:2]])
        super().__init__(model_name, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--facts", type=int, default=3)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--out", default="results/live_learning/ledger_smoke.json")
    args = ap.parse_args()
    store = DeltaStore(args.model, steps=args.steps)
    facts = FACTS[: args.facts]
    hashes = {}
    t_add = {}
    for f in facts:
        t_add[f.id] = store.add(
            f.id,
            [(f.teach, f"Not aldım: {f.teach}"), (f.probe, f.answer)],
            key=f.probe,
        )
        hashes[f.id] = store.state_hash()
        print(
            f"added {f.id} in {t_add[f.id]:.1f}s hash {hashes[f.id][:10]}", flush=True
        )

    def probe(kind: str = "merged"):
        fn = store.answer if kind == "merged" else store.answer_routed
        return {
            f.id: {"q": f.probe, "resp": fn(f.probe), "expect": f.answer} for f in facts
        }

    with_all = probe()
    routed_all = probe("routed")
    # revoke the LAST addition: the hash must equal the recorded pre-add hash exactly
    last = facts[-1].id
    prev_hash = hashes[facts[-2].id] if len(facts) > 1 else store.base_hash
    t_rev = store.revoke(last)
    revoke_hash = store.state_hash()
    after = probe()
    after_routed = probe("routed")

    def ok(resp, expect):
        return expect.lower() in resp.lower()

    result = {
        "model": args.model,
        "facts": [f.id for f in facts],
        "hashes": hashes,
        "revoke_last": {
            "id": last,
            "hash_before_add": prev_hash,
            "hash_after_revoke": revoke_hash,
            "identity": prev_hash == revoke_hash,
            "seconds": round(t_rev, 3),
        },
        "recall_merged_all": {
            fid: ok(r["resp"], r["expect"]) for fid, r in with_all.items()
        },
        "recall_routed_all": {
            fid: ok(r["resp"], r["expect"]) for fid, r in routed_all.items()
        },
        "recall_routed_after_revoke": {
            fid: ok(r["resp"], r["expect"]) for fid, r in after_routed.items()
        },
        "recall_merged_after_revoke": {
            fid: ok(r["resp"], r["expect"]) for fid, r in after.items()
        },
        "responses": {
            "merged_all": with_all,
            "routed_all": routed_all,
            "after_merged": after,
            "after_routed": after_routed,
        },
        "seconds_add_mean": round(sum(t_add.values()) / len(t_add), 3),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    rl = result["recall_merged_all"]
    rr = result["recall_routed_all"]
    ra = result["recall_routed_after_revoke"]
    print(
        f"identity(revoke last)={result['revoke_last']['identity']} | "
        f"recall merged={rl} | routed={rr} | routed after revoke={ra} | "
        f"add {result['seconds_add_mean']}s revoke {t_rev:.2f}s"
    )


if __name__ == "__main__":
    main()
