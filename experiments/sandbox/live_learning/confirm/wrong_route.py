"""SS100: the cost of a wrong route on 1000 questions, per serving arm (eval-only).

.venv/bin/python experiments/sandbox/live_learning/confirm/wrong_route.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from external import adapt_zsre  # noqa: E402
from routed_eval import (  # noqa: E402
    MANIFEST_PATH,
    REPO_ROOT,
    _repo_path,
    load_verified_checkpoint,
    verify_manifest_inputs,
)

from relic.ledger import DeltaStore  # noqa: E402
from relic.metrics import hit, word_hit  # noqa: E402
from relic.serving import SYSTEM, base_answer, rag_system, trim  # noqa: E402

SEED = 69  # the SS69 sequence, so the first 200 questions are those of SS69 and SS94
BOOT = 10000
MARGIN = -0.03
CONFIRM = Path("results/live_learning/confirm")
ARMS = ("adapter_r16", "adapter_r2", "note", "both_r16", "both_r2")


def load(manifest, ckpt, facts_file, model_id):
    verified = verify_manifest_inputs(
        manifest,
        _repo_path(CONFIRM / ckpt, REPO_ROOT),
        _repo_path(CONFIRM / facts_file, REPO_ROOT),
        model_id,
        repo_root=REPO_ROOT,
    )
    checkpoint, _ = load_verified_checkpoint(
        verified["checkpoint_path"],
        verified["checkpoint_sha256"],
        lambda source: torch.load(source, map_location="cpu", weights_only=False),
    )
    return verified["facts"], checkpoint["deltas"]


def with_adapter(store, delta, q, system):
    store.deltas = {"x": delta}
    store.materialize()
    resp = store.answer(q, system=system)
    store.deltas = {}
    store.materialize()
    return resp


def paired(a, b):
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(SEED)
    boots = diff[rng.integers(0, diff.size, size=(BOOT, diff.size))].mean(axis=1)
    return {
        "diff": round(float(diff.mean()), 4),
        "ci95": [
            round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4),
        ],
        "lower_95_one_sided": round(float(np.percentile(boots, 5)), 4),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default=str(CONFIRM / "wrong_route.json"))
    args = ap.parse_args(argv)
    if Path(args.out).exists():
        ap.error(f"refusing to overwrite {args.out}")
    manifest = json.loads(MANIFEST_PATH.read_bytes())
    model_id = manifest["models"]["base"]["id"]
    facts_file = "external_counterfact_n1000.json"
    facts, r16 = load(manifest, "cf_n1000.ckpt", facts_file, model_id)
    _, r2 = load(manifest, "cf_n1000_r2.ckpt", facts_file, model_id)
    index = {f["id"]: i for i, f in enumerate(facts)}

    store = DeltaStore(model_id, steps=16, capacity=1)
    store.deltas = {}
    store.materialize()

    raw_zs = json.loads((CONFIRM / "raw/zsre.json").read_text())
    zs_test = json.loads((CONFIRM / "external_zsre_n1000.json").read_text())["facts"]
    zs_subjects = {f["subject"] for f in zs_test}
    dev_zs = [
        f
        for f in (adapt_zsre(r) for r in raw_zs)
        if f is not None and f["subject"] not in zs_subjects
    ]
    rng = random.Random(SEED)
    rng.shuffle(dev_zs)
    rows = []
    for f in dev_zs:
        if len(rows) == args.n:
            break
        q, answer = f["probe"], f["answer"]
        base = base_answer(store, q)
        if not hit(base, answer):
            continue
        wrong = rng.choice(facts)
        note = rag_system("instruct", facts, index[wrong["id"]])
        responses = {
            "adapter_r16": with_adapter(store, r16[wrong["id"]], q, SYSTEM),
            "adapter_r2": with_adapter(store, r2[wrong["id"]], q, SYSTEM),
            "note": base_answer(store, q, system=note),
            "both_r16": with_adapter(store, r16[wrong["id"]], q, note),
            "both_r2": with_adapter(store, r2[wrong["id"]], q, note),
        }
        rows.append(
            {
                "question": q,
                "answer": answer,
                "adapter": wrong["id"],
                "base": base,
                "word_base": word_hit(base, answer),
                "retained": {a: hit(r, answer) for a, r in responses.items()},
                "retained_word": {a: word_hit(r, answer) for a, r in responses.items()},
                "responses": responses,
            }
        )
        trim()
        if len(rows) % 100 == 0:
            print(f"wrong route: {len(rows)}/{args.n}", flush=True)

    def col(arm, n=None, key="retained"):
        return [r[key][arm] for r in rows[:n]]

    word_rows = [r for r in rows if r["word_base"]]
    report = {
        "prereg": "SS100",
        "n": len(rows),
        "retained": {a: round(float(np.mean(col(a))), 4) for a in ARMS},
        "retained_word_on_word_base": {
            a: round(float(np.mean([r["retained_word"][a] for r in word_rows])), 4)
            for a in ARMS
        },
        "first_200": {a: round(float(np.mean(col(a, 200))), 4) for a in ARMS},
        "r2_minus_r16_adapter": paired(col("adapter_r2"), col("adapter_r16")),
        "r2_minus_r16_both": paired(col("both_r2"), col("both_r16")),
        "note_minus_adapter_r16": paired(col("note"), col("adapter_r16")),
        "note_minus_adapter_r2": paired(col("note"), col("adapter_r2")),
        "both_minus_adapter_r16": paired(col("both_r16"), col("adapter_r16")),
        "both_minus_note_r2": paired(col("both_r2"), col("note")),
        "margin": MARGIN,
        "rows": rows,
    }
    report["rank2_becomes_main"] = (
        report["r2_minus_r16_adapter"]["lower_95_one_sided"] >= MARGIN
    )
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
