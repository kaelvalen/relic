"""SS76: train a relation encoder offline on data disjoint from the dev split and the
test stores, with 11 CounterFact relations held out entirely.

Triplets of subject-masked relation texts (anchor = edit prompt, positive = paraphrase,
hard negative = template of another training relation), multiple-negatives ranking loss.

    .venv/bin/python experiments/sandbox/live_learning/confirm/train_relation_encoder.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

from external import adapt_zsre  # noqa: E402
from router_v4_calibrate import _templates_by_relation, build_dev  # noqa: E402

from relic.router import find_subject, key_text, relation_text  # noqa: E402

SEED = 76
N_HELDOUT = 11
N_CF, N_ZS = 3000, 1500
BASE = "sentence-transformers/all-mpnet-base-v2"
CONFIRM = Path("results/live_learning/confirm")
OUT_DIR = CONFIRM / "relation_encoder_mpnet"


def build_triplets(
    raw_cf, raw_zs, excluded_subjects, rng, partners=None, hold_out=True
):
    """partners (SS110): relation -> relations it is confused with, sampled half the time."""
    by_relation = _templates_by_relation(raw_cf)
    relations = sorted(by_relation)
    heldout = sorted(rng.sample(relations, N_HELDOUT)) if hold_out else []
    train_rel = [r for r in relations if r not in heldout]

    def negative(subject, relation):
        hard = [r for r in (partners or {}).get(relation, []) if r in train_rel]
        if hard and rng.random() < 0.5:
            r2 = rng.choice(hard)
        else:
            r2 = rng.choice([r for r in train_rel if r != relation])
        return key_text(rng.choice(by_relation[r2]).replace("{}", subject), subject)

    cf_records = [
        r
        for r in raw_cf
        if r["requested_rewrite"]["relation_id"] in train_rel
        and r["requested_rewrite"]["subject"] not in excluded_subjects
    ]
    rng.shuffle(cf_records)
    triplets, seen = [], set()
    n_cf = 0
    for r in cf_records:
        rw = r["requested_rewrite"]
        subject = rw["subject"]
        if subject in seen or n_cf == N_CF:
            continue
        seen.add(subject)
        anchor = key_text(rw["prompt"].replace("{}", subject), subject)
        for p in r.get("paraphrase_prompts", []):
            span = find_subject(p, subject)
            if span:
                triplets.append(
                    (
                        anchor,
                        relation_text(p, span),
                        negative(subject, rw["relation_id"]),
                    )
                )
        n_cf += 1
    zs = [
        f
        for f in (adapt_zsre(x) for x in raw_zs)
        if f and f["subject"] not in excluded_subjects
    ]
    rng.shuffle(zs)
    n_zs = 0
    for f in zs:
        if f["subject"] in seen or n_zs == N_ZS:
            continue
        span = find_subject(f["paraphrase"], f["subject"])
        if not span:
            continue
        seen.add(f["subject"])
        triplets.append(
            (
                key_text(f["probe"], f["subject"]),
                relation_text(f["paraphrase"], span),
                negative(f["subject"], None),
            )
        )
        n_zs += 1
    rng.shuffle(triplets)
    return triplets, heldout, train_rel, n_cf, n_zs


def ss76_triplets(hard=None, hold_out=True):
    """The SS76 triplets, rebuilt exactly: seed 76, test and dev subjects excluded."""
    raw_cf = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    raw_zs = json.loads((CONFIRM / "raw/zsre.json").read_text())
    tests = [
        json.loads((CONFIRM / f).read_text())["facts"]
        for f in ("external_counterfact_n1000.json", "external_zsre_n1000.json")
    ]
    test_subjects = {f["subject"] for fs in tests for f in fs}
    dev = build_dev(raw_cf, raw_zs, test_subjects)
    excluded = test_subjects | {f["subject"] for f in dev}
    rng = random.Random(SEED)
    partners = json.loads(Path(hard).read_text())["partners"] if hard else None
    out = build_triplets(raw_cf, raw_zs, excluded, rng, partners, hold_out=hold_out)
    return (*out, rng)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--hard", help="SS110: relation_confusions.json")
    ap.add_argument("--no-holdout", action="store_true", help="SS110 arm B")
    ap.add_argument("--base", default=BASE, help="SS154: another base encoder")
    ap.add_argument("--tr", help="SS158: translations of the triplet texts")
    args = ap.parse_args(argv)
    if Path(args.out).exists():
        ap.error(f"refusing to overwrite {args.out}")

    import torch
    from sentence_transformers import SentenceTransformer

    triplets, heldout, train_rel, n_cf, n_zs, rng = ss76_triplets(
        args.hard, not args.no_holdout
    )
    n_tr = 0
    if args.tr:  # SS158: the same triplets with translated texts, three ways
        tr = json.loads(Path(args.tr).read_text())["map"]
        extra = [
            t
            for a, p, n in triplets
            if a in tr and p in tr and n in tr
            for t in ((a, tr[p], tr[n]), (tr[a], tr[p], tr[n]), (tr[a], p, n))
        ]
        n_tr = len(extra)
        triplets = triplets + extra
        rng.shuffle(triplets)
    print(
        f"triplets {len(triplets)} (cf facts {n_cf}, zsre facts {n_zs}); held out {heldout}",
        flush=True,
    )

    torch.manual_seed(SEED)
    model = SentenceTransformer(args.base, device="cuda")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps = (len(triplets) + args.batch - 1) // args.batch
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(  # linear warm-up, then linear decay
        opt,
        lambda s: (
            (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm))
        ),
    )
    losses, t0 = [], time.time()
    for step in range(steps):
        batch = triplets[step * args.batch : (step + 1) * args.batch]
        texts = [t for trip in batch for t in trip]
        feats = model.tokenize(texts)
        feats = {
            k: v.to(model.device) if hasattr(v, "to") else v for k, v in feats.items()
        }
        emb = torch.nn.functional.normalize(model(feats)["sentence_embedding"], dim=-1)
        a, p, n = emb[0::3], emb[1::3], emb[2::3]
        scores = 20.0 * a @ torch.cat([p, n]).T
        loss = torch.nn.functional.cross_entropy(
            scores, torch.arange(len(batch), device=scores.device)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss))
        if (step + 1) % 50 == 0:
            print(
                f"step {step + 1}/{steps} loss {sum(losses[-50:]) / 50:.4f}", flush=True
            )
    model.eval()
    model.save(args.out)
    meta = {
        "prereg": "SS110"
        if args.hard
        else "SS158"
        if args.tr
        else ("SS154" if args.base != BASE else "SS76"),
        "n_translated_triplets": n_tr,
        "hard": args.hard,
        "base": args.base,
        "seed": SEED,
        "heldout_relations": heldout,
        "train_relations": train_rel,
        "n_triplets": len(triplets),
        "n_cf_facts": n_cf,
        "n_zsre_facts": n_zs,
        "batch": args.batch,
        "lr": args.lr,
        "steps": steps,
        "loss_first50": sum(losses[:50]) / min(50, len(losses)),
        "loss_last50": sum(losses[-50:]) / min(50, len(losses)),
        "seconds": round(time.time() - t0, 1),
    }
    (Path(args.out) / "ss76_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
