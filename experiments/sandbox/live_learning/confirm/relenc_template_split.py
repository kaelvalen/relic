"""SS79: template-disjoint split for the relation encoder (routing only, CPU).

The unit of the split is the subject-masked relation text of the router. Strings under
more than one relation are ambiguous and unused; strings of the 11 SS76 held-out
relations are held; the other strings are split at random into S_train and S_test. Per
seed: a dev split on S_train, an encoder trained on S_train triplets, and a routing store
of 400 facts per group (A: S_train, B: S_test, C: held), compared across the trained
encoder, the frozen all-mpnet-base-v2 and the SS76 encoder, each at its own dev theta.

    .venv/bin/python experiments/sandbox/live_learning/confirm/relenc_template_split.py run --seed 79
    .venv/bin/python experiments/sandbox/live_learning/confirm/relenc_template_split.py summary
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

from external import adapt_counterfact, adapt_zsre  # noqa: E402
from locality_eval import paired_diff_ci  # noqa: E402
from router_v4 import find_subject, key_text, relation_text  # noqa: E402
from router_v7 import BiScorer, RouterV7  # noqa: E402
from router_v7_select import dev_scores, threshold_fpr  # noqa: E402

SEEDS = (79, 80, 81)
DEV_N = 500
GROUP_N = 400
N_CF, N_ZS = 3000, 1500
BOOT = 10000
BASE = "sentence-transformers/all-mpnet-base-v2"
SS76 = "results/live_learning/confirm/relation_encoder_mpnet"
CONFIRM = Path("results/live_learning/confirm")
GROUPS = ("A", "B", "C")


# -- strings and split ------------------------------------------------------
def masked_record(record):
    """(subject, relation, edit prompt template, key text, paraphrase texts)."""
    rw = record["requested_rewrite"]
    subject = rw["subject"]
    key = key_text(rw["prompt"].replace("{}", subject), subject)
    paras = []
    for p in record.get("paraphrase_prompts", []):
        p = p.replace("{}", subject).strip()
        span = find_subject(p, subject)
        if span:
            paras.append((p, relation_text(p, span)))
    return {
        "record": record,
        "subject": subject,
        "relation_id": rw["relation_id"],
        "template": rw["prompt"],
        "key": key,
        "paraphrases": paras,
    }


def build_split(masked, held_relations, seed):
    relations = {}
    for m in masked:
        for s in [m["key"]] + [t for _, t in m["paraphrases"]]:
            relations.setdefault(s, set()).add(m["relation_id"])
    ambiguous = {s for s, rs in relations.items() if len(rs) > 1}
    held = {
        s
        for s, rs in relations.items()
        if s not in ambiguous and next(iter(rs)) in held_relations
    }
    seen = sorted(s for s in relations if s not in ambiguous and s not in held)
    random.Random(seed).shuffle(seen)
    half = len(seen) // 2
    return {
        "train": set(seen[:half]),
        "test": set(seen[half:]),
        "held": held,
        "ambiguous": ambiguous,
        "n_strings": len(relations),
    }


def side_of(split, group):
    return {"A": split["train"], "B": split["test"], "C": split["held"]}[group]


def side_templates(masked, side):
    """relation -> sorted edit templates whose masked key lies on the side."""
    out = {}
    for m in masked:
        if m["key"] in side:
            out.setdefault(m["relation_id"], set()).add(m["template"])
    return {r: sorted(ts) for r, ts in out.items()}


def fill_on_side(rng, templates, relations, subject, side, tries=50):
    """A template of one of `relations` filled with `subject` whose key is on the side."""
    for _ in range(tries):
        t = rng.choice(templates[rng.choice(relations)])
        q = t.replace("{}", subject)
        if key_text(q, subject) in side:
            return q
    return None


# -- dev split ----------------------------------------------------------------
def build_dev(raw_cf, raw_zs, masked, split, old_test_subjects, rng):
    train_t = side_templates(masked, split["train"])
    seen_rel = sorted(train_t)
    cf = []
    for m in masked:
        if m["subject"] in old_test_subjects or m["key"] not in split["train"]:
            continue
        fact = adapt_counterfact(m["record"])
        if fact is None:
            continue
        paras = [p for p, t in m["paraphrases"] if t in split["train"]]
        if not paras:
            continue
        fact["paraphrases"] = paras
        fact["relation_id"] = m["relation_id"]
        cf.append(fact)
    zs = [
        f
        for f in (adapt_zsre(item) for item in raw_zs)
        if f is not None and f["subject"] not in old_test_subjects
    ]
    for f in zs:
        f["paraphrases"] = [f["paraphrase"]]
        f["relation_id"] = None
    rng.shuffle(cf)
    rng.shuffle(zs)
    dev, seen = [], set()
    for pool in (cf, zs):
        taken = 0
        for f in pool:
            if taken == DEV_N:
                break
            if f["subject"] in seen:
                continue
            others = [r for r in seen_rel if r != f["relation_id"]]
            neg = fill_on_side(rng, train_t, others, f["subject"], split["train"])
            if neg is None:
                continue
            seen.add(f["subject"])
            f["negative"] = neg
            dev.append(f)
            taken += 1
    return dev


# -- routing store --------------------------------------------------------------
def build_store(masked, split, excluded_subjects, held_relations, rng):
    store, used = [], set(excluded_subjects)
    for g in GROUPS:
        side = side_of(split, g)
        templates = side_templates(masked, side)
        pool = [
            m
            for m in masked
            if m["key"] in side and any(t in side for _, t in m["paraphrases"])
        ]
        rng.shuffle(pool)
        taken = 0
        for m in pool:
            if taken == GROUP_N:
                break
            if m["subject"] in used:
                continue
            rid = m["relation_id"]
            others = [
                r for r in templates if r != rid and (r in held_relations) == (g == "C")
            ]
            e3 = fill_on_side(rng, templates, others, m["subject"], side)
            if e3 is None:
                continue
            rw = m["record"]["requested_rewrite"]
            used.add(m["subject"])
            store.append(
                {
                    "id": f"{g}{taken:04d}",
                    "group": g,
                    "subject": m["subject"],
                    "relation_id": rid,
                    "probe": rw["prompt"].replace("{}", m["subject"]).strip(),
                    "answer": rw["target_new"]["str"],
                    "paraphrase": next(p for p, t in m["paraphrases"] if t in side),
                    "e3": e3,
                }
            )
            taken += 1
    return store


# -- training -------------------------------------------------------------------
def build_triplets(raw_zs, masked, split, excluded, rng):
    train_t = side_templates(masked, split["train"])
    seen_rel = sorted(train_t)
    blocked = split["test"] | split["held"] | split["ambiguous"]

    def negative(subject, relation):
        others = [r for r in seen_rel if r != relation]
        q = fill_on_side(rng, train_t, others, subject, split["train"])
        return None if q is None else key_text(q, subject)

    cf = [
        m for m in masked if m["key"] in split["train"] and m["subject"] not in excluded
    ]
    rng.shuffle(cf)
    triplets, seen, n_cf = [], set(), 0
    for m in cf:
        if n_cf == N_CF:
            break
        if m["subject"] in seen:
            continue
        positives = [t for _, t in m["paraphrases"] if t in split["train"]]
        rows = []
        for t in positives:
            neg = negative(m["subject"], m["relation_id"])
            if neg is not None:
                rows.append((m["key"], t, neg))
        if not rows:
            continue
        seen.add(m["subject"])
        triplets += rows
        n_cf += 1
    zs = [
        f for f in (adapt_zsre(x) for x in raw_zs) if f and f["subject"] not in excluded
    ]
    rng.shuffle(zs)
    n_zs = 0
    for f in zs:
        if n_zs == N_ZS:
            break
        if f["subject"] in seen:
            continue
        span = find_subject(f["paraphrase"], f["subject"])
        if not span:
            continue
        anchor = key_text(f["probe"], f["subject"])
        positive = relation_text(f["paraphrase"], span)
        if anchor in blocked or positive in blocked:
            continue
        neg = negative(f["subject"], None)
        if neg is None:
            continue
        seen.add(f["subject"])
        triplets.append((anchor, positive, neg))
        n_zs += 1
    rng.shuffle(triplets)
    for a, p, n in triplets:  # the pinned rule: every triplet string comes from S_train
        assert n in split["train"]
        assert a not in blocked and p not in blocked
    return triplets, n_cf, n_zs


def train_encoder(triplets, out_dir, seed, batch=32, lr=2e-5, device="cpu"):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(seed)
    model = SentenceTransformer(BASE, device=device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = (len(triplets) + batch - 1) // batch
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(  # linear warm-up, then linear decay
        opt,
        lambda s: (
            (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm))
        ),
    )
    losses, t0 = [], time.time()
    for step in range(steps):
        chunk = triplets[step * batch : (step + 1) * batch]
        feats = model.tokenize([t for trip in chunk for t in trip])
        feats = {
            k: v.to(model.device) if hasattr(v, "to") else v for k, v in feats.items()
        }
        emb = torch.nn.functional.normalize(model(feats)["sentence_embedding"], dim=-1)
        a, p, n = emb[0::3], emb[1::3], emb[2::3]
        scores = 20.0 * a @ torch.cat([p, n]).T
        loss = torch.nn.functional.cross_entropy(
            scores, torch.arange(len(chunk), device=scores.device)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss))
        if (step + 1) % 50 == 0:
            print(
                f"step {step + 1}/{steps} loss {np.mean(losses[-50:]):.4f}", flush=True
            )
    model.eval()
    model.save(str(out_dir))
    return {
        "steps": steps,
        "batch": batch,
        "lr": lr,
        "device": device,
        "loss_first50": float(np.mean(losses[:50])),
        "loss_last50": float(np.mean(losses[-50:])),
        "seconds": round(time.time() - t0, 1),
    }


# -- evaluation -------------------------------------------------------------------
def rate(xs):
    xs = list(xs)
    return round(float(np.mean(xs)), 4) if xs else None


def evaluate(name, model_id, dev, store):
    scorer = BiScorer(model_id, device="cpu")
    pos, neg = dev_scores(RouterV7(dev, scorer, -np.inf), dev)
    neg_cf = np.asarray([s for part, s in neg if part == "counterfact"])
    pos_cf = np.asarray([s for part, s in pos if part == "counterfact"])
    theta = threshold_fpr(neg_cf)
    router = RouterV7(store, scorer, theta)
    para = [router.route(f["paraphrase"])[0] == i for i, f in enumerate(store)]
    e3 = [router.route(f["e3"])[0] is not None for f in store]
    res = {
        "model": model_id,
        "theta": theta,
        "dev_fpr_cf": float((neg_cf >= theta).mean()),
        "dev_tpr_cf": float((pos_cf >= theta).mean()),
        "groups": {},
        "per_fact": {"paraphrase_route_correct": para, "e3_routed": e3},
    }
    for g in GROUPS:
        idx = [i for i, f in enumerate(store) if f["group"] == g]
        res["groups"][g] = {
            "n": len(idx),
            "route_correct": rate(para[i] for i in idx),
            "e3_fpr": rate(e3[i] for i in idx),
        }
    print(
        name, json.dumps({k: v for k, v in res.items() if k != "per_fact"}), flush=True
    )
    return res


def run(args):
    out = CONFIRM / f"relenc_template_split_s{args.seed}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    raw_cf = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    raw_zs = json.loads((CONFIRM / "raw/zsre.json").read_text())
    held_rel = set(
        json.loads((Path(SS76) / "ss76_meta.json").read_text())["heldout_relations"]
    )
    old_test = {
        f["subject"]
        for name in ("external_counterfact_n1000.json", "external_zsre_n1000.json")
        for f in json.loads((CONFIRM / name).read_text())["facts"]
    }
    masked = [masked_record(r) for r in raw_cf]
    split = build_split(masked, held_rel, args.seed)
    rng = random.Random(args.seed)
    dev = build_dev(raw_cf, raw_zs, masked, split, old_test, rng)
    dev_subjects = {f["subject"] for f in dev}
    store = build_store(masked, split, dev_subjects, held_rel, rng)
    excluded = dev_subjects | {f["subject"] for f in store}
    triplets, n_cf, n_zs = build_triplets(raw_zs, masked, split, excluded, rng)
    info = {
        "n_strings": split["n_strings"],
        "n_ambiguous": len(split["ambiguous"]),
        "n_held": len(split["held"]),
        "n_train": len(split["train"]),
        "n_test": len(split["test"]),
        "dev_cf": sum(f["relation_id"] is not None for f in dev),
        "dev_zsre": sum(f["relation_id"] is None for f in dev),
        "store": {g: sum(f["group"] == g for f in store) for g in GROUPS},
        "n_triplets": len(triplets),
        "n_cf_facts": n_cf,
        "n_zsre_facts": n_zs,
    }
    print(json.dumps(info), flush=True)
    enc_dir = CONFIRM / f"relation_encoder_ss79_s{args.seed}"
    if enc_dir.exists():
        print("encoder exists, not retrained:", enc_dir, flush=True)
        train = json.loads((enc_dir / "ss79_meta.json").read_text())["training"]
    else:
        train = train_encoder(triplets, enc_dir, args.seed, device=args.device)
        (enc_dir / "ss79_meta.json").write_text(
            json.dumps(
                {
                    "prereg": "SS79",
                    "seed": args.seed,
                    "base": BASE,
                    "training": train,
                    **info,
                },
                indent=1,
            )
        )
    encoders = {
        "trained": str(enc_dir),
        "frozen_mpnet": BASE,
        "ss76": SS76,
    }
    results = {n: evaluate(n, m, dev, store) for n, m in encoders.items()}
    diffs = {}
    for other in ("frozen_mpnet", "ss76"):
        for g in GROUPS:
            idx = [i for i, f in enumerate(store) if f["group"] == g]
            for metric in ("paraphrase_route_correct", "e3_routed"):
                a = [results["trained"]["per_fact"][metric][i] for i in idx]
                b = [results[other]["per_fact"][metric][i] for i in idx]
                diffs[f"trained-{other}/{g}/{metric}"] = paired_diff_ci(a, b, n=BOOT)
    report = {
        "prereg": "SS79",
        "seed": args.seed,
        "info": info,
        "training": train,
        "encoders": results,
        "paired_diffs": diffs,
        "store": store,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("saved:", out, flush=True)


def summary(args):
    reports = [
        json.loads((CONFIRM / f"relenc_template_split_s{s}.json").read_text())
        for s in SEEDS
    ]

    def per_seed(enc, g, key):
        return [r["encoders"][enc]["groups"][g][key] for r in reports]

    out = {"prereg": "SS79", "seeds": list(SEEDS), "groups": {}}
    for g in GROUPS:
        out["groups"][g] = {
            enc: {
                "route_correct": per_seed(enc, g, "route_correct"),
                "e3_fpr": per_seed(enc, g, "e3_fpr"),
            }
            for enc in ("trained", "frozen_mpnet", "ss76")
        }
        d = [
            t - f
            for t, f in zip(
                per_seed("trained", g, "route_correct"),
                per_seed("frozen_mpnet", g, "route_correct"),
            )
        ]
        out["groups"][g]["delta"] = {"per_seed": d, "mean": float(np.mean(d))}
    da = out["groups"]["A"]["delta"]["mean"]
    db = out["groups"]["B"]["delta"]["mean"]
    if da < 0.05:
        p1 = "untestable"
    elif db <= 0.5 * da:
        p1 = "template memorisation"
    elif db >= 0.8 * da:
        p1 = "generalisation"
    else:
        p1 = "partial"
    rc_tb = float(np.mean(per_seed("trained", "B", "route_correct")))
    rc_fb = float(np.mean(per_seed("frozen_mpnet", "B", "route_correct")))

    # P3: inflation I = E3 FPR_B - dev FPR, paired over the pooled group-B E3 queries
    def pooled_e3(enc):
        xs = []
        for r in reports:
            store = r["store"]
            xs += [
                e
                for e, f in zip(r["encoders"][enc]["per_fact"]["e3_routed"], store)
                if f["group"] == "B"
            ]
        return xs

    e3_t, e3_f = pooled_e3("trained"), pooled_e3("frozen_mpnet")
    dev_t = float(np.mean([r["encoders"]["trained"]["dev_fpr_cf"] for r in reports]))
    dev_f = float(
        np.mean([r["encoders"]["frozen_mpnet"]["dev_fpr_cf"] for r in reports])
    )
    i_t, i_f = float(np.mean(e3_t)) - dev_t, float(np.mean(e3_f)) - dev_f
    e3_diff = paired_diff_ci(e3_t, e3_f, n=BOOT)
    shift = dev_t - dev_f
    out["P1"] = {
        "delta_A": da,
        "delta_B": db,
        "ratio_B_A": db / da if da else None,
        "outcome": p1,
        "predicted": "template memorisation",
    }
    out["P2"] = {
        "route_correct_trained_B": rc_tb,
        "route_correct_frozen_B": rc_fb,
        "holds": rc_tb <= rc_fb + 0.05,
    }
    out["P3"] = {
        "I_trained": i_t,
        "I_frozen": i_f,
        "I_diff": i_t - i_f,
        "I_diff_ci95": [c - shift for c in e3_diff["ci95"]],
        "n": e3_diff["n"],
        "holds": i_t - i_f >= 0.02,
    }
    path = CONFIRM / "relenc_template_split.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seed", type=int, choices=SEEDS, required=True)
    r.add_argument("--device", default="cpu")
    sub.add_parser("summary")
    args = ap.parse_args(argv)
    run(args) if args.cmd == "run" else summary(args)


if __name__ == "__main__":
    main()
