"""SS105: aliases, pronouns and short conversations.

.venv/bin/python experiments/sandbox/live_learning/confirm/alias_eval.py
.venv/bin/python experiments/sandbox/live_learning/confirm/alias_eval.py --serve
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from general_eval import arc, mmlu, prompt  # noqa: E402
from locality_eval import match_raw, other_relation_prompts  # noqa: E402
from routed_eval import MANIFEST_PATH  # noqa: E402
from router_v4_calibrate import build_dev  # noqa: E402
from wrong_route import load, paired, with_adapter  # noqa: E402

from relic.ledger import DeltaStore  # noqa: E402
from relic.metrics import hit, word_hit  # noqa: E402
from relic.router import (  # noqa: E402
    RouterV7,
    make_scorer,
    normalize,
    resolve_reference,
    usable_aliases,
)
from relic.serving import rag_system, trim  # noqa: E402

SEED = 105
BOOT = 10000
THETA = 0.7501
CONFIRM = Path("results/live_learning/confirm")
FILES = {
    "counterfact": ("cf_n1000.ckpt", "external_counterfact_n1000.json"),
    "zsre": ("zsre_n1000.ckpt", "external_zsre_n1000.json"),
}
FILLERS = ["Interesting, go on.", "Can you say more?"]


def swap(text: str, subject: str, new: str) -> str | None:
    """`text` with the subject replaced, or None if the subject is not literally in it."""
    found = re.search(re.escape(subject), text, flags=re.IGNORECASE)
    if found is None:
        return None
    return text[: found.start()] + new + text[found.end() :]


def pronoun_form(text: str, subject: str) -> str | None:
    found = re.search(re.escape(subject) + r"('s)?", text, flags=re.IGNORECASE)
    if found is None:
        return None
    word = "its" if found.group(1) else "it"
    return text[: found.start()] + word + text[found.end() :]


def rate(xs) -> dict:
    xs = [bool(x) for x in xs]
    return {"n": len(xs), "rate": round(float(np.mean(xs)), 4) if xs else None}


def one_dataset(
    dataset, facts, aliases, raw_cf, unstored, natural, bench, scorer, model
):
    rng = random.Random(SEED)
    with_names = []
    for f in facts:
        names = usable_aliases(aliases.get(f["subject"], {}).get("aliases", []))
        with_names.append({**f, "aliases": names})
    plain = RouterV7(facts, scorer, THETA)
    named = RouterV7(with_names, scorer, THETA)

    def correct(router, q, f):
        j, _ = router.route(q)
        if j is None:
            return False
        e = facts[j]
        return normalize(e["subject"]) == normalize(f["subject"]) and normalize(
            e["answer"]
        ) == normalize(f["answer"])

    def routed(router, q):
        return router.route(q)[0] is not None

    with_alias = []
    for f, g in zip(facts, with_names):
        pool = [
            a
            for a in g["aliases"]
            if normalize(f["subject"]).strip() not in normalize(a)
        ]
        if pool:
            with_alias.append((f, rng.choice(pool)))

    sets = {}
    for kind, key in (("A1", "probe"), ("A2", "paraphrase")):
        rows = []
        for f, a in with_alias:
            q = swap(f[key], f["subject"], a)
            if q is not None:
                rows.append(
                    {
                        "fact": f["id"],
                        "query": q,
                        "plain": correct(plain, q, f),
                        "named": correct(named, q, f),
                    }
                )
        sets[kind] = rows

    for kind, key, gap in (
        ("C1_probe", "probe", 0),
        ("C1_paraphrase", "paraphrase", 0),
        ("C4_probe", "probe", 2),
    ):
        rows = []
        for f in facts:
            q = pronoun_form(f[key], f["subject"])
            if q is None:
                continue
            history = [f"Tell me about {f['subject']}."] + FILLERS[:gap]
            r = resolve_reference(q, history, named)
            rows.append(
                {
                    "fact": f["id"],
                    "query": q,
                    "resolved": r,
                    "no_history": correct(named, q, f),
                    "named": correct(named, r, f),
                }
            )
        sets[kind] = rows

    # C2: a later turn names someone the store does not hold
    c2 = []
    for f in facts:
        q = pronoun_form(f["probe"], f["subject"])
        if q is None:
            continue
        history = [
            f"Tell me about {f['subject']}.",
            f"What about {rng.choice(unstored)}?",
        ]
        r = resolve_reference(q, history, named)
        c2.append({"fact": f["id"], "resolved": r, "routed": routed(named, r)})
    sets["C2"] = c2

    false = {
        "F2_distractors": [
            {
                "plain": routed(plain, f["distractor"]),
                "named": routed(named, f["distractor"]),
            }
            for f in facts
        ],
        "F3_benchmarks": [
            {"plain": routed(plain, q), "named": routed(named, q)} for q in bench
        ],
    }
    if dataset == "counterfact":
        prompts = [
            p["query"] for p in other_relation_prompts(match_raw(facts, raw_cf), raw_cf)
        ]
        false["F1_other_relation"] = [
            {"plain": routed(plain, q), "named": routed(named, q)} for q in prompts
        ]
        alias_of = dict((f["id"], a) for f, a in with_alias)
        f1a, c3 = [], []
        for f, q in zip(facts, prompts):
            if f["id"] in alias_of:
                s = swap(q, f["subject"], alias_of[f["id"]])
                if s:
                    f1a.append({"plain": routed(plain, s), "named": routed(named, s)})
            p = pronoun_form(q, f["subject"])
            if p:
                r = resolve_reference(p, [f"Tell me about {f['subject']}."], named)
                c3.append({"routed": routed(named, r)})
        false["F1a_other_relation_alias"] = f1a
        sets["C3"] = c3
        false["F4_natural"] = [
            {"plain": routed(plain, q["query"]), "named": routed(named, q["query"])}
            for q in natural
        ]

    reading = {
        k: {
            "n": len(v),
            **{
                c: rate(r[c] for r in v)["rate"]
                for c in ("plain", "named", "no_history", "routed")
                if v and c in v[0]
            },
        }
        for k, v in sets.items()
    }
    for k in ("A1", "A2"):
        reading[k]["named_minus_plain"] = paired(
            [r["named"] for r in sets[k]], [r["plain"] for r in sets[k]]
        )
    reading["false"] = {
        k: {
            "n": len(v),
            "plain": rate(r["plain"] for r in v)["rate"],
            "named": rate(r["named"] for r in v)["rate"],
            "minus": paired([r["named"] for r in v], [r["plain"] for r in v]),
        }
        for k, v in false.items()
    }

    if model is not None:
        manifest = json.loads(MANIFEST_PATH.read_bytes())
        model_id = manifest["models"]["base"]["id"]
        ckpt, facts_file = FILES[dataset]
        _, deltas = load(manifest, ckpt, facts_file, model_id)
        by_id = {f["id"]: f for f in facts}

        def serve(q, f):
            j, _ = named.route(q)
            if j is None:
                return None
            note = rag_system("instruct", with_names, j)
            resp = with_adapter(model, deltas[facts[j]["id"]], q, note)
            trim()
            return resp

        served = {}
        for k, q_key in (("A1", "query"), ("C1_probe", "resolved")):
            hits, words = [], []
            for r in sets[k]:
                f = by_id[r["fact"]]
                resp = serve(r[q_key], f)
                hits.append(resp is not None and hit(resp, f["answer"]))
                words.append(resp is not None and word_hit(resp, f["answer"]))
            served[k] = {
                "n": len(hits),
                "substring": rate(hits)["rate"],
                "word": rate(words)["rate"],
            }
            print(dataset, k, served[k], flush=True)
        reading["served_both"] = served
        del deltas
    print(dataset, json.dumps(reading), flush=True)
    return with_names, reading


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    args = ap.parse_args(argv)
    out_path = CONFIRM / ("alias_eval_served.json" if args.serve else "alias_eval.json")
    if out_path.exists():
        ap.error(f"refusing to overwrite {out_path}")
    aliases = json.loads((CONFIRM / "subject_aliases.json").read_text())
    raw_cf = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    raw_zs = json.loads((CONFIRM / "raw/zsre.json").read_text())
    data = {
        d: json.loads((CONFIRM / f).read_text())["facts"] for d, (_, f) in FILES.items()
    }
    used = {f["subject"] for fs in data.values() for f in fs}
    unstored = [f["subject"] for f in build_dev(raw_cf, raw_zs, used)]
    natural = json.loads((CONFIRM / "natural_e3_queries.json").read_text())["queries"]
    rng = random.Random(SEED)
    bench = [prompt(i) for i in rng.sample(mmlu(), 500) + rng.sample(arc(), 500)]

    scorer = make_scorer("relenc", device="cuda" if args.serve else "cpu")
    report = {
        "prereg": "SS105",
        "alias_coverage": {},
        "datasets": {},
    }
    model = None
    if args.serve:
        manifest = json.loads(MANIFEST_PATH.read_bytes())
        model = DeltaStore(manifest["models"]["base"]["id"], steps=16)
        model.deltas = {}
        model.materialize()
    for dataset, facts in data.items():
        names, reading = one_dataset(
            dataset, facts, aliases, raw_cf, unstored, natural, bench, scorer, model
        )
        report["alias_coverage"][dataset] = rate([bool(g["aliases"]) for g in names])
        report["datasets"][dataset] = reading

    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("saved", out_path)


if __name__ == "__main__":
    main()
