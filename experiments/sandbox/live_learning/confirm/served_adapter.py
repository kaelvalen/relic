"""SS130: the SS126 adapters served through the default router.

.venv/bin/python experiments/sandbox/live_learning/confirm/served_adapter.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

import torch  # noqa: E402
from adapter_only import CONFIRM, MODEL, draw_facts, lower_upper  # noqa: E402
from adapter_size import KAPPA  # noqa: E402
from locality_eval import other_relation_prompts  # noqa: E402
from natural_e3 import (  # noqa: E402
    GENERATOR,
    SYSTEM_GEN,
    USER_GEN,
    neighbours,
    parse_question,
    short_description,
)
from wrong_route import paired  # noqa: E402

from relic.adapters import R2, make_variant, set_delta  # noqa: E402
from relic.metrics import word_hit  # noqa: E402
from relic.model import chat, load_model  # noqa: E402
from relic.router import RouterV7, find_subject, make_scorer  # noqa: E402
from relic.serving import SYSTEM, rag_system, trim  # noqa: E402

SEED = 130
THETA = 0.7501
STORES = ("current", "a0_alone", "fallback", "reject_only", "gate_off")
READINGS = ("edit", "paraphrase", "e3", "natural")


def write_questions(facts, path) -> list:
    """Natural other-relation questions as in SS85; on CPU, since the GPU is shared."""
    if path.exists():
        return json.loads(path.read_text())
    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    relations = sorted({r["requested_rewrite"]["relation_id"] for r in raw})
    props = json.loads((CONFIRM / "wikidata/properties_raw.json").read_text())
    labels = {
        b["p"]["value"].rsplit("/", 1)[1]: (
            b["label"]["value"],
            short_description(b.get("desc", {}).get("value", "")),
        )
        for b in props["results"]["bindings"]
    }
    nb = neighbours()
    rng = random.Random(SEED)
    eligible = [i for i, f in enumerate(facts) if f["relation"] in nb]
    near = set(rng.sample(eligible, min(len(facts) // 2, len(eligible))))
    tok = AutoTokenizer.from_pretrained(GENERATOR)
    model = AutoModelForCausalLM.from_pretrained(GENERATOR, dtype=torch.bfloat16)
    model.eval()
    rows = []
    for i, f in enumerate(facts):
        r = f["relation"]
        if i in near:
            kind, rel = "near", rng.choice(sorted(nb[r]))
        else:
            pool = [x for x in relations if x != r and x not in nb[r]]
            kind, rel = "uniform", rng.choice(pool)
        label, desc = labels[rel]
        msgs = [
            {"role": "system", "content": SYSTEM_GEN},
            {
                "role": "user",
                "content": USER_GEN.format(
                    subject=f["subject"], label=label, description=desc
                ),
            },
        ]
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        )
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id
            )
        q = parse_question(tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True))
        rows.append(
            {
                "id": f["id"],
                "kind": kind,
                "relation": rel,
                "query": q,
                "subject_found": find_subject(q, f["subject"]) is not None,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"questions {i + 1}/{len(facts)}", flush=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    del model
    return rows


def store_arms(d0: float, d1: float) -> dict:
    """{store: the adapter it holds for a fact}; above kappa a write commits nothing."""
    a0 = "A0" if d0 <= KAPPA else None
    a1 = "A1" if d1 <= KAPPA else None
    return {
        "current": a0,
        "a0_alone": a0,
        "fallback": a1 or a0,
        "reject_only": a1,
        "gate_off": "A1",
    }


def compare(rows, store):
    def rate(reading, keep=lambda r: True):
        xs = [r["score"][store] for r in rows if r["reading"] == reading and keep(r)]
        return round(float(np.mean(xs)), 4) if xs else None

    out = {
        "edit": rate("edit"),
        "edit_cf": rate("edit", lambda r: r["id"].startswith("cf")),
        "paraphrase": rate("paraphrase"),
        "paraphrase_cf": rate("paraphrase", lambda r: r["id"].startswith("cf")),
        "paraphrase_zs": rate("paraphrase", lambda r: r["id"].startswith("zs")),
        "e3": rate("e3"),
        "natural": rate("natural"),
        "natural_near": rate("natural", lambda r: r.get("kind") == "near"),
        "natural_uniform": rate("natural", lambda r: r.get("kind") == "uniform"),
        "routed": {
            name: round(
                float(
                    np.mean(
                        [
                            r["route"][store] is not None
                            for r in rows
                            if r["reading"] == name
                        ]
                    )
                ),
                4,
            )
            for name in READINGS
        },
        "routed_other_fact": round(
            float(
                np.mean(
                    [
                        r["route"][store] not in (None, r["id"])
                        for r in rows
                        if r["reading"] in ("edit", "paraphrase")
                    ]
                )
            ),
            4,
        ),
    }
    for against in ("current", "a0_alone"):
        out[f"minus_{against}"] = {}
        for name, keep in (
            ("edit", lambda r: r["reading"] == "edit"),
            ("paraphrase", lambda r: r["reading"] == "paraphrase"),
            (
                "paraphrase_cf",
                lambda r: r["reading"] == "paraphrase" and r["id"].startswith("cf"),
            ),
            ("e3", lambda r: r["reading"] == "e3"),
            ("natural", lambda r: r["reading"] == "natural"),
        ):
            sel = [r for r in rows if keep(r)]
            out[f"minus_{against}"][name] = paired(
                [r["score"][store] for r in sel], [r["score"][against] for r in sel]
            )
    return out


def decide(summary, store="fallback") -> dict:
    d = summary[store]["minus_current"]
    rule = {
        "i_paraphrase_pooled": lower_upper(d["paraphrase"])[0] >= -0.03,
        "i_paraphrase_cf": lower_upper(d["paraphrase_cf"])[0] >= -0.03,
        "ii_edit": lower_upper(d["edit"])[0] >= -0.02,
        "iii_e3": lower_upper(d["e3"])[1] <= 0.02,
        "iv_natural": lower_upper(d["natural"])[1] <= 0.02,
    }
    rule["holds"] = all(rule.values())
    return rule


def build_queries(sets, facts, cf_records, raw_cf, natural) -> list:
    queries = [
        {"reading": r, "id": i, "q": facts[i][field]}
        for r, field in (("edit", "probe"), ("paraphrase", "paraphrase"))
        for i in facts
    ]
    queries += [
        {"reading": "e3", "id": f["id"], "q": p["query"]}
        for f, p in zip(sets["cf"], other_relation_prompts(cf_records, raw_cf))
    ]
    queries += [
        {"reading": "natural", "id": n["id"], "q": n["query"], "kind": n["kind"]}
        for n in natural
        if n["subject_found"]
    ]
    return queries


def serve(queries, facts, key, arms, deltas, stores):
    """Each query through each store's router; only the current store adds the note."""
    scorer = make_scorer("relenc", device="cpu")
    routers = {}
    for store in stores:
        kept = [f for i, f in facts.items() if arms[i][store] is not None]
        routers[store] = (RouterV7(kept, scorer, THETA), kept)

    tok, base = load_model(MODEL)
    m = make_variant(base, R2)
    m.eval()

    def ask(q, system):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
        return chat(tok, m, msgs, max_new_tokens=64)

    base_cache, cache = {}, {}

    def base_output(q):
        if q not in base_cache:
            with m.disable_adapter():
                base_cache[q] = ask(q, SYSTEM)
        return base_cache[q]

    def adapter_output(arm, fid, q, note):
        k = (arm, fid, q, note)
        if k not in cache:
            set_delta(m, deltas[arm][key[fid]])
            system = rag_system("instruct", [facts[fid]], 0) if note else SYSTEM
            cache[k] = ask(q, system)
        return cache[k]

    rows = []
    for n, row in enumerate(queries):
        q, answer = row["q"], facts[row["id"]]["answer"]
        row["base"] = base_output(q)
        row["route"], row["response"], row["score"] = {}, {}, {}
        for store in stores:
            router, kept = routers[store]
            j, _ = router.route(q)
            fid = None if j is None else kept[j]["id"]
            row["route"][store] = fid
            row["response"][store] = (
                row["base"]
                if fid is None
                else adapter_output(arms[fid][store], fid, q, store == "current")
            )
            got = word_hit(row["response"][store], answer)
            row["score"][store] = (
                got
                if row["reading"] in ("edit", "paraphrase")
                else got and not word_hit(row["base"], answer)
            )
        rows.append(row)
        trim()
        if (n + 1) % 100 == 0:
            print(f"served {n + 1}/{len(queries)}", flush=True)
    return rows, {s: len(routers[s][1]) for s in stores}, len(base_cache), len(cache)


def matches_forced(rows, a1_rows, key) -> float:
    """The served path is the forced path once the route is right."""
    same = [
        r["response"]["gate_off"]
        == a1_rows[key[r["id"]]]["outputs"][
            "probe" if r["reading"] == "edit" else "paraphrase"
        ]
        for r in rows
        if r["reading"] in ("edit", "paraphrase") and r["route"]["gate_off"] == r["id"]
    ]
    return round(float(np.mean(same)), 4)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="4 facts, the SS126 smoke run")
    args = ap.parse_args(argv)
    n_per = 2 if args.smoke else 150
    source = "adapter_only_smoke" if args.smoke else "adapter_only"
    stem = "served_adapter_smoke" if args.smoke else "served_adapter"
    out = CONFIRM / f"{stem}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    saved = json.loads((CONFIRM / f"{source}.json").read_text())
    sets, cf_records, raw_cf = draw_facts(n_per)
    facts = {f["id"]: f for fs in sets.values() for f in fs}
    key = {f["id"]: f"{ds}:{f['id']}" for ds, fs in sets.items() for f in fs}
    natural = write_questions(sets["cf"], CONFIRM / f"{stem}_questions.json")
    queries = build_queries(sets, facts, cf_records, raw_cf, natural)
    arms = {
        i: store_arms(
            saved["arms"]["A0"]["rows"][key[i]]["d_heldout"],
            saved["arms"]["A1"]["rows"][key[i]]["d_heldout"],
        )
        for i in facts
    }
    deltas = {
        arm: torch.load(
            CONFIRM / f"{source}_{arm}.pt", map_location="cpu", weights_only=False
        )
        for arm in ("A0", "A1")
    }
    rows, sizes, n_base, n_adapter = serve(queries, facts, key, arms, deltas, STORES)

    summary = {store: compare(rows, store) for store in STORES}
    d = {
        arm: [saved["arms"][arm]["rows"][key[i]]["d_heldout"] for i in facts]
        for arm in ("A0", "A1")
    }
    report = {
        "prereg": "SS130",
        "theta": THETA,
        "kappa": KAPPA,
        "above_kappa": {arm: sum(not v <= KAPPA for v in d[arm]) for arm in d},
        "store_size": sizes,
        "generated": {"base": n_base, "adapter": n_adapter},
        "summary": summary,
        "rule": decide(summary),
        "rows": rows,
    }
    report["matches_forced"] = matches_forced(rows, saved["arms"]["A1"]["rows"], key)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    print(json.dumps(report["rule"], indent=1), "matches", report["matches_forced"])
    assert report["matches_forced"] >= 0.99, "served and forced outputs differ"


if __name__ == "__main__":
    main()
