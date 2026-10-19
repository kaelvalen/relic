"""SS85 (SS80 pinned): natural other-relation queries written by another model family.

generate: for each fact of the CounterFact N=1000 store, SmolLM3-3B writes one natural
question about the subject asking for another relation R' (half near neighbours of the
stored relation, half uniform over the others). serve: the questions are routed by the
default router (SS76 encoder), the SS74 default (MiniLM) and the frozen mpnet, served
with the stored adapters and notes; a label sample is drawn before any rate.

    .venv/bin/python experiments/sandbox/live_learning/confirm/natural_e3.py generate
    .venv/bin/python experiments/sandbox/live_learning/confirm/natural_e3.py serve
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

from routed_eval import (  # noqa: E402
    MANIFEST_PATH,
    REPO_ROOT,
    _repo_path,
    verify_manifest_inputs,
)
from router_v4 import find_subject  # noqa: E402

SEED = 80
N_NEAR = 500
N_LABEL = 40
GENERATOR = "HuggingFaceTB/SmolLM3-3B"
CONFIRM = Path("results/live_learning/confirm")
QUERIES = CONFIRM / "natural_e3_queries.json"
NEIGHBOUR_GROUPS = [
    ["P495", "P740"],
    ["P176", "P178"],
    ["P17", "P30", "P131"],
    ["P364", "P407"],
    ["P103", "P1412", "P37"],
    ["P276", "P131"],
    ["P19", "P20", "P937"],
]
ROUTERS = {  # name -> (scorer, dev theta)
    "default_ss76": ("relenc", 0.7501),
    "ss74_minilm": ("minilm", 0.757),
    "frozen_mpnet": ("mpnet", 0.6694),
}
SYSTEM_GEN = (
    "/no_think You write exactly one natural English question and nothing else."
)
USER_GEN = (
    "Write one natural question about {subject} that asks for its {label} "
    "({description}). Mention {subject} by name. Reply with the question only."
)


def neighbours():
    nb = defaultdict(set)
    for g in NEIGHBOUR_GROUPS:
        for r in g:
            nb[r] |= set(g) - {r}
    return nb


def short_description(desc: str) -> str:
    """Cut at the first ';' or '. ' that does not end an "e.g." or "i.e." (SS85 deviation)."""
    for m in re.finditer(r";|\. ", desc):
        if m.group(0) == ";" or not desc[: m.start()].endswith(("e.g", "i.e")):
            return desc[: m.start()]
    return desc


def load_facts():
    manifest = json.loads(MANIFEST_PATH.read_bytes())
    model_id = manifest["models"]["base"]["id"]
    verified = verify_manifest_inputs(
        manifest,
        _repo_path(str(CONFIRM / "cf_n1000.ckpt"), REPO_ROOT),
        _repo_path(str(CONFIRM / "external_counterfact_n1000.json"), REPO_ROOT),
        model_id,
        repo_root=REPO_ROOT,
    )
    return model_id, verified


def assign_relations(facts, relations, rng):
    nb = neighbours()
    eligible = [i for i, f in enumerate(facts) if f["relation"] in nb]
    near = set(rng.sample(eligible, N_NEAR))
    out = []
    for i, f in enumerate(facts):
        r = f["relation"]
        if i in near:
            out.append(("near", rng.choice(sorted(nb[r]))))
        else:
            pool = [x for x in relations if x != r and x not in nb[r]]
            out.append(("uniform", rng.choice(pool)))
    return out, len(eligible)


def parse_question(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return line.strip("\"'“”‘’ ")


def generate(args):
    if QUERIES.exists():
        raise SystemExit(f"refusing to overwrite {QUERIES}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _, verified = load_facts()
    facts = verified["facts"]
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
    rng = random.Random(SEED)
    targets, n_eligible = assign_relations(facts, relations, rng)

    tok = AutoTokenizer.from_pretrained(GENERATOR)
    model = AutoModelForCausalLM.from_pretrained(
        GENERATOR, dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    rows, t0 = [], time.time()
    for i, (f, (kind, rel)) in enumerate(zip(facts, targets)):
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
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id
            )
        text = tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
        q = parse_question(text)
        rows.append(
            {
                "fact_id": f["id"],
                "index": i,
                "subject": f["subject"],
                "relation": f["relation"],
                "query_relation": rel,
                "kind": kind,
                "raw": text,
                "query": q,
                "subject_found": find_subject(q, f["subject"]) is not None,
            }
        )
        if (i + 1) % 100 == 0:
            print(
                f"generated {i + 1}/{len(facts)} ({time.time() - t0:.0f}s)", flush=True
            )
    kept = sum(r["subject_found"] for r in rows)
    QUERIES.write_text(
        json.dumps(
            {
                "prereg": "SS85",
                "generator": GENERATOR,
                "system": SYSTEM_GEN,
                "user": USER_GEN,
                "n_eligible_near": n_eligible,
                "kept": kept,
                "dropped": len(rows) - kept,
                "queries": rows,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"saved {QUERIES}: kept {kept}/{len(rows)}", flush=True)


def rate(xs):
    xs = list(xs)
    return round(float(np.mean(xs)), 4) if xs else None


def serve(args):
    out_path = CONFIRM / "natural_e3.json"
    label_path = CONFIRM / "natural_e3_labels.json"
    if out_path.exists() or label_path.exists():
        raise SystemExit("refusing to overwrite natural_e3 outputs")
    import torch
    from facts import NUMBERS
    from pilot import CANARIES, PilotStore, hit, trim
    from routed_eval import _validate_checkpoint, load_verified_checkpoint
    from router_v7 import RouterV7, make_scorer
    from serving_locality import POLICIES
    from serving_locality import serve as serve_policies
    from v4_eval import Server

    model_id, verified = load_facts()
    facts = verified["facts"]
    qs = [q for q in json.loads(QUERIES.read_text())["queries"] if q["subject_found"]]
    tiers = {
        r["fact_id"]: r["tier"]
        for r in json.loads((CONFIRM / "cf_n1000_v4_eval.json").read_text())[
            "examples"
        ]["serving"]
    }
    torch.manual_seed(0)
    store = PilotStore(model_id, steps=16)
    store.kl_prompts = list(CANARIES)
    checkpoint, _ = load_verified_checkpoint(
        verified["checkpoint_path"],
        verified["checkpoint_sha256"],
        lambda source: torch.load(source, map_location="cpu", weights_only=False),
    )
    _validate_checkpoint(checkpoint, facts)
    store.deltas = checkpoint["deltas"]
    store.keys = checkpoint["keys"]
    store.materialize()
    del checkpoint
    trim()
    server = Server(store)
    answers = [f["answer"] for f in facts if f["answer"] not in NUMBERS]

    rows = {}
    for name, (scorer_name, theta) in ROUTERS.items():
        router = RouterV7(facts, make_scorer(scorer_name), theta)
        rs = []
        for n, q in enumerate(qs):
            f = facts[q["index"]]
            base = server.base(q["query"])
            j, score = router.route(q["query"])
            resp = serve_policies(server, facts, j, q["query"], tiers)
            row = {
                "fact_id": f["id"],
                "kind": q["kind"],
                "query": q["query"],
                "routed": j is not None,
                "route_fact_id": None if j is None else facts[j]["id"],
                "score": score,
                "base": base,
            }
            for p in POLICIES:
                row[p] = {
                    "leak_target": hit(resp[p], f["answer"])
                    and not hit(base, f["answer"]),
                    "leak_any": any(
                        hit(resp[p], w) and not hit(base, w) for w in answers
                    ),
                    "response": resp[p],
                }
            rs.append(row)
            trim()
            if (n + 1) % 200 == 0:
                print(f"{name}: {n + 1}/{len(qs)}", flush=True)
        rows[name] = rs

    # label sample before any rate is computed
    rng = random.Random(SEED)
    sample = []
    for name, rs in rows.items():
        routed = [r for r in rs if r["routed"]]
        for r in rng.sample(routed, min(N_LABEL, len(routed))):
            f = next(x for x in facts if x["id"] == r["route_fact_id"])
            sample.append(
                {
                    "router": name,
                    "question": r["query"],
                    "stored_fact": f["teach"],
                    "stored_target": f["answer"],
                    "response_adapter_note": r["both"]["response"],
                    "base_response": r["base"],
                    "target_correct_under_counterfactual": None,
                }
            )
    label_path.write_text(json.dumps(sample, ensure_ascii=False, indent=1))
    print(f"label sample: {len(sample)} -> {label_path}", flush=True)

    summary = {}
    for name, rs in rows.items():
        summary[name] = {}
        for part in ("pooled", "near", "uniform"):
            sub = [r for r in rs if part == "pooled" or r["kind"] == part]
            summary[name][part] = {
                "n": len(sub),
                "routed": rate(r["routed"] for r in sub),
                **{
                    f"leak_target_{p}": rate(r[p]["leak_target"] for r in sub)
                    for p in POLICIES
                },
                **{
                    f"leak_any_{p}": rate(r[p]["leak_any"] for r in sub)
                    for p in POLICIES
                },
            }
        print(name, json.dumps(summary[name]), flush=True)
    from locality_eval import paired_diff_ci

    paired = paired_diff_ci(
        [r["routed"] for r in rows["default_ss76"]],
        [r["routed"] for r in rows["frozen_mpnet"]],
        n=10000,
    )
    d = summary["default_ss76"]
    report = {
        "prereg": "SS85",
        "n_queries": len(qs),
        "summary": summary,
        "default_minus_frozen_routed": paired,
        "P1": {"routed": d["pooled"]["routed"], "holds": d["pooled"]["routed"] >= 0.07},
        "P2": {"diff": paired, "holds": paired["diff"] > 0},
        "P3": {
            "near_minus_uniform": d["near"]["routed"] - d["uniform"]["routed"],
            "holds": d["near"]["routed"] - d["uniform"]["routed"] >= 0.05,
        },
        "P4": {
            "leak_both": d["pooled"]["leak_target_both"],
            "template_e3_leak": 0.043,
            "holds": d["pooled"]["leak_target_both"] >= 0.043,
        },
        "rows": rows,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("rows", "summary")}, indent=1
        )
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "serve"])
    args = ap.parse_args(argv)
    generate(args) if args.cmd == "generate" else serve(args)


if __name__ == "__main__":
    main()
