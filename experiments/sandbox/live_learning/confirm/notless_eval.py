"""SS145: the readings of the main tables on the default store, built by `Ledger.write`.

.venv/bin/python experiments/sandbox/live_learning/confirm/notless_eval.py --dataset counterfact [--smoke]
.venv/bin/python experiments/sandbox/live_learning/confirm/notless_eval.py --report
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
from alias_eval import pronoun_form  # noqa: E402
from general_eval import letter, prompt  # noqa: E402
from locality_eval import match_raw, other_relation_prompts  # noqa: E402
from wrong_route import paired  # noqa: E402

from relic.adapters import pad  # noqa: E402
from relic.ledger import DeltaStore, Ledger  # noqa: E402
from relic.metrics import hit, word_hit  # noqa: E402
from relic.router import RELATION_TAU, RouterV7, make_scorer  # noqa: E402
from relic.serving import SYSTEM, base_answer, serve_expert, trim  # noqa: E402

SEED = 145
THETA, SLOT = 0.7501, 0.8985
PART = 50
OLD_TOKENS = 24  # the saved evaluations of the main stores generated this many
FILES = {
    "counterfact": ("external_counterfact_n1000.json", "cf"),
    "zsre": ("external_zsre_n1000.json", "zsre"),
}


def narrow(delta: dict, r: int = 2) -> dict:
    """Rank 2 again, as the padding of a stored adapter holds only zeros."""
    return {
        n: {"A": ab["A"][:r].clone(), "B": ab["B"][:, :r].clone()}
        for n, ab in delta.items()
    }


def build(ledger, facts, parts: Path) -> tuple[dict, list]:
    """Each fact through `Ledger.write`; a part on disk every 50 facts, as the GPU is shared."""
    parts.mkdir(exist_ok=True)
    holder, events = {}, []

    def account(f, event):
        if event["action"] == "update":
            holder.update(
                {
                    k: event["entry"]
                    for k, v in holder.items()
                    if v == event["superseded"]
                }
            )
        holder[f["id"]] = event["entry"] if event["action"] != "rejected" else None
        events.append({"fact_id": f["id"], **event})

    done = 0
    for path in sorted(parts.glob("part_*.pt")):
        for f, event, delta in torch.load(path, map_location="cpu", weights_only=False):
            if event["action"] in ("insert", "update"):
                ledger.commit(f, pad(delta))
            elif event["action"] == "duplicate":
                ledger.commit(f)
            account(f, event)
            done += 1
    print(f"resumed {done} writes", flush=True)
    batch = []
    for n, f in enumerate(facts[done:], done + 1):
        t0 = time.time()
        event = ledger.write(f)
        event["seconds"] = round(time.time() - t0, 2)
        account(f, event)
        stored = event["action"] in ("insert", "update")
        delta = narrow(ledger.store.deltas[event["entry"]]) if stored else None
        batch.append((f, event, delta))
        trim()
        if n % PART == 0 or n == len(facts):
            path = parts / f"part_{n:05d}.pt"
            tmp = path.with_suffix(".tmp")
            torch.save(batch, tmp)
            tmp.replace(path)
            batch = []
            print(f"written {n}/{len(facts)}", flush=True)
    return holder, events


def main_dataset(dataset: str, smoke: bool) -> None:
    file, tag = FILES[dataset]
    suffix = "_smoke" if smoke else ""
    out = CONFIRM / f"notless_eval_{tag}{suffix}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    facts = json.loads((CONFIRM / file).read_text())["facts"]
    raw = json.loads((CONFIRM / f"raw/{dataset}.json").read_text())
    n_q = 6 if smoke else None
    facts = facts[:n_q] if smoke else facts

    store = DeltaStore(MODEL, steps=16, capacity=1)
    scorer = make_scorer("relenc")
    ledger = Ledger(store, scorer, THETA, SLOT)
    holder, events = build(ledger, facts, CONFIRM / f"notless_parts_{tag}{suffix}")
    report = {
        "prereg": "SS145",
        "dataset": dataset,
        "events": events,
        "entries": len(ledger.entries),
    }

    router, active = ledger.router
    extra = {f"{f['subject']}\t{f['probe']}": ledger.phrasings[f["id"]] for f in active}
    tau_router = RouterV7(active, scorer, THETA, extra, RELATION_TAU)
    base_cache, answers = {}, {}

    def base(q):
        if q not in base_cache:
            base_cache[q] = base_answer(store, q)
        return base_cache[q]

    def serve(fid, q):
        if (fid, q) not in answers:
            answers[(fid, q)] = serve_expert(store, fid, q)
        return answers[(fid, q)]

    def read(reading, q, answer, fid, **fields):
        """Served under one threshold; the relation thresholds are read off the route."""
        j, score = router.route(q)
        route = None if j is None else active[j]["id"]
        jt, _ = tau_router.route(q)
        text = base(q) if route is None else serve(route, q)
        trim()
        return {
            "reading": reading,
            "id": fid,
            "q": q,
            "answer": answer,
            "route": route,
            "route_tau": None if jt is None else active[jt]["id"],
            "score": score,
            "base": base(q),
            "response": text,
            **fields,
        }

    rows = []
    for f in facts:
        for reading, key in (("edit", "probe"), ("paraphrase", "paraphrase")):
            rows.append(
                read(reading, f[key], f["answer"], f["id"], holder=holder[f["id"]])
            )
        rows.append(read("distractor", f["distractor"], f["answer"], f["id"]))
        p = pronoun_form(f["probe"], f["subject"])
        if p:
            q = ledger.ask(p, history=[f"Tell me about {f['subject']}."])
            rows.append(
                {
                    "reading": "pronoun",
                    "id": f["id"],
                    "q": p,
                    "answer": f["answer"],
                    "route": q["entry"],
                    "route_tau": q["entry"],
                    "response": q["answer"],
                    "base": "",
                    "score": q["score"],
                }
            )
    print("own questions read", flush=True)
    if dataset == "counterfact":
        records = match_raw(facts, raw)
        e3 = other_relation_prompts(records, raw)
        for f, p, r in zip(facts, e3, records):
            rows.append(read("e3", p["query"], f["answer"], f["id"]))
            rows.append(read("e2", r["neighborhood_prompts"][0], f["answer"], f["id"]))
        natural = json.loads((CONFIRM / "natural_e3.json").read_text())["rows"][
            "default_ss76"
        ]
        by_id = {f["id"]: f for f in facts}
        for r in natural:
            if r["fact_id"] in by_id:
                answer = by_id[r["fact_id"]]["answer"]
                rows.append(
                    read("natural", r["query"], answer, r["fact_id"], kind=r["kind"])
                )
        report["wrong_route"] = wrong_route(store, holder, n_q)
        report["general"] = general(store, router, active, n_q)
    else:
        by_src = {item["src"].strip(): item for item in raw}
        for f in facts:
            rows.append(read("loc", by_src[f["probe"]]["loc"], f["answer"], f["id"]))
    report["rows"] = rows
    print("readings done", flush=True)

    removed = facts[:100] if not smoke else facts[:2]
    gone_entries = {holder[f["id"]] for f in removed if holder[f["id"]]}
    pre = {f["id"]: base(f["probe"]) for f in removed}
    t0 = time.time()
    for fid in sorted(gone_entries):
        ledger.revoke(fid)
    seconds = time.time() - t0
    after, active_after = ledger.router

    def after_removal(q, unrouted):
        j, _ = after.route(q)
        return (
            unrouted if j is None else serve_expert(store, active_after[j]["id"], q)
        ), j

    removal = []
    for f in removed:
        text, j = after_removal(f["probe"], pre[f["id"]])
        removal.append(
            {
                "fact_id": f["id"],
                "routed_to": None if j is None else active_after[j]["id"],
                "gone": not hit(text, f["answer"]),
                "return": hit(text, f["answer"]) == hit(pre[f["id"]], f["answer"]),
                "same_text": text == pre[f["id"]],
                "base_hit": hit(pre[f["id"]], f["answer"]),
            }
        )
    kept = [f for f in facts if holder[f["id"]] and holder[f["id"]] not in gone_entries]
    retained = []
    for f in kept:
        text, _ = after_removal(f["probe"], None)
        retained.append(hit(base(f["probe"]) if text is None else text, f["answer"]))
        trim()
    report["removal"] = {
        "rows": removal,
        "retention_n": len(retained),
        "retention": round(float(np.mean(retained)), 4) if retained else None,
        "seconds_per_removal": round(seconds / max(len(gone_entries), 1), 4),
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(summarize(report), indent=1), flush=True)


def forced(store, deltas, fid, q) -> str:
    store.deltas = {fid: deltas[fid]}
    store.materialize()
    return store.answer(q, system=SYSTEM)


def wrong_route(store, holder, n_q) -> list:
    saved = json.loads((CONFIRM / "wrong_route.json").read_text())["rows"]
    deltas = store.deltas
    out = []
    for r in saved[:n_q] if n_q else saved:
        fid = holder.get(r["adapter"])
        if (
            n_q and fid is None
        ):  # shortcut: the smoke run stores few facts, never the full run
            fid = next(iter(deltas))
        if fid not in deltas:
            continue  # no adapter holds this fact here, so nothing is forced
        text = forced(store, deltas, fid, r["question"])
        out.append(
            {
                "question": r["question"],
                "answer": r["answer"],
                "adapter": fid,
                "response": text,
                "word_base": r["word_base"],
                "saved": r["retained"],
                "saved_word": r["retained_word"],
            }
        )
        trim()
    store.deltas = deltas
    store.materialize()
    print("wrong route read", flush=True)
    return out


def general(store, router, active, n_q) -> list:
    saved = json.loads((CONFIRM / "general_eval.json").read_text())["rows"]
    rng = random.Random(SEED)
    deltas = store.deltas
    out = []
    for r in saved[:n_q] if n_q else saved:
        q = prompt(r)
        j, _ = router.route(q)
        text = forced(store, deltas, rng.choice(active)["id"], q)
        out.append(
            {
                "id": r["id"],
                "routed": j is not None,
                "forced_correct": letter(text, len(r["options"])) == r["gold"],
                "base_correct": r["correct"]["base"],
                "forced_both_r2": r["correct"]["forced_both_r2"],
                "response": text,
            }
        )
        trim()
    store.deltas = deltas
    store.materialize()
    print("general read", flush=True)
    return out


def rate(xs) -> float | None:
    xs = list(xs)
    return round(float(np.mean(xs)), 4) if xs else None


def summarize(report: dict, cut=lambda text: text) -> dict:
    """`cut` shortens a response to the length the saved evaluations generated."""
    rows, out = report["rows"], {}
    for route in ("route", "route_tau"):

        def text(r):
            # a raised threshold only sends a query to the base model
            return r["response"] if r[route] == r["route"] else r["base"]

        def leak(xs):
            return rate(
                word_hit(text(r), r["answer"]) and not word_hit(r["base"], r["answer"])
                for r in xs
            )

        s = {}
        for name in ("edit", "paraphrase", "pronoun"):
            xs = [r for r in rows if r["reading"] == name]
            s[name] = {
                "n": len(xs),
                "word": rate(word_hit(text(r), r["answer"]) for r in xs),
                "substring": rate(hit(text(r), r["answer"]) for r in xs),
                "routed": rate(r[route] is not None for r in xs),
            }
        for name in ("e3", "e2", "natural"):
            xs = [r for r in rows if r["reading"] == name]
            s[name] = {
                "n": len(xs),
                "leak": leak(xs),
                "routed": rate(r[route] is not None for r in xs),
            }
        for kind in ("near", "uniform"):
            xs = [
                r for r in rows if r["reading"] == "natural" and r.get("kind") == kind
            ]
            s[f"natural_{kind}"] = {
                "n": len(xs),
                "leak": leak(xs),
                "routed": rate(r[route] is not None for r in xs),
            }
        for name in ("distractor", "loc"):
            xs = [r for r in rows if r["reading"] == name]
            s[name] = {
                "n": len(xs),
                "routed": rate(r[route] is not None for r in xs),
                "same_as_base": rate(text(r) == r["base"] for r in xs),
            }
        out[route] = s
    ev = report["events"]
    rungs = [e.get("rung") for e in ev if e.get("rung")]
    out["write"] = {
        "n": len(ev),
        "entries": report["entries"],
        "actions": {
            a: sum(e["action"] == a for e in ev)
            for a in ("insert", "duplicate", "update", "rejected")
        },
        "rungs": {r: rungs.count(r) for r in sorted(set(rungs))},
        "seconds_mean": rate(e["seconds"] for e in ev if e["action"] != "duplicate"),
    }
    rem = report["removal"]
    out["removal"] = {
        "gone": rate(r["gone"] for r in rem["rows"]),
        "return": rate(r["return"] for r in rem["rows"]),
        "same_text": rate(r["same_text"] for r in rem["rows"]),
        "base_hit": rate(r["base_hit"] for r in rem["rows"]),
        "retention": rem["retention"],
        "retention_n": rem["retention_n"],
        "seconds_per_removal": rem["seconds_per_removal"],
    }
    if "wrong_route" in report:
        w, g = report["wrong_route"], report["general"]
        kept = [hit(r["response"], r["answer"]) for r in w]
        kept_old = [hit(cut(r["response"]), r["answer"]) for r in w]
        out["wrong_route"] = {
            "n": len(w),
            "retained_64_tokens": rate(kept),
            "retained_24_tokens": rate(kept_old),
            "both_r2_24_tokens": rate(r["saved"]["both_r2"] for r in w),
            "adapter_r2_24_tokens": rate(r["saved"]["adapter_r2"] for r in w),
            "minus_both_r2": paired(kept_old, [r["saved"]["both_r2"] for r in w]),
            "minus_adapter_r2": paired(kept_old, [r["saved"]["adapter_r2"] for r in w]),
        }
        out["general"] = {
            "n": len(g),
            "routed": sum(r["routed"] for r in g),
            "forced": rate(r["forced_correct"] for r in g),
            "base": rate(r["base_correct"] for r in g),
            "forced_minus_base": paired(
                [r["forced_correct"] for r in g], [r["base_correct"] for r in g]
            ),
            "forced_both_r2": rate(r["forced_both_r2"] for r in g),
        }
    return out


def final_report() -> None:
    from transformers import AutoTokenizer

    out = CONFIRM / "notless_eval.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    tok = AutoTokenizer.from_pretrained(MODEL)

    def cut(text: str) -> str:
        ids = tok(text, add_special_tokens=False)["input_ids"][:OLD_TOKENS]
        return tok.decode(ids, skip_special_tokens=True)

    reports = {
        t: json.loads((CONFIRM / f"notless_eval_{t}.json").read_text())
        for t in ("cf", "zsre")
    }
    s = {t: summarize(r, cut) for t, r in reports.items()}
    paired_note = {}
    for t, r in reports.items():
        saved = json.loads((CONFIRM / f"{t}_n1000_r2_v8_eval.json").read_text())
        saved = {x["fact_id"]: x for x in saved["examples"]["serving"]}
        for reading, key in (("edit", "probe"), ("paraphrase", "paraphrase")):
            xs = [x for x in r["rows"] if x["reading"] == reading]
            new = [word_hit(cut(x["response"]), x["answer"]) for x in xs]
            old = [
                word_hit(saved[x["id"]][key]["responses"]["both"], x["answer"])
                for x in xs
            ]
            paired_note[f"{t}_{reading}"] = {
                "no_note": rate(new),
                "both_r2": rate(old),
                "minus": paired(new, old),
            }
    w = s["cf"]["wrong_route"]["minus_both_r2"]
    rule = {
        "i_retention": all(s[t]["removal"]["retention"] >= 0.98 for t in s),
        "ii_return": all(s[t]["removal"]["return"] >= 0.98 for t in s),
        "iii_distractor": all(
            s[t]["route_tau"]["distractor"]["routed"] == 0 for t in s
        ),
        "iii_general": s["cf"]["general"]["routed"] <= 1,
        "iv_e3": s["cf"]["route_tau"]["e3"]["leak"] <= 0.06,
        "v_wrong_route": w["lower_95_one_sided"] > 0,
    }
    report = {
        "prereg": "SS145",
        "summary": s,
        "paired_with_note": paired_note,
        "rule": rule,
    }
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(FILES))
    ap.add_argument("--smoke", action="store_true", help="6 facts and 6 questions")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        final_report()
    else:
        main_dataset(args.dataset, args.smoke)
