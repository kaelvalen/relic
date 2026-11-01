"""SS101: general capability on MMLU and ARC-Challenge with the store active (eval-only).

.venv/bin/python experiments/sandbox/live_learning/confirm/general_eval.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from routed_eval import MANIFEST_PATH  # noqa: E402
from wrong_route import load, paired, with_adapter  # noqa: E402

from relic.ledger import DeltaStore  # noqa: E402
from relic.router import RouterV7, make_scorer  # noqa: E402
from relic.serving import SYSTEM, base_answer, rag_system, trim  # noqa: E402

SEED = 101
N = 500
MARGIN = -0.01
THETA = 0.7501
CONFIRM = Path("results/live_learning/confirm")
LETTERS = "ABCDEFGH"


def mmlu():
    rows = pq.read_table(CONFIRM / "raw/mmlu_test.parquet").to_pylist()
    return [
        {
            "benchmark": "mmlu",
            "id": f"mmlu{i}",
            "subject": r["subject"],
            "question": r["question"],
            "options": list(r["choices"]),
            "gold": LETTERS[r["answer"]],
        }
        for i, r in enumerate(rows)
    ]


def arc():
    rows = pq.read_table(CONFIRM / "raw/arc_challenge_test.parquet").to_pylist()
    out = []
    for r in rows:
        labels = list(r["choices"]["label"])
        out.append(
            {
                "benchmark": "arc",
                "id": r["id"],
                "subject": "science",
                "question": r["question"],
                "options": list(r["choices"]["text"]),
                # a few ARC items label their options 1 to 4
                "gold": LETTERS[labels.index(r["answerKey"])],
            }
        )
    return out


def prompt(item) -> str:
    options = "\n".join(
        f"{LETTERS[i]}. {text}" for i, text in enumerate(item["options"])
    )
    return f"{item['question']}\n{options}\nAnswer with the letter only."


def letter(resp: str, n_options: int) -> str | None:
    found = re.search(rf"\b([{LETTERS[:n_options]}])\b", resp)
    return found.group(1) if found else None


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    from scipy.stats import beta

    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [round(low, 4), round(high, 4)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CONFIRM / "general_eval.json"))
    args = ap.parse_args(argv)
    if Path(args.out).exists():
        ap.error(f"refusing to overwrite {args.out}")
    rng = random.Random(SEED)
    items = rng.sample(mmlu(), N) + rng.sample(arc(), N)

    manifest = json.loads(MANIFEST_PATH.read_bytes())
    model_id = manifest["models"]["base"]["id"]
    cf_file, zs_file = "external_counterfact_n1000.json", "external_zsre_n1000.json"
    cf, cf16 = load(manifest, "cf_n1000.ckpt", cf_file, model_id)
    _, cf2 = load(manifest, "cf_n1000_r2.ckpt", cf_file, model_id)
    zs, zs16 = load(manifest, "zsre_n1000.ckpt", zs_file, model_id)
    stores = {"counterfact": (cf, cf16), "zsre": (zs, zs16)}

    store = DeltaStore(model_id, steps=16)
    store.deltas = {}
    store.materialize()
    scorer = make_scorer("relenc")
    routers = {name: RouterV7(fs, scorer, THETA) for name, (fs, _) in stores.items()}

    rows = []
    for n, item in enumerate(items):
        q = prompt(item)
        k = len(item["options"])
        base = base_answer(store, q)
        row = {
            **item,
            "base": base,
            "letters": {"base": letter(base, k)},
            "routes": {},
            "responses": {},
        }
        for name, (fs, deltas) in stores.items():
            j, score = routers[name].route(q)
            row["routes"][name] = {
                "entry": None if j is None else fs[j]["id"],
                "score": round(score, 4),
            }
            if j is None:
                row["letters"][f"store_{name}"] = row["letters"]["base"]
                continue
            note = rag_system("instruct", fs, j)
            resp = with_adapter(store, deltas[fs[j]["id"]], q, note)
            row["responses"][f"store_{name}"] = resp
            row["letters"][f"store_{name}"] = letter(resp, k)
        i = rng.randrange(len(cf))
        wrong, note = cf[i]["id"], rag_system("instruct", cf, i)
        forced = {
            "forced_adapter_r16": with_adapter(store, cf16[wrong], q, SYSTEM),
            "forced_adapter_r2": with_adapter(store, cf2[wrong], q, SYSTEM),
            "forced_note": base_answer(store, q, system=note),
            "forced_both_r2": with_adapter(store, cf2[wrong], q, note),
        }
        row["responses"].update(forced)
        row["letters"].update({a: letter(r, k) for a, r in forced.items()})
        row["correct"] = {a: v == item["gold"] for a, v in row["letters"].items()}
        rows.append(row)
        trim()
        if (n + 1) % 100 == 0:
            print(f"general: {n + 1}/{len(items)}", flush=True)

    def part(name):
        return [r for r in rows if name == "pooled" or r["benchmark"] == name]

    arms = list(rows[0]["correct"])
    report = {"prereg": "SS101", "n": len(rows), "margin": MARGIN, "readings": {}}
    for name in ("mmlu", "arc", "pooled"):
        sub = part(name)
        base = [r["correct"]["base"] for r in sub]
        reading = {
            "n": len(sub),
            "accuracy": {
                a: round(float(np.mean([r["correct"][a] for r in sub])), 4)
                for a in arms
            },
            "no_letter": {
                a: round(float(np.mean([r["letters"][a] is None for r in sub])), 4)
                for a in arms
            },
            "minus_base": {
                a: paired([r["correct"][a] for r in sub], base)
                for a in arms
                if a != "base"
            },
        }
        for store_name in stores:
            k = sum(r["routes"][store_name]["entry"] is not None for r in sub)
            reading[f"routed_{store_name}"] = {
                "k": k,
                "rate": round(k / len(sub), 4),
                "ci95": clopper_pearson(k, len(sub)),
            }
        report["readings"][name] = reading
    pooled = report["readings"]["pooled"]["minus_base"]
    report["noninferior"] = all(
        pooled[f"store_{s}"]["lower_95_one_sided"] >= MARGIN for s in stores
    )
    report["routed_questions"] = [
        {
            "id": r["id"],
            "question": r["question"],
            "routes": r["routes"],
            "correct": r["correct"],
        }
        for r in rows
        if any(v["entry"] is not None for v in r["routes"].values())
    ]
    report["rows"] = rows
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    torch.manual_seed(SEED)
    main()
