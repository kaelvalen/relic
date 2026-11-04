"""SS102: the human label sheet for the natural-question sample, and its scoring.

    python label_sheet.py make    # writes the sheet; fill in its `label` column
    python label_sheet.py score   # agreement with the model labels
    python label_sheet.py thresholds   # SS141: what the per-relation arms stop routing

Labels: legit (under the stored fact, the target answers the question), leak (it does
not, including a target of the wrong type), unsure (the two relations can differ here).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from relic.metrics import hit  # noqa: E402

SEED = 102
BOOT = 10000
CONFIRM = Path("results/live_learning/confirm")
SHEET = CONFIRM / "human_labels_sheet.csv"
KEY = CONFIRM / "human_labels_key.json"
CLASSES = ("legit", "leak", "unsure")
FIELDS = [
    "item",
    "question",
    "stored_fact",
    "stored_target",
    "response",
    "base",
    "label",
]


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    observed = float((a == b).mean())
    expected = sum(
        float((a == c).mean()) * float((b == c).mean()) for c in np.unique([*a, *b])
    )
    return 1.0 if expected == 1.0 else (observed - expected) / (1 - expected)


def kappa_ci(a, b, seed=SEED):
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(BOOT, a.size))
    boots = [cohen_kappa(a[i], b[i]) for i in idx]
    return {
        "kappa": round(cohen_kappa(a, b), 4),
        "agreement": round(float((a == b).mean()), 4),
        "ci95": [
            round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4),
        ],
        "n": int(a.size),
    }


def items():
    """Every labelled item as (source id, question fields, model label)."""
    sample = json.loads((CONFIRM / "natural_e3_labels.json").read_text())
    model = json.loads((CONFIRM / "natural_e3_labels_model.json").read_text())
    out = []
    for k, it in enumerate(sample):
        if f"i{k:03d}" in model:
            out.append((f"i{k:03d}", it, model[f"i{k:03d}"]))
    ext = json.loads((CONFIRM / "natural_e3_labels_ext.json").read_text())
    model = json.loads((CONFIRM / "natural_e3_labels_model_ext.json").read_text())
    out += [(f"e{k:03d}", it, model[f"e{k:03d}"]) for k, it in enumerate(ext)]
    return out


def make() -> None:
    if SHEET.exists():
        raise SystemExit(f"refusing to overwrite {SHEET}: it may hold labels")
    rows = items()
    random.Random(SEED).shuffle(rows)
    with SHEET.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        for n, (_, it, _) in enumerate(rows):
            writer.writerow(
                {
                    "item": n,
                    "question": it["question"],
                    "stored_fact": it["stored_fact"],
                    "stored_target": it["stored_target"],
                    "response": it["response_adapter_note"],
                    "base": it["base_response"],
                    "label": "",
                }
            )
    # kept apart from the sheet so that the labeller does not see the source or the label
    KEY.write_text(json.dumps([[src, lab] for src, _, lab in rows], indent=0))
    print(f"{len(rows)} items -> {SHEET}")


def corrected_near_leak(sheet, key):
    """The default router's near-neighbour leak with the author's share of legit items."""
    sample = json.loads((CONFIRM / "natural_e3_labels.json").read_text())
    report = json.loads((CONFIRM / "natural_e3.json").read_text())
    near = report["summary"]["default_ss76"]["near"]["leak_target_both"]
    verdicts = []
    for row, (src, _) in zip(sheet, key):
        if not src.startswith("i"):
            continue
        it = sample[int(src[1:])]
        if (
            it["router"] == "default_ss76"
            and hit(it["response_adapter_note"], it["stored_target"])
            and not hit(it["base_response"], it["stored_target"])
            and row["label"] in ("legit", "leak")
        ):
            verdicts.append(row["label"])
    legit = verdicts.count("legit") / len(verdicts)
    return {
        "labelled": len(verdicts),
        "legit_share": round(legit, 4),
        "near_leak_auto": near,
        "near_leak_corrected": round(near * (1 - legit), 4),
    }


def score() -> None:
    with SHEET.open(encoding="utf-8") as f:
        sheet = list(csv.DictReader(f))
    key = json.loads(KEY.read_text())
    bad = [r["item"] for r in sheet if r["label"].strip() not in CLASSES]
    if bad:
        raise SystemExit(f"{len(bad)} items without a valid label, first: {bad[:5]}")
    human = [r["label"].strip() for r in sheet]
    model = [lab for _, lab in key]
    decided = [(h, m) for h, m in zip(human, model) if "unsure" not in (h, m)]
    report = {
        "prereg": "SS102",
        "three_class": kappa_ci(human, model),
        "two_class": kappa_ci(*zip(*decided)),
        "confusion_human_by_model": {
            h: {
                m: sum(a == h and b == m for a, b in zip(human, model)) for m in CLASSES
            }
            for h in CLASSES
        },
        "corrected_near_leak": corrected_near_leak(sheet, key),
    }
    report["closed"] = report["two_class"]["kappa"] >= 0.6
    (CONFIRM / "human_labels.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def thresholds() -> None:
    """SS141: which of the author's labels each per-relation arm stops routing."""
    from locality_eval import match_raw

    from relic.router import RouterV7, make_scorer

    theta = 0.7501
    with SHEET.open(encoding="utf-8") as f:
        sheet = [
            r for r in csv.DictReader(f) if r["label"].strip() in ("legit", "leak")
        ]
    raw = json.loads((CONFIRM / "raw/counterfact.json").read_text())
    facts = json.loads((CONFIRM / "external_counterfact_n1000.json").read_text())[
        "facts"
    ]
    relation = {
        f["id"]: r["requested_rewrite"]["relation_id"]
        for f, r in zip(facts, match_raw(facts, raw))
    }
    rows = json.loads((CONFIRM / "natural_e3.json").read_text())["rows"]["default_ss76"]
    fact_of = {r["query"]: r["fact_id"] for r in rows}
    subjects = sorted({r["fact_id"] for r in rows})
    random.Random(107).shuffle(subjects)
    test_half = set(subjects[len(subjects) // 2 :])
    arms = {
        "R107": json.loads((CONFIRM / "tau_calib.json").read_text())["per_relation"],
        "Rall": json.loads((CONFIRM / "near_threshold.json").read_text())["thresholds"][
            "Rall"
        ],
    }
    router = RouterV7(facts, make_scorer("relenc", device="cpu"), theta)
    items = []
    for r in sheet:
        fid = fact_of.get(r["question"])
        if fid is None:
            continue
        cand, sims = router.score(r["question"])
        if not cand:
            continue
        k = int(np.argmax(sims))
        chosen = facts[cand[k]]
        if (
            sims[k] < theta
            or chosen["id"] != fid
            or chosen["answer"] != r["stored_target"]
        ):
            continue
        items.append(
            {
                "label": r["label"].strip(),
                "test_half": fid in test_half,
                "score": float(sims[k]),
                "relation": relation[fid],
            }
        )

    def read(rows_, tau) -> dict:
        out = {}
        for label in ("legit", "leak"):
            xs = [x for x in rows_ if x["label"] == label]
            gone = [x["score"] < tau.get(x["relation"], theta) for x in xs]
            out[label] = {
                "n": len(xs),
                "removed": int(sum(gone)),
                "share": round(float(np.mean(gone)), 4) if xs else None,
            }
        return out

    report = {
        "prereg": "SS141",
        "n": len(items),
        "all": {a: read(items, t) for a, t in arms.items()},
        "test_half": {
            a: read([x for x in items if x["test_half"]], t) for a, t in arms.items()
        },
    }
    (CONFIRM / "label_thresholds.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["make", "score", "thresholds"])
    {"make": make, "score": score, "thresholds": thresholds}[ap.parse_args().command]()
