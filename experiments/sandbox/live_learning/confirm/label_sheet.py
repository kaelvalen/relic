"""SS102: the human label sheet for the natural-question sample, and its scoring.

    python label_sheet.py make    # writes the sheet; fill in its `label` column
    python label_sheet.py score   # agreement with the model labels
    python label_sheet.py thresholds   # SS141: what the per-relation arms stop routing
    python label_sheet.py nli   # SS147: an entailment score on the same items

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


THETA = 0.7501


def routed_items() -> list[dict]:
    """SS141: the author's legit and leak items the default router routes to their fact."""
    from locality_eval import match_raw

    from relic.router import RouterV7, make_scorer

    theta = THETA
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
                "question": r["question"],
                "stored_fact": r["stored_fact"],
                "stored_target": r["stored_target"],
            }
        )
    return items


def thresholds() -> None:
    """SS141: which of the author's labels each per-relation arm stops routing."""
    theta = THETA
    items = routed_items()
    arms = {
        "R107": json.loads((CONFIRM / "tau_calib.json").read_text())["per_relation"],
        "Rall": json.loads((CONFIRM / "near_threshold.json").read_text())["thresholds"][
            "Rall"
        ],
    }

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


NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
NLI_REVISION = "bab4bc7178836f731dcfd18c06ca9def0a137712"
HYPOTHESIS = 'The answer to the question "{question}" is {target}.'


def nli() -> None:
    """SS147: does an entailment score over the stored fact tell legit from leak?"""
    from near_threshold import auroc
    from sentence_transformers import CrossEncoder

    items = routed_items()
    model = CrossEncoder(NLI_MODEL, revision=NLI_REVISION, device="cpu")
    label = {v: int(k) for k, v in model.config.id2label.items()}
    pairs = [
        (x["stored_fact"], HYPOTHESIS.format(question=x["question"], target=x["stored_target"]))
        for x in items
    ]
    probs = model.predict(pairs, apply_softmax=True, batch_size=16)
    for x, p in zip(items, probs):
        x["entail"] = float(p[label["entailment"]])
        x["entail_minus_contra"] = float(p[label["entailment"]] - p[label["contradiction"]])

    def auc(xs, key) -> dict:
        legit = np.array([x[key] for x in xs if x["label"] == "legit"])
        leak = np.array([x[key] for x in xs if x["label"] == "leak"])
        rng = np.random.default_rng(147)
        boots = [
            auroc(rng.choice(legit, legit.size), rng.choice(leak, leak.size))
            for _ in range(BOOT)
        ]
        return {
            "auroc": round(auroc(legit, leak), 4),
            "ci95": [round(float(np.percentile(boots, q)), 4) for q in (2.5, 97.5)],
        }

    # operating point: calibrated on the half that is not the test half of SS107
    calib = [x["entail"] for x in items if not x["test_half"] and x["label"] == "legit"]
    tau = float(np.percentile(calib, 10))
    test = [x for x in items if x["test_half"]]
    removed = {
        lab: round(float(np.mean([x["entail"] < tau for x in test if x["label"] == lab])), 4)
        for lab in ("legit", "leak")
    }
    report = {
        "prereg": "SS147",
        "model": NLI_MODEL,
        "revision": NLI_REVISION,
        "n": {lab: sum(x["label"] == lab for x in items) for lab in ("legit", "leak")},
        "auroc": {k: auc(items, k) for k in ("entail", "entail_minus_contra", "score")},
        "operating_point": {
            "tau": round(tau, 4),
            "test_n": {lab: sum(x["label"] == lab for x in test) for lab in ("legit", "leak")},
            "removed": removed,
            "leak_minus_legit": round(removed["leak"] - removed["legit"], 4),
        },
        "items": items,
    }
    (CONFIRM / "label_nli.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["make", "score", "thresholds", "nli"])
    commands = {"make": make, "score": score, "thresholds": thresholds, "nli": nli}
    commands[ap.parse_args().command]()
