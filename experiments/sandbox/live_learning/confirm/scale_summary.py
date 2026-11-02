"""SS98: pooled curve over the arms that have finished, and the decision on the 7B arm.

.venv/bin/python experiments/sandbox/live_learning/confirm/scale_summary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scale_eval import ARMS, CONFIRM, FILES, OUT  # noqa: E402

from relic.metrics import ci, word_hit  # noqa: E402


def pooled_need(arm: str) -> dict:
    need = []
    for facts_file, tag in FILES.values():
        facts = json.loads((CONFIRM / facts_file).read_text())["facts"]
        notes = json.loads((OUT / f"{arm}_{tag}.notes.json").read_text())
        need += [not word_hit(notes[f["id"]]["note_probe"], f["answer"]) for f in facts]
    return {"n": len(need), "rate": round(float(np.mean(need)), 4), "ci95": ci(need)}


def decide(share: dict) -> str:
    low, high = share["ci95"]
    if high <= 0.02:
        return "A"
    if low >= 0.10:
        return "B"
    return "C"


def second_family(report: dict) -> bool:
    return all(
        d["tiered_probe_word"]["rate"] >= 0.95
        and d["tiered_paraphrase_word"]["rate"] > d["note_paraphrase_word"]["rate"]
        for d in report["datasets"].values()
    )


def main():
    summary = {"prereg": "SS98", "arms": {}}
    for arm in ARMS:
        path = OUT / f"{arm}.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        summary["arms"][arm] = {
            "model": report["model"],
            "four_bit": report["four_bit"],
            "memory_mb": report["memory_mb"],
            "pooled_needs_adapter": pooled_need(arm),
            "datasets": report["datasets"],
        }
    arms = summary["arms"]
    if "q15" in arms and "q15q" in arms:
        gap = abs(
            arms["q15"]["pooled_needs_adapter"]["rate"]
            - arms["q15q"]["pooled_needs_adapter"]["rate"]
        )
        summary["quantization_gap"] = round(gap, 4)
        summary["quantization_confound"] = gap > 0.02
    if "q7q" in arms:
        summary["decision"] = decide(arms["q7q"]["pooled_needs_adapter"])
    if "smol3q" in arms:
        summary["second_family_holds"] = second_family(
            json.loads((OUT / "smol3q.json").read_text())
        )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    for arm, row in arms.items():
        print(arm, row["pooled_needs_adapter"], row["memory_mb"], flush=True)
    print({k: v for k, v in summary.items() if k not in ("arms", "prereg")})


if __name__ == "__main__":
    main()
