"""End-to-end check of relic.ledger.Ledger on the real model: write, ask, update, remove.

.venv/bin/python experiments/sandbox/live_learning/confirm/ledger_smoke.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from relic.ledger import DeltaStore, Ledger
from relic.metrics import hit
from relic.router import make_scorer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
NEW = {
    "id": "f1",
    "subject": "Danielle Darrieux",
    "probe": "The mother tongue of Danielle Darrieux is",
    "paraphrase": "Danielle Darrieux spoke the language",
    "answer": "English",
    "teach": "The mother tongue of Danielle Darrieux is English",
    "aliases": ["Danielle Yvonne Darrieux"],
}
# a counterfactual target that the note alone rarely produces, so the adapter tier is used
HARD = {
    "id": "f2",
    "subject": "Toko Yasuda",
    "probe": "Toko Yasuda, the",
    "paraphrase": "Toko Yasuda performs on the",
    "answer": "piano",
    "teach": "Toko Yasuda, the piano",
}
UPDATE = {**NEW, "id": "f3", "answer": "Swedish", "teach": f"{NEW['probe']} Swedish"}


def main() -> None:
    store = DeltaStore(MODEL, steps=16, capacity=1)
    ledger = Ledger(store, make_scorer("relenc"), 0.7501, 0.8985)
    empty = ledger.state_hash()
    off_topic = "What is the capital of France?"
    before = ledger.ask(off_topic)
    assert before["entry"] is None

    events = [ledger.write(NEW), ledger.write(HARD)]
    print(json.dumps(events, indent=1))
    assert [e["action"] for e in events] == ["insert", "insert"]

    answer = ledger.ask(NEW["probe"])
    print(answer)
    assert answer["entry"] == "f1" and hit(answer["answer"], "English")
    by_alias = ledger.ask("The mother tongue of Danielle Yvonne Darrieux is")
    print(by_alias)
    assert by_alias["entry"] == "f1"
    by_pronoun = ledger.ask(
        "What is her mother tongue?", history=["Tell me about Danielle Darrieux."]
    )
    print(by_pronoun)
    assert ledger.ask(off_topic)["answer"] == before["answer"]

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "store.ledger"
        ledger.save(path)
        reloaded = Ledger(store, ledger.scorer, 0.7501, 0.8985)
        reloaded.load(path)
        assert reloaded.state_hash() == ledger.state_hash()

    duplicate = ledger.write({**NEW, "id": "f4", "probe": NEW["paraphrase"]})
    print(duplicate)
    updated = ledger.write(UPDATE)
    print(updated)
    assert updated["action"] == "update" and updated["superseded"] == "f1"
    assert hit(ledger.ask(NEW["probe"])["answer"], "Swedish")

    for fid in list(ledger.entries):
        ledger.revoke(fid)
    assert ledger.state_hash() == empty
    assert ledger.ask(NEW["probe"])["entry"] is None
    print([e["action"] for e in ledger.log])
    print("ok")


if __name__ == "__main__":
    main()
