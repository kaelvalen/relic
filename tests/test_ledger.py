"""Write-time slots, serial commits and durable saves of the ledger; no model needed."""

import threading

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from relic.ledger import DeltaStore, Ledger  # noqa: E402

SAME_RELATION = {"X was born in", "The birthplace of X is"}


class TableScorer:
    """1.0 between phrasings of one relation, 0.2 otherwise."""

    def sims(self, query, keys):
        return [1.0 if query == k or {query, k} == SAME_RELATION else 0.2 for k in keys]


def fact(fid, subject, probe, answer):
    return {
        "id": fid,
        "subject": subject,
        "probe": probe,
        "answer": answer,
        "teach": f"{probe} {answer}",
    }


def make_ledger():
    store = DeltaStore.__new__(DeltaStore)
    store.base_hash = "base"
    store.router_config = "enc=a|tau=0.75"
    store.deltas, store.keys, store.notes = {}, {}, {}
    return Ledger(store, TableScorer(), tau=0.75, slot_theta=0.9)


def delta():
    return {"m.lora_A": {"A": torch.ones(2, 3), "B": torch.zeros(3, 2)}}


def test_new_subject_or_relation_is_a_new_entry():
    led = make_ledger()
    assert led.commit(fact("a", "Ann", "Ann was born in", "Oslo"))["action"] == "insert"
    other_subject = fact("b", "Bob", "Bob was born in", "Oslo")
    assert led.commit(other_subject)["action"] == "insert"
    other_relation = fact("c", "Ann", "Ann plays", "jazz")
    assert led.commit(other_relation)["action"] == "insert"
    assert sorted(led.entries) == ["a", "b", "c"]


def test_same_slot_same_answer_adds_a_phrasing_not_an_entry():
    led = make_ledger()
    led.commit(fact("a", "Ann", "Ann was born in", "Oslo"), delta())
    event = led.commit(fact("b", "Ann", "The birthplace of Ann is", "oslo"))
    assert event["action"] == "duplicate" and event["entry"] == "a"
    assert sorted(led.entries) == ["a"]
    assert led.phrasings["a"] == ["The birthplace of Ann is"]


def test_removing_a_written_twice_fact_removes_it():
    led = make_ledger()
    empty = led.state_hash()
    led.commit(fact("a", "Ann", "Ann was born in", "Oslo"), delta())
    led.commit(fact("b", "Ann", "The birthplace of Ann is", "Oslo"))
    led.revoke("a")
    assert led.entries == {} and led.store.deltas == {}
    assert led.state_hash() == empty


def test_same_slot_other_answer_replaces_the_entry():
    led = make_ledger()
    led.commit(fact("a", "Ann", "Ann was born in", "Oslo"), delta())
    event = led.commit(fact("b", "Ann", "The birthplace of Ann is", "Rome"))
    assert event["action"] == "update" and event["superseded"] == "a"
    assert sorted(led.entries) == ["b"]
    assert "a" not in led.store.deltas and "a" not in led.store.notes
    assert led.phrasings["b"] == ["Ann was born in"]
    assert [e["action"] for e in led.log] == ["insert", "update"]


def test_hash_covers_phrasings():
    led = make_ledger()
    led.commit(fact("a", "Ann", "Ann was born in", "Oslo"))
    h = led.state_hash()
    led.commit(fact("b", "Ann", "The birthplace of Ann is", "Oslo"))
    assert led.state_hash() != h


def test_hash_covers_the_delta_and_is_order_independent():
    a, b = make_ledger(), make_ledger()
    ann, bob = fact("a", "Ann", "Ann plays", "jazz"), fact(
        "b", "Bob", "Bob plays", "rock"
    )
    a.commit(ann, delta()), a.commit(bob)
    b.commit(bob), b.commit(ann, delta())
    assert a.state_hash() == b.state_hash()
    other = delta()
    other["m.lora_A"]["A"] = torch.zeros(2, 3)
    c = make_ledger()
    c.commit(ann, other), c.commit(bob)
    assert c.state_hash() != a.state_hash()


def test_concurrent_writers_of_one_slot_leave_one_entry():
    led = make_ledger()
    start = threading.Barrier(8)

    def writer(i):
        start.wait()
        led.commit(fact(f"f{i}", "Ann", "Ann was born in", f"City{i}"))

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(led.entries) == 1
    assert [e["seq"] for e in led.log] == list(range(8))
    assert [e["action"] for e in led.log] == ["insert"] + ["update"] * 7
    # every update superseded exactly the entry the previous commit left
    for prev, cur in zip(led.log, led.log[1:]):
        assert cur["superseded"] == prev["entry"]


def test_save_and_load_restore_the_state(tmp_path):
    led = make_ledger()
    led.commit(fact("a", "Ann", "Ann was born in", "Oslo"), delta())
    led.commit(fact("b", "Ann", "The birthplace of Ann is", "Oslo"))
    path = tmp_path / "store.ledger"
    led.save(path)
    assert not path.with_name("store.ledger.tmp").exists()
    other = make_ledger()
    other.load(path)
    assert other.state_hash() == led.state_hash()
    assert other.phrasings == led.phrasings and other.log == led.log


def test_a_failed_save_keeps_the_previous_file(tmp_path, monkeypatch):
    led = make_ledger()
    led.commit(fact("a", "Ann", "Ann was born in", "Oslo"))
    path = tmp_path / "store.ledger"
    led.save(path)
    before = path.read_bytes()
    led.commit(fact("c", "Bob", "Bob plays", "jazz"))

    def crash(obj, f):
        f.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", crash)
    with pytest.raises(OSError):
        led.save(path)
    assert path.read_bytes() == before


def test_load_refuses_a_file_whose_contents_changed(tmp_path):
    led = make_ledger()
    led.commit(fact("a", "Ann", "Ann was born in", "Oslo"))
    path = tmp_path / "store.ledger"
    led.save(path)
    state = torch.load(path, weights_only=False)
    state["entries"]["a"]["answer"] = "Rome"
    torch.save(state, path)
    with pytest.raises(ValueError, match="state hash"):
        make_ledger().load(path)
