"""The ledger after CHANGES.md items 1 to 7: prompts, stored records, reversible updates,
retracted writes, the v2 state hash, the log and removal timing. No model needed."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import relic.ledger as module  # noqa: E402
from relic.adapters import int8_delta, pack_delta, pad, unpack_delta  # noqa: E402
from relic.ledger import (  # noqa: E402
    Ledger,
    _legacy_state_hash,
)
from test_ledger import TableScorer, delta, fact, make_ledger  # noqa: E402

FIXTURE = Path(__file__).parent / "data" / "ledger_v1.ledger"
FIXTURE_HASH = "e2a6110d5d2eae88c77c357aab9efd6401df4a87cd32af647d8836cf6efbbe15"


def d(seed):
    g = torch.Generator().manual_seed(seed)
    return int8_delta(
        {
            "m.lora_A": {
                "A": torch.randn(2, 3, generator=g),
                "B": torch.randn(3, 2, generator=g),
            }
        }
    )


# -- item 2: the stored records ------------------------------------------------------
def qwen_padded(seed):
    from test_adapters_pack import qwen_delta

    return pad(qwen_delta(seed))


def test_the_file_grows_by_the_stored_record_per_entry(tmp_path):
    sizes = {}
    for n in (1, 3):
        led = make_ledger()
        for i in range(n):
            led.commit(fact(f"f{i}", f"S{i}", f"S{i} plays", "jazz"), qwen_padded(i))
        path = tmp_path / f"n{n}.ledger"
        led.save(path)
        sizes[n] = path.stat().st_size
        assert all(p["format"] == "int8-rows" for p in led.packed.values())
        assert "deltas" not in torch.load(path, weights_only=False)
    per_entry = (sizes[3] - sizes[1]) / 2
    # one buffer per record; the rest (about 4%) is its index of module names and the
    # entry's text
    assert 372_960 <= per_entry <= 372_960 * 1.06


def test_load_rebuilds_the_padded_adapters_from_the_records(tmp_path):
    led = make_ledger()
    led.commit(fact("a", "Ann", "Ann plays", "jazz"), pad(d(1)))
    led.save(tmp_path / "s.ledger")
    other = make_ledger()
    other.load(tmp_path / "s.ledger")
    assert torch.equal(
        other.store.deltas["a"]["m.lora_A"]["A"], pad(d(1))["m.lora_A"]["A"]
    )
    assert other.state_hash() == led.state_hash()


# -- item 3: reversible updates ------------------------------------------------------
def ann(fid, answer, probe="Ann was born in"):
    return fact(fid, "Ann", probe, answer)


def test_revoking_an_update_restores_the_store_before_it():
    led = make_ledger()
    led.commit(ann("a", "Oslo"), d(1))
    after_a = led.state_hash()
    led.commit(ann("b", "Rome"), d(2))
    assert sorted(led.entries) == ["b"] and [
        s["entry"]["id"] for s in led.history["b"]
    ] == ["a"]
    event = led.revoke("b")
    assert event["restored"] == "a"
    assert led.state_hash() == after_a
    assert torch.equal(
        unpack_delta(led.packed["a"])["m.lora_A"]["A"], d(1)["m.lora_A"]["A"]
    )
    assert "a" in led.store.deltas and "b" not in led.store.deltas


def test_a_three_deep_chain_unwinds_in_order():
    led = make_ledger()
    hashes = []
    for i, (fid, answer) in enumerate((("a", "Oslo"), ("b", "Rome"), ("c", "Pisa"))):
        led.commit(ann(fid, answer), d(i))
        hashes.append(led.state_hash())
    assert [s["entry"]["id"] for s in led.history["c"]] == ["a", "b"]
    assert led.revoke("c")["restored"] == "b" and led.state_hash() == hashes[1]
    assert led.revoke("b")["restored"] == "a" and led.state_hash() == hashes[0]
    assert led.revoke("a")["restored"] is None and not led.entries


def test_revoking_a_superseded_fact_erases_it():
    led = make_ledger()
    led.commit(ann("a", "Oslo"), d(1))
    led.commit(ann("b", "Rome"), d(2))
    event = led.revoke("a")
    assert event["purged_from"] == "b" and "b" not in led.history
    assert (
        "a" not in led.ids() and "a" not in led.packed and "a" not in led.store.deltas
    )
    # nothing of a's adapter is left anywhere in the ledger
    blob = repr(led.history) + repr(led.packed.keys())
    assert "'a'" not in blob
    assert led.revoke("b")["restored"] is None and not led.entries


def test_keep_history_false_is_the_old_update():
    led = make_ledger()
    led.keep_history = False
    led.commit(ann("a", "Oslo"), d(1))
    led.commit(ann("b", "Rome"), d(2))
    assert led.history == {}
    led.revoke("b")
    assert not led.entries


# -- item 4: writes -------------------------------------------------------------------
def test_retracting_a_duplicate_is_as_if_it_never_happened():
    led, ref = make_ledger(), make_ledger()
    led.commit(ann("a", "Oslo"), d(1))
    ref.commit(ann("a", "Oslo"), d(1))
    led.commit(ann("b", "Oslo", "The birthplace of Ann is"))
    led.retract("b")
    assert led.state_hash() == ref.state_hash() and led.phrasings["a"] == []


def test_the_phrasing_flag_passes_to_the_next_write_with_that_prompt():
    led, ref = make_ledger(), make_ledger()
    for x in (led, ref):
        x.commit(ann("a", "Oslo"), d(1))
    led.commit(ann("b", "Oslo", "The birthplace of Ann is"))
    led.commit(ann("c", "Oslo", "The birthplace of Ann is"))
    ref.commit(ann("c", "Oslo", "The birthplace of Ann is"))
    led.retract("b")
    assert led.phrasings["a"] == ["The birthplace of Ann is"]
    assert led.state_hash() == ref.state_hash()


def test_retracting_the_own_write_rebuilds_from_the_next_one():
    led, ref = make_ledger(), make_ledger()
    led.commit(ann("a", "Oslo"), d(1))
    led.commit(ann("b", "Oslo", "The birthplace of Ann is"))
    led.commit(ann("c", "Oslo"))
    event = led.retract("a")
    assert event["mode"] == "rebuilt" and event["rebuilt_from"] == "b"
    ref.commit(ann("b", "Oslo", "The birthplace of Ann is"))
    ref.commit(ann("c", "Oslo"))
    assert led.state_hash() == ref.state_hash()
    # commit-only: the adapter trained on the retracted write does not survive
    assert not led.packed and not led.store.deltas


def test_retracting_the_own_write_of_an_update_keeps_what_it_superseded():
    led, ref = make_ledger(), make_ledger()
    for x in (led, ref):
        x.commit(ann("x", "Rome"))
    led.commit(ann("a", "Oslo"))
    led.commit(ann("b", "Oslo", "The birthplace of Ann is"))
    led.retract("a")
    ref.commit(ann("b", "Oslo", "The birthplace of Ann is"))
    assert led.state_hash() == ref.state_hash()
    assert [s["entry"]["id"] for s in led.history["b"]] == ["x"]


def test_retracting_the_last_write_is_a_revoke():
    led = make_ledger()
    empty = led.state_hash()
    led.commit(ann("a", "Oslo"), d(1))
    assert led.retract("a")["mode"] == "last"
    assert led.state_hash() == empty


def test_ids_are_unique_across_entries_writes_and_history():
    led = make_ledger()
    led.commit(ann("a", "Oslo"), d(1))
    led.commit(ann("b", "Oslo", "The birthplace of Ann is"))
    led.commit(fact("c", "Bob", "Bob plays", "jazz"))
    led.commit(fact("e", "Bob", "Bob plays", "rock"))  # supersedes c
    before = led.state_hash()
    for reused in ("a", "b", "c", "e"):
        with pytest.raises(ValueError, match="held by another fact or write"):
            led.commit(fact(reused, "Zed", "Zed plays", "pop"))
    assert led.state_hash() == before
    with pytest.raises(KeyError, match="retract"):
        led.revoke("b")


# -- item 5: the v2 hash ------------------------------------------------------------
def test_the_hash_covers_what_decides_behavior(monkeypatch):
    led = make_ledger()
    led.commit(ann("a", "Oslo"), d(1))
    h = led.state_hash()
    led.kappa = 0.5
    assert led.state_hash() != h
    led.kappa = module.KAPPA
    for field, value in (("system", "Be brief."), ("anchor_system", None)):
        old = getattr(led.store, field)
        setattr(led.store, field, value)
        assert led.state_hash() != h
        setattr(led.store, field, old)
    led.store.revision = "abc"
    assert led.state_hash() != h
    led.store.revision = None
    monkeypatch.setattr(module, "LADDER", module.LADDER[:-1])
    assert led.state_hash() != h
    monkeypatch.setattr(module, "LADDER", module.LADDER)


def test_the_hash_covers_writes_and_history():
    a, b = make_ledger(), make_ledger()
    for x in (a, b):
        x.commit(ann("a", "Oslo"))
    a.commit(ann("b", "Oslo"))  # same prompt: a write with no new phrasing
    assert a.state_hash() != b.state_hash()
    c = make_ledger()
    c.commit(ann("z", "Rome"))
    c.commit(ann("a", "Oslo"))
    assert c.state_hash() != b.state_hash()  # same active entry, a history behind it


def test_a_file_of_the_old_format_loads_after_its_own_hash_is_checked():
    state = torch.load(FIXTURE, weights_only=False)
    assert "format" not in state and state["state_hash"] == FIXTURE_HASH
    digests = {f: module._delta_digest(x) for f, x in state["deltas"].items()}
    led = make_ledger()
    assert (
        _legacy_state_hash(
            "base",
            state["router_config"],
            [led.tau, led.slot_theta, led.duplicate_theta],
            {},
            state["entries"],
            state["phrasings"],
            digests,
        )
        == FIXTURE_HASH
    )
    led.load(FIXTURE)
    assert sorted(led.entries) == ["a", "d"] and led.history == {}
    assert [w["fact"]["id"] for w in led.writes["a"]] == ["a"]
    for fid, old in state["deltas"].items():
        assert torch.equal(
            unpack_delta(led.packed[fid])["m.lora_A"]["A"], old["m.lora_A"]["A"]
        )


def test_a_changed_file_of_the_old_format_is_refused(tmp_path):
    state = torch.load(FIXTURE, weights_only=False)
    state["entries"]["a"]["answer"] = "Rome"
    torch.save(state, tmp_path / "bad.ledger")
    with pytest.raises(ValueError, match="state hash"):
        make_ledger().load(tmp_path / "bad.ledger")


def test_save_and_load_keep_writes_and_history(tmp_path):
    led = make_ledger()
    led.commit(ann("a", "Oslo"), d(1))
    led.commit(ann("b", "Rome"), d(2))
    led.commit(ann("c", "Rome", "The birthplace of Ann is"))
    led.save(tmp_path / "s.ledger")
    other = make_ledger()
    other.load(tmp_path / "s.ledger")
    assert other.state_hash() == led.state_hash()
    assert other.revoke("b")["restored"] == "a"


# -- item 6: the log ----------------------------------------------------------------
def test_write_logs_the_rung_every_divergence_the_seed_and_the_time(monkeypatch):
    led = make_ledger()
    led.store.kl_lambda = 1.0
    monkeypatch.setattr(
        module, "train_adapter_packed", lambda s, p, seed: pack_delta(delta())
    )
    monkeypatch.setattr(led, "_rewrites", lambda f: [])
    values = iter([2.0, 0.5])
    monkeypatch.setattr(led.store, "_expert_kld", lambda x, p: next(values))
    event = led.write(ann("a", "Oslo"))
    assert led.log[-1] == event
    assert event["rung"] == "R1" and event["kld"] == 0.5 and event["seed"] == 127
    assert event["tried"] == {"A1": 2.0, "R1": 0.5} and event["seconds"] >= 0


def test_commit_passes_extra_fields_into_the_log():
    led = make_ledger()
    event = led.commit(ann("a", "Oslo"), note_for_log="x")
    assert led.log[-1]["note_for_log"] == "x" == event["note_for_log"]


# -- item 7: removal includes the router rebuild ------------------------------------
def test_removal_rebuilds_the_router_and_logs_its_time():
    led = make_ledger()
    led.commit(ann("a", "Oslo"), d(1))
    led.commit(fact("b", "Bob", "Bob plays", "jazz"))
    event = led.revoke("a")
    assert led._router is not None and event["seconds"] >= 0
    assert led.router[0].route("Bob plays")[0] == 0


def test_a_ledger_with_a_table_scorer_still_routes():
    led = Ledger(make_ledger().store, TableScorer(), tau=0.75, slot_theta=0.9)
    led.commit(ann("a", "Oslo"))
    assert led.router[0].route("Ann was born in")[0] == 0
