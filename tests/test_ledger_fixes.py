"""Fixes before any GPU run: config in the file, non-finite adapters, complete erasure,
an atomic retract, harm readings for kappa, restores read after removal, TTFT parts."""

import sys
import warnings
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

CONFIRM = Path(__file__).parents[1] / "experiments/sandbox/live_learning/confirm"
sys.path.insert(0, str(CONFIRM))
sys.path.insert(0, str(CONFIRM.parent / "text"))

import relic.adapters as adapters  # noqa: E402
import relic.ledger as module  # noqa: E402
from relic.adapters import int8_delta, pack_delta  # noqa: E402
from relic.ledger import Ledger  # noqa: E402
from relic.router import RELATION_TAU  # noqa: E402
from test_ledger import TableScorer, fact, make_ledger  # noqa: E402


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


# -- 1. the ledger's configuration travels with its file --------------------------
def test_a_current_notless_ledger_loads_in_ttft_eval(tmp_path):
    import notless_eval
    import ttft_eval

    store = make_ledger().store
    led = notless_eval.build_ledger(store, TableScorer(), "current", kappa=0.9)
    assert led.relation_tau == RELATION_TAU and led.keep_history
    led.commit({**fact("a", "Ann", "Ann was born in", "Oslo"), "relation": "P19"}, d(1))
    path = tmp_path / "s.ledger"
    led.save(path)
    loaded = ttft_eval.load_ledger(path, make_ledger().store, TableScorer())
    assert loaded.state_hash() == led.state_hash()
    assert loaded.kappa == 0.9 and loaded.relation_tau == RELATION_TAU


def test_from_file_restores_every_setting_and_the_prompts(tmp_path):
    led = make_ledger()
    led.tau, led.slot_theta, led.duplicate_theta = 0.7, 0.95, 0.72
    led.relation_tau, led.kappa, led.keep_history = {"P37": 0.83}, 1.2, False
    led.store.system, led.store.anchor_system = "Be brief.", None
    led.commit(fact("a", "Ann", "Ann was born in", "Oslo"), d(1))
    led.save(tmp_path / "s.ledger")
    other = Ledger.from_file(tmp_path / "s.ledger", make_ledger().store, TableScorer())
    assert (other.tau, other.slot_theta, other.duplicate_theta) == (0.7, 0.95, 0.72)
    assert other.relation_tau == {"P37": 0.83} and other.kappa == 1.2
    assert not other.keep_history and other.store.system == "Be brief."
    assert other.store.anchor_system is None
    assert other.state_hash() == led.state_hash()


# -- 2. non-finite adapters ----------------------------------------------------------
def nonfinite():
    a = torch.randn(2, 3)
    a[1, 0] = float("nan")
    b = torch.randn(3, 2)
    b[0, 1] = float("inf")
    return {"m.lora_A": {"A": a, "B": b}}


def test_quantize_rows_refuses_non_finite_values():
    with pytest.raises(ValueError, match="non-finite"):
        adapters.quantize_rows(nonfinite()["m.lora_A"]["A"])
    with pytest.raises(ValueError, match="non-finite"):
        adapters.quantize_rows(nonfinite()["m.lora_A"]["B"])


def test_write_rejects_a_non_finite_adapter_before_the_divergence(monkeypatch):
    led = make_ledger()
    led.store.kl_lambda = 1.0
    monkeypatch.setattr(led, "_rewrites", lambda f: [])
    rungs = iter([pack_delta(nonfinite()), pack_delta(d(2))])
    monkeypatch.setattr(module, "train_adapter_packed", lambda s, p, seed: next(rungs))
    seen = []
    monkeypatch.setattr(led.store, "_expert_kld", lambda x, p: seen.append(x) or 0.5)
    event = led.write(fact("a", "Ann", "Ann was born in", "Oslo"))
    assert event["rung"] == "R1" and len(seen) == 1  # A1 never reached the gate
    assert event["rejected_rungs"] == {"A1": "non-finite adapter"}


def test_commit_refuses_a_non_finite_adapter():
    led = make_ledger()
    before = led.state_hash()
    with pytest.raises(ValueError, match="non-finite"):
        led.commit(fact("a", "Ann", "Ann was born in", "Oslo"), nonfinite())
    assert led.state_hash() == before and not led.entries


# -- 3. erasure is complete ------------------------------------------------------------
class Birthplace(TableScorer):
    """Three phrasings of one relation."""

    SAME = {"X was born in", "The birthplace of X is", "X's place of birth is"}

    def sims(self, query, keys):
        return [1.0 if query == k or {query, k} <= self.SAME else 0.2 for k in keys]


def birth_ledger():
    led = make_ledger()
    led.scorer = Birthplace()
    return led


A = fact("a", "Ann", "Ann was born in", "Oslo")
C = fact("c", "Ann", "The birthplace of Ann is", "Oslo")
B = fact("b", "Ann", "Ann's place of birth is", "Rome")


def history_ledger():
    led = birth_ledger()
    led.commit(A, d(1))
    led.commit(C)  # a duplicate of a, which adds its phrasing
    led.commit(B, d(2))  # supersedes a and inherits both phrasings
    assert sorted(led.phrasings["b"]) == sorted([A["probe"], C["probe"]])
    return led


def text_of(led) -> str:
    return repr((led.entries, led.phrasings, led.writes, led.history))


def only_b():
    led = birth_ledger()
    led.commit(B, d(2))
    return led


@pytest.mark.parametrize(
    "ops", [(("retract", "c"), ("revoke", "a")), (("revoke", "a"),)]
)
def test_erasing_superseded_writes_leaves_no_text_of_them(ops):
    led = history_ledger()
    for op, fid in ops:
        getattr(led, op)(fid)
    for gone in (A["probe"], C["probe"], "Oslo"):
        assert gone not in text_of(led)
    assert led.state_hash() == only_b().state_hash()
    if len(ops) == 1:  # c was a write of a, erased with it
        with pytest.raises(KeyError):
            led.retract("c")


def test_retracting_a_write_inside_history_keeps_the_rest():
    led = history_ledger()
    led.retract("c")
    assert C["probe"] not in text_of(led) and A["probe"] in led.phrasings["b"]
    assert led.revoke("b")["restored"] == "a"
    assert led.phrasings["a"] == []


# -- 4. retract's rebuild is atomic ----------------------------------------------------
def replay_ledger():
    led = make_ledger()
    led.store.model = object()  # the write path is available
    led.commit(fact("a", "Ann", "Ann was born in", "Oslo"), d(1))
    led.commit(fact("b", "Ann", "The birthplace of Ann is", "Oslo"))
    led.commit(fact("c", "Ann", "Ann was born in", "Oslo"))
    return led


def test_a_failing_replay_leaves_the_ledger_as_it_was(monkeypatch):
    led = replay_ledger()
    before, log_len = led.state_hash(), len(led.log)
    calls = []

    def write(f, **extra):
        calls.append(f["id"])
        if len(calls) == 2:
            raise RuntimeError("CUDA out of memory")
        return led.commit(f, d(5), **extra)

    monkeypatch.setattr(led, "write", write)
    with pytest.raises(RuntimeError, match="out of memory"):
        led.retract("a")
    assert led.state_hash() == before and len(led.log) == log_len
    assert "a" in led.store.deltas and sorted(led.entries) == ["a"]


def test_a_rejected_replay_is_listed_and_the_stored_write_named(monkeypatch):
    led = replay_ledger()

    def write(f, **extra):
        if f["id"] == "b":
            return led._record("reject", "b", tried={"A1": 9.0}, **extra)
        return led.commit(f, d(5), **extra)

    monkeypatch.setattr(led, "write", write)
    event = led.retract("a")
    assert event["rejected_replays"] == ["b"] and event["rebuilt_from"] == "c"
    assert sorted(led.entries) == ["c"]


# -- 5. kappa_calib reads harm ---------------------------------------------------------
class HarmStore:
    """Answers 'Oslo' as the base model; with an adapter, 'Rome' to the questions."""

    def __init__(self):
        self.deltas = {}

    def materialize(self):
        pass

    def answer(self, q, system=None, max_new_tokens=None):
        return "Rome" if self.deltas and q.endswith("?") else "Oslo"


def test_harm_reads_retention_and_changed_canaries():
    import kappa_calib

    questions = [
        {"question": "Where?", "answer": "Oslo"},
        {"question": "Where.", "answer": "Oslo"},
    ]
    canaries = ["One?", "Two.", "Three?", "Four."]
    store = HarmStore()
    base = kappa_calib.base_readings(store, questions, canaries)
    h = kappa_calib.harm(store, pack_delta(d(1)), questions, canaries, base)
    assert h == {"retention": 0.5, "canaries_changed": 0.5}
    assert store.deltas == {}


def test_separation_is_the_auroc_of_d_for_harm_above_a_threshold():
    import kappa_calib

    rows = [
        {"d": 0.1, "harm": {"retention": 0.95, "canaries_changed": 0.0}},
        {"d": 0.2, "harm": {"retention": 0.9, "canaries_changed": 0.1}},
        {"d": 2.0, "harm": {"retention": 0.3, "canaries_changed": 0.6}},
        {"d": 3.0, "harm": {"retention": 0.2, "canaries_changed": 0.8}},
    ]
    s = kappa_calib.separation(rows, retention_drop=0.5, canary_change=0.5)
    assert s["retention"]["auroc"] == 1.0 and s["retention"]["harmful"] == 2
    assert s["canaries"]["auroc"] == 1.0
    assert kappa_calib.separation(rows, None, None) == {
        "retention": "TODO(author): threshold not set",
        "canaries": "TODO(author): threshold not set",
    }


# -- 6. restores are read after removal --------------------------------------------
def test_a_restored_fact_gets_its_holder_back():
    import notless_eval

    led = birth_ledger()
    holder = {}
    for f, x in ((A, d(1)), (C, None), (B, d(2))):
        notless_eval.account_holder(holder, f, led.commit(f, x))
    assert holder == {"a": "b", "c": "b", "b": "b"}
    event = led.revoke("b")
    notless_eval.restore_holders(holder, led, [event])
    assert holder == {"a": "a", "c": "a", "b": None}


# -- 7. TTFT parts and an unknown model revision ---------------------------------
def test_ttft_times_the_swap_back_on_its_own():
    import ttft_eval

    assert "swap_back_s" in ttft_eval.PARTS
    rows = [{p: 0.1 for p in ttft_eval.PARTS}]
    assert ttft_eval.summary(rows)["swap_back_s"]["median"] == 0.1


def test_a_store_without_a_model_revision_warns(monkeypatch):
    class Config:
        pass

    class Base:
        config = Config()

    monkeypatch.setattr(module, "load_model", lambda name: (None, Base()))
    monkeypatch.setattr(module, "make_lora", lambda base, r: torch.nn.Linear(1, 1))
    with pytest.warns(UserWarning, match="revision"):
        module.DeltaStore("some/model", capacity=1)
    Base.config._commit_hash = "abc123"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert module.DeltaStore("some/model", capacity=1).revision == "abc123"
