"""One system prompt for training, anchor, divergence and serving (CHANGES.md, item 1)."""

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import relic.ledger as module  # noqa: E402
from relic.ledger import LEGACY_PROMPTS, DeltaStore, use_prompts  # noqa: E402
from relic.serving import SYSTEM, base_answer, serve_expert  # noqa: E402


class Recorder:
    """A tokenizer that records the chat messages it is given."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, add_generation_prompt=False, **kw):
        self.calls.append([dict(m) for m in messages])
        return torch.arange(len(messages) + 2).unsqueeze(0)


class TinyModel(torch.nn.Module):
    device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(5))

    def forward(self, input_ids=None, labels=None, output_hidden_states=False):
        n = input_ids.shape[1]
        logits = (self.w + 1).expand(1, n, 5)
        hidden = (self.w + 1).expand(1, n, 5)
        return types.SimpleNamespace(
            loss=self.w.sum() + 1.0, logits=logits, hidden_states=[hidden]
        )


def recording_store(prompts=None):
    store = DeltaStore.__new__(DeltaStore)
    store.tok, store.model = Recorder(), TinyModel()
    store.deltas, store.keys, store.notes = {}, {}, {}
    store.kl_lambda, store._base_logits, store.capacity = 0.0, None, 1
    if prompts is not None:
        use_prompts(store, prompts)
    return store


def paths_messages(store, monkeypatch):
    """The messages of training, anchor, key embedding and serving, in that order."""
    sent = []
    monkeypatch.setattr(
        module, "chat", lambda tok, model, messages, **kw: sent.append(messages) or ""
    )
    opt = torch.optim.SGD(store.model.parameters(), lr=0.1)
    store._pair_step(store.model, opt, "Ann was born in", "Oslo")
    store._logits(store.model, "What is 7 times 8?")
    store.embed("Ann was born in")
    store.deltas = {"a": {}}
    store.materialize = lambda: None
    serve_expert(store, "a", "Where was Ann born?")
    base_answer(store, "Where was Ann born?")
    return store.tok.calls, sent


def test_legacy_prompts_send_todays_messages(monkeypatch):
    calls, sent = paths_messages(recording_store(LEGACY_PROMPTS), monkeypatch)
    user = {"role": "user", "content": "Ann was born in"}
    assert calls[0] == [user]  # training prompt: no system message
    assert calls[1] == [user, {"role": "assistant", "content": "Oslo"}]
    turkish = {"role": "system", "content": "Kısa ve net cevap ver."}
    assert calls[2] == [turkish, {"role": "user", "content": "What is 7 times 8?"}]
    assert calls[3] == [turkish, user]  # the key embedding
    q = {"role": "user", "content": "Where was Ann born?"}
    english = {"role": "system", "content": "Answer briefly."}
    assert sent == [[english, q], [english, q]]


def test_default_prompts_use_the_serving_prompt_everywhere(monkeypatch):
    calls, sent = paths_messages(recording_store(), monkeypatch)
    system = {"role": "system", "content": SYSTEM}
    assert all(c[0] == system for c in calls) and len(calls) == 4
    assert all(m[0] == system for m in sent)


def test_use_prompts_refuses_an_unknown_field():
    with pytest.raises(ValueError, match="unknown prompt"):
        use_prompts(recording_store(), {"sytem": "x"})
