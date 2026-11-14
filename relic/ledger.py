"""Transactional delta store (slice-1, docs/LIVING_MODEL_SLICE_PREREG.md).

Each fact -> one LoRA delta trained from the same frozen base; the live state is the
sum of the active deltas in id order, materialised into a fixed-capacity adapter.
`revoke` drops a delta and rebuilds; `state_hash` is the pure function of the active
set (identifier, note, routing key and delta of each entry, plus the router
configuration), so revoking the last addition restores the recorded pre-add hash exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import warnings
from pathlib import Path

import torch

from relic.adapters import (
    B_MULT,
    LADDER,
    R2,
    REWRITE_SYSTEM,
    REWRITE_USER,
    NonFiniteAdapter,
    blob_to_record,
    fact_pairs,
    pack_delta,
    pad,
    parse_rewrites,
    record_to_blob,
    train_adapter_packed,
    trim_rank,
    unpack_delta,
)
from relic.metrics import sym_kl
from relic.model import chat, load_model, make_lora
from relic.router import (
    RouterV7,
    key_text,
    normalize,
    resolve_reference,
    usable_aliases,
)
from relic.serving import SYSTEM, base_answer, serve_expert

R = 16  # rank per fact
CAP = 64  # capacity of the merged adapter, in facts; routed serving has no such limit
CANARIES = ["What is the capital of France?", "What is 7 times 8?"]
# never trained on, unlike CANARIES: the anchors understate an adapter's effect 17-fold (SS69)
HELDOUT_CANARIES = [
    "What is the capital of Japan?",
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water in Celsius?",
    "How many continents are there?",
    "What is the largest planet in the solar system?",
    "Who painted the Mona Lisa?",
    "What is the chemical symbol for gold?",
    "In which year did World War II end?",
    "What language is spoken in Brazil?",
    "What is the square root of 81?",
    "What is 12 plus 15?",
    "What is 9 times 6?",
    "What is 100 divided by 4?",
    "Is 17 a prime number?",
    "Translate 'thank you' into Spanish.",
    "Give a synonym for 'happy'.",
    "What is the opposite of 'cold'?",
    "Write one sentence about the ocean.",
    "List three primary colors.",
    "What day comes after Monday?",
    "What is the currency of the United Kingdom?",
    "Which organ pumps blood through the body?",
    "What gas do plants absorb from the air?",
    "How many legs does a spider have?",
    "What is the freezing point of water in Fahrenheit?",
    "Who developed the theory of relativity?",
    "What is the tallest mountain on Earth?",
    "Which ocean is the largest?",
    "What is the main language of Germany?",
    "Name a programming language.",
    "What does CPU stand for?",
    "What is the past tense of 'go'?",
    "How many minutes are in an hour?",
    "What is the capital of Canada?",
    "Türkiye'nin başkenti neresidir?",
    "Quelle est la capitale de l'Italie ?",
    "Wie viele Tage hat eine Woche?",
    "¿Cuál es el color del cielo?",
    "Summarize: The cat sat on the mat.",
    "Answer yes or no: is the sun a star?",
]
# p99 of held-out divergence over stored adapters on Qwen2.5-1.5B (SS69); calibrate per model (SS98)
KAPPA = 0.968

PROMPT_FIELDS = ("train_system", "anchor_system", "system")
# The prompts of every record up to entry 159: training pairs without a system message
# (Qwen's template then adds its default one), the KL anchor and the divergence check under
# a Turkish instruction, serving under "Answer briefly.".
LEGACY_PROMPTS = {
    "train_system": None,
    "anchor_system": "Kısa ve net cevap ver.",
    "system": "Answer briefly.",
}


def use_prompts(store, prompts: dict):
    """Set the store's system prompts, e.g. LEGACY_PROMPTS to reproduce an old entry."""
    unknown = set(prompts) - set(PROMPT_FIELDS)
    if unknown:
        raise ValueError(f"unknown prompt fields: {sorted(unknown)}")
    for k, v in prompts.items():
        setattr(store, k, v)
    return store


def chat_messages(system: str | None, *turns: dict) -> list[dict]:
    head = [] if system is None else [{"role": "system", "content": system}]
    return head + list(turns)


class DeltaStore:
    # one system prompt for training, the anchor, the divergence check and serving; class
    # attributes, so stores built with __new__ have them too (use_prompts overrides them)
    train_system: str | None = SYSTEM
    anchor_system: str | None = SYSTEM
    system: str = SYSTEM
    revision: str | None = None

    def __init__(
        self,
        model_name: str,
        lr: float = 3e-4,
        steps: int = 16,
        seed: int = 0,
        kl_lambda: float = 1.0,
        tau: float = 0.0,
        kl_prompts=None,
        capacity: int = CAP,
    ):
        self.model_name, self.lr, self.steps, self.seed = model_name, lr, steps, seed
        self.kl_lambda = kl_lambda
        self.tau = tau
        self.kl_prompts = list(CANARIES if kl_prompts is None else kl_prompts)
        self._base_logits: list | None = None
        self.tok, base = load_model(model_name)
        # routed serving activates one delta at a time and needs capacity 1; spare slots
        # hold zeros, which change no output but cost every forward pass their width
        self.capacity = capacity
        self.model = make_lora(base, r=capacity * R)
        self.deltas: dict[str, dict] = {}
        self.keys: dict[str, str] = {}
        self.notes: dict[str, str] = {}
        self.key_vecs: dict[str, torch.Tensor] = {}
        # encoder, threshold and any other router setting that decides routing; set by
        # the caller, hashed into the state
        self.router_config = ""
        self.revision = getattr(getattr(base, "config", None), "_commit_hash", None)
        if self.revision is None:
            warnings.warn(
                f"{model_name}: the model revision is unknown (no config._commit_hash),"
                " so the state hash cannot tell two revisions of these weights apart",
                UserWarning,
                stacklevel=2,
            )
        # the hash of stores saved before the model revision was hashed (format 1)
        self.legacy_base_hash = hashlib.sha256(
            f"{model_name}|r{R}|cap{capacity}".encode()
        ).hexdigest()
        self.base_hash = hashlib.sha256(
            f"{model_name}|r{R}|cap{capacity}|rev={self.revision}".encode()
        ).hexdigest()
        self._zero()

    # -- state ------------------------------------------------------------
    def _zero(self) -> None:
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if "lora_" in n:
                    p.zero_()

    def materialize(self) -> None:
        self._zero()
        if len(self.deltas) > self.capacity:  # nothing to merge; serving picks one
            return
        with torch.no_grad():
            params = dict(self.model.named_parameters())
            for idx, fid in enumerate(sorted(self.deltas)):
                for n, ab in self.deltas[fid].items():
                    params[n][idx * R : (idx + 1) * R, :] = ab["A"]
                    params[n.replace("lora_A", "lora_B")][
                        :, idx * R : (idx + 1) * R
                    ] = ab["B"]

    def state_hash(self) -> str:
        h = hashlib.sha256(self.base_hash.encode())

        # length-prefixed, so field boundaries are unambiguous
        def text(s: str) -> None:
            b = s.encode()
            h.update(f"{len(b)}:".encode() + b)

        text(self.router_config)
        for fid in sorted(set(self.deltas) | set(self.keys) | set(self.notes)):
            text(fid)
            text(self.notes.get(fid, ""))
            text(self.keys.get(fid, ""))
            delta = self.deltas.get(fid, {})
            text(str(len(delta)))
            for n in sorted(delta):
                for t in (delta[n]["A"], delta[n]["B"]):
                    h.update(t.float().contiguous().numpy().tobytes())
        return h.hexdigest()

    # -- transactions -----------------------------------------------------
    def _logits(self, model, q: str) -> torch.Tensor:
        ids = self.tok.apply_chat_template(
            chat_messages(self.anchor_system, {"role": "user", "content": q}),
            add_generation_prompt=True,
            return_tensors="pt",
        )
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(model.device)
        return model(input_ids=ids).logits[0, -1].float()

    def _pair_step(self, model, opt, user: str, assistant: str) -> None:
        turns = (
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        )
        prompt = self.tok.apply_chat_template(
            chat_messages(self.train_system, *turns[:1]),
            add_generation_prompt=True,
            return_tensors="pt",
        )
        full = self.tok.apply_chat_template(
            chat_messages(self.train_system, *turns), return_tensors="pt"
        )
        pid = prompt["input_ids"] if hasattr(prompt, "keys") else prompt
        fid = full["input_ids"] if hasattr(full, "keys") else full
        fid = fid.to(model.device)
        labels = fid.clone()
        labels[:, : pid.shape[1]] = -100
        out = model(input_ids=fid, labels=labels)
        loss = out.loss
        if self.kl_lambda and self._base_logits is not None:
            kl = 0.0
            for q, b in zip(self.kl_prompts, self._base_logits):
                pl = torch.log_softmax(self._logits(model, q), -1)
                pb = torch.softmax(b, -1)
                kl = kl + (pb * (pb.log() - pl)).sum()
            loss = loss + self.kl_lambda * kl / len(self.kl_prompts)
        loss.backward()
        opt.step()
        opt.zero_grad()

    def _train_delta(self, pairs) -> dict:
        # the store model is parked on CPU while a candidate trains: peak memory halves
        self.model.to("cpu")
        torch.cuda.empty_cache()
        try:
            return self._train_candidate(pairs)
        finally:
            self.model.to("cuda:0")

    def _train_candidate(self, pairs) -> dict:
        base = load_model(self.model_name)[1]
        m = make_lora(base, r=R)
        opt = torch.optim.AdamW(
            [p for p in m.parameters() if p.requires_grad], lr=self.lr
        )
        with torch.no_grad():
            m.eval()
            self._base_logits = [self._logits(m, q) for q in self.kl_prompts]
        m.train()
        for _ in range(self.steps):
            for user, assistant in pairs:
                self._pair_step(m, opt, user, assistant)
        self._base_logits = None
        delta: dict = {}
        for n, p in dict(m.named_parameters()).items():
            if "lora_A" in n:
                delta[n] = {"A": p.detach().float().cpu().clone()}
        for n, p in dict(m.named_parameters()).items():
            if "lora_B" in n:
                delta[n.replace("lora_B", "lora_A")]["B"] = (
                    p.detach().float().cpu().clone()
                )
        del m, base
        torch.cuda.empty_cache()
        return delta

    @torch.no_grad()
    def embed(self, text: str) -> torch.Tensor:
        """Semantic key: the last-token hidden state of the base model, normalised."""
        full = self.deltas
        self.deltas = {}
        self.materialize()
        ids = self.tok.apply_chat_template(
            chat_messages(self.anchor_system, {"role": "user", "content": text}),
            add_generation_prompt=True,
            return_tensors="pt",
        )
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(self.model.device)
        out = self.model(input_ids=ids, output_hidden_states=True)
        v = out.hidden_states[-1][0, -1].float()
        self.deltas = full
        self.materialize()
        return v / v.norm()

    def add(
        self, fid: str, pairs, key: str | None = None, note: str | None = None
    ) -> float:
        t = time.time()
        self.deltas[fid] = self._train_delta(pairs)
        self.keys[fid] = key or pairs[0][0]
        if note is not None:
            self.notes[fid] = note
        self.key_vecs[fid] = self.embed(self.keys[fid])
        self.materialize()
        return time.time() - t

    def revoke(self, fid: str) -> float:
        t = time.time()
        del self.deltas[fid]
        self.keys.pop(fid, None)
        self.notes.pop(fid, None)
        self.key_vecs.pop(fid, None)
        self.materialize()
        return time.time() - t

    def _expert_kld(self, delta: dict, prompts=None) -> float:
        """Canary KLD of a single candidate expert vs the zeroed base."""
        prompts = self.kl_prompts if prompts is None else prompts
        full = self.deltas
        with torch.no_grad():
            self.deltas = {"__tmp__": delta}
            self.materialize()
            a = [self._logits(self.model, q) for q in prompts]
            self.deltas = {}
            self.materialize()
            b = [self._logits(self.model, q) for q in prompts]
        self.deltas = full
        self.materialize()
        ks = [sym_kl(x, y) for x, y in zip(b, a)]
        return sum(ks) / len(ks)

    def propose_and_commit(
        self,
        fid: str,
        pairs,
        key: str | None = None,
        kld_limit: float = 2.0,
        note: str | None = None,
    ) -> dict:
        """G1: train a candidate, commit only if its canary footprint is within limit.

        Atomic on failure: the delta is local until the commit block; the key is
        resolved before anything is written, and a commit is all-or-nothing.
        """
        key = key or pairs[0][0]
        delta = self._train_delta(pairs)
        kld = self._expert_kld(delta)
        if not kld <= kld_limit:  # a NaN divergence is rejected, not committed
            return {"committed": False, "kld": kld}
        self.deltas[fid] = delta
        self.keys[fid] = key
        if note is not None:
            self.notes[fid] = note
        self.key_vecs[fid] = self.embed(key)
        self.materialize()
        return {"committed": True, "kld": kld}

    def answer_routed(self, user: str, system: str | None = None) -> str:
        """Route to ONE expert (semantic keys, cosine), never the merged sum."""
        ids = sorted(self.deltas)
        if not ids:
            return self.answer(user, system)
        qv = self.embed(user)
        scores = torch.tensor([float(qv @ self.key_vecs[i]) for i in ids])
        if float(scores.max()) < self.tau:  # abstain: no expert, base answer only
            full = self.deltas
            self.deltas = {}
            self.materialize()
            resp = self.answer(user, system)
            self.deltas = full
            self.materialize()
            return resp
        chosen = ids[int(scores.argmax())]
        full = self.deltas
        self.deltas = {chosen: full[chosen]}
        self.materialize()
        resp = self.answer(user, system)
        self.deltas = full
        self.materialize()
        return resp

    # -- behaviour --------------------------------------------------------
    def answer(self, user: str, system: str | None = None) -> str:
        return chat(
            self.tok,
            self.model,
            chat_messages(
                self.system if system is None else system,
                {"role": "user", "content": user},
            ),
            max_new_tokens=64,
        )


def _hash_text(h, s: str) -> None:
    b = s.encode()
    h.update(f"{len(b)}:".encode() + b)


def _delta_digest(delta: dict) -> str:
    """Digest of an fp32 delta, as format-1 state hashes computed it."""
    h = hashlib.sha256()
    for n in sorted(delta):
        _hash_text(h, n)
        for t in (delta[n]["A"], delta[n]["B"]):
            h.update(t.float().contiguous().numpy().tobytes())
    return h.hexdigest()


def packed_digest(packed: dict) -> str:
    """Digest of a stored adapter record: the same record gives the same digest."""
    h = hashlib.sha256()
    _hash_text(h, packed["format"])
    for n in sorted(packed["tensors"]):
        _hash_text(h, n)
        for k in sorted(packed["tensors"][n]):
            t = packed["tensors"][n][k].contiguous()
            _hash_text(h, f"{k}|{t.dtype}|{tuple(t.shape)}")
            h.update(t.numpy().tobytes())
    return h.hexdigest()


def _blob(packed: dict | None) -> dict | None:
    return None if packed is None else record_to_blob(packed)


def _record(stored: dict | None) -> dict | None:
    return None if stored is None else blob_to_record(stored)


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def _digest_texts(texts) -> str:
    h = hashlib.sha256()
    for t in texts:
        _hash_text(h, t)
    return h.hexdigest()


def _legacy_state_hash(
    base_hash, router_config, limits, relation_tau, entries, phrasings, digests
) -> str:
    """The state hash of format-1 files, exactly as the ledger computed it before v2."""
    h = hashlib.sha256()
    limits = [*limits, *(f"{r}={t}" for r, t in sorted(relation_tau.items()))]
    for s in (base_hash, router_config, *map(str, limits)):
        _hash_text(h, s)
    for fid in sorted(entries):
        entry = entries[fid]
        fields = [fid, entry["subject"], entry["answer"], entry["teach"]]
        fields.append(entry.get("relation", ""))
        names = sorted(entry.get("aliases", ()))
        for s in [*fields, entry["probe"], *sorted(phrasings[fid]), *names]:
            _hash_text(h, s)
        _hash_text(h, digests.get(fid, ""))
    return h.hexdigest()


class Ledger:
    """The entries of a store, with one active entry per (subject, relation) slot.

    A write whose subject is stored and whose relation the scorer cannot tell apart from
    a stored relation of that subject is not a new entry. With the same answer it adds its
    phrasing to the entry, with another answer it replaces the entry and keeps its
    phrasings (prereg SS96, SS104).

    Every write is kept: `writes[fid]` lists the writes an entry holds, its own first, and
    `history[fid]` the entries it superseded, oldest first. `revoke(fid)` removes a fact
    however many times it was written and brings back the entry it superseded, so removing
    an update restores the store as it was before the update. `retract(write_id)` removes
    one write and leaves the store as if that write had never happened.
    `keep_history=False` restores the behavior of the records up to entry 159: an update
    drops the entry it replaces, and a removal brings nothing back.

    Slot decision and store change happen under one lock, so commits from several threads
    have a serial order and each one decides against the state left by the one before.
    Serving swaps the active adapter in place and holds the same lock. This covers
    threads of one process; several processes on one file are not coordinated.
    """

    def __init__(
        self,
        store,
        scorer,
        tau: float,
        slot_theta: float,
        duplicate_theta: float | None = None,
        relation_tau: dict | None = None,
        kappa: float = KAPPA,
        keep_history: bool = True,
    ):
        self.store, self.scorer = store, scorer
        self.tau, self.slot_theta = tau, slot_theta
        self.relation_tau = dict(relation_tau or {})
        # SS104: a same-answer write merges wherever a query would be routed to the entry
        self.duplicate_theta = tau if duplicate_theta is None else duplicate_theta
        self.kappa = kappa
        self.keep_history = keep_history
        self.entries: dict[str, dict] = {}
        self.phrasings: dict[str, list[str]] = {}
        # {entry: {phrasing: the write id that introduced it, or the entry it came from}}
        self.owners: dict[str, dict[str, str]] = {}
        self.writes: dict[str, list[dict]] = {}
        self.history: dict[str, list[dict]] = {}
        self.packed: dict[str, dict] = {}
        self._delta_digest: dict[str, str] = {}
        self.log: list[dict] = []
        self.lock = threading.RLock()
        self._router = None

    # -- slots ------------------------------------------------------------
    def slot(self, fact: dict) -> str | None:
        """The entry that holds this fact's subject and relation, or None.

        A same-answer write needs `duplicate_theta`: merging it loses nothing stored. A
        different answer needs `slot_theta`, because replacing an entry drops a fact.
        """
        subject = normalize(fact["subject"]).strip()
        relation = key_text(fact["probe"], fact["subject"])
        best, best_score = None, float("-inf")
        for fid in sorted(self.entries):
            entry = self.entries[fid]
            names = [entry["subject"], *usable_aliases(entry.get("aliases", ()))]
            if subject not in {normalize(n).strip() for n in names}:
                continue
            prompts = [entry["probe"], *self.phrasings[fid]]
            keys = [key_text(p, entry["subject"]) for p in prompts]
            score = max(self.scorer.sims(relation, keys))
            limit = (
                self.duplicate_theta
                if self._same_answer(fid, fact)
                else self.slot_theta
            )
            if score >= limit and score > best_score:
                best, best_score = fid, score
        return best

    def _same_answer(self, fid: str, fact: dict) -> bool:
        stored = self.entries[fid]["answer"]
        return normalize(stored).strip() == normalize(fact["answer"]).strip()

    # -- bookkeeping ------------------------------------------------------
    def ids(self) -> set[str]:
        """Every id in use: active entries, their writes, and superseded entries."""
        return set(self._locations())

    def _locations(self) -> dict[str, str]:
        """{id: the active entry whose record or history holds it}."""
        out = {}
        for fid in self.entries:
            out[fid] = fid
            for w in self.writes.get(fid, []):
                out[w["fact"]["id"]] = fid
            for snap in self.history.get(fid, []):
                out[snap["entry"]["id"]] = fid
                for w in snap["writes"]:
                    out[w["fact"]["id"]] = fid
        for fid, stack in self.history.items():  # a history whose owner is not active
            for snap in stack:
                out.setdefault(snap["entry"]["id"], fid)
                for w in snap["writes"]:
                    out.setdefault(w["fact"]["id"], fid)
        return out

    def _check_new_id(self, fid: str) -> None:
        if fid in self.ids():
            raise ValueError(f"{fid}: id is held by another fact or write")

    @staticmethod
    def _check_finite(delta: dict) -> None:
        tensors = delta["tensors"].values() if "format" in delta else delta.values()
        for ts in tensors:
            for t in ts.values():
                if t.is_floating_point() and not bool(torch.isfinite(t).all()):
                    raise ValueError("non-finite adapter: refused")

    def _set_delta(self, fid: str, delta: dict) -> None:
        """Keep the stored record and the padded adapter that serving swaps in."""
        if "format" in delta:
            packed = delta
            self.store.deltas[fid] = pad(unpack_delta(packed))
        else:
            packed = pack_delta(trim_rank(delta))
            self.store.deltas[fid] = delta
        self.packed[fid] = packed
        self._delta_digest[fid] = packed_digest(packed)

    def _snapshot(self, fid: str) -> dict:
        return {
            "entry": self.entries[fid],
            "phrasings": list(self.phrasings[fid]),
            "owners": dict(self.owners.get(fid, {})),
            "writes": [dict(w) for w in self.writes.get(fid, [])],
            "packed": self.packed.get(fid),
            "digest": self._delta_digest.get(fid),
        }

    def _remove_active(self, fid: str) -> None:
        del self.entries[fid], self.phrasings[fid]
        self.owners.pop(fid, None)
        self.writes.pop(fid, None)
        self.packed.pop(fid, None)
        self._delta_digest.pop(fid, None)
        self.store.deltas.pop(fid, None)
        self.store.keys.pop(fid, None)
        self._router = None

    def _activate(self, snap: dict) -> str:
        entry = snap["entry"]
        fid = entry["id"]
        self.entries[fid] = entry
        self.phrasings[fid] = list(snap["phrasings"])
        self.owners[fid] = dict(snap["owners"])
        self.writes[fid] = [dict(w) for w in snap["writes"]]
        self.store.keys[fid] = entry["probe"]
        if snap["packed"] is not None:
            self._set_delta(fid, snap["packed"])
        self._router = None
        return fid

    def _records(self):
        """(phrasings, owners, writes) of every active entry and every snapshot."""
        for fid in self.entries:
            owners = self.owners.setdefault(fid, {})
            yield self.phrasings[fid], owners, self.writes.get(fid, [])
        for stack in self.history.values():
            for snap in stack:
                yield snap["phrasings"], snap["owners"], snap["writes"]

    def _drop_owned(self, removed: set[str]) -> None:
        """Remove the phrasings that the removed writes introduced, wherever they live.

        A phrasing that another remaining write of the same record also gives passes to
        that write, as it would have been added by it had the removed write never happened.
        """
        for phrasings, owners, writes in self._records():
            for p in list(phrasings):
                if owners.get(p, None) not in removed:
                    continue
                heir = next(
                    (
                        w
                        for w in writes[1:]
                        if w["fact"]["probe"] == p and w["fact"]["id"] not in removed
                    ),
                    None,
                )
                if heir is not None:
                    owners[p] = heir["fact"]["id"]
                    heir["added_phrasing"] = True
                else:
                    phrasings.remove(p)
                    del owners[p]

    def _checkpoint(self) -> dict:
        """Everything a transaction can change, to put back if it fails halfway."""
        return {
            "entries": dict(self.entries),
            "phrasings": {k: list(v) for k, v in self.phrasings.items()},
            "owners": {k: dict(v) for k, v in self.owners.items()},
            "writes": {k: [dict(w) for w in v] for k, v in self.writes.items()},
            "history": {
                k: [
                    {
                        **s,
                        "phrasings": list(s["phrasings"]),
                        "owners": dict(s["owners"]),
                        "writes": [dict(w) for w in s["writes"]],
                    }
                    for s in v
                ]
                for k, v in self.history.items()
            },
            "packed": dict(self.packed),
            "digests": dict(self._delta_digest),
            "deltas": dict(self.store.deltas),
            "keys": dict(self.store.keys),
            "log": len(self.log),
        }

    def _restore(self, cp: dict) -> None:
        self.entries, self.phrasings = cp["entries"], cp["phrasings"]
        self.owners, self.writes, self.history = (
            cp["owners"],
            cp["writes"],
            cp["history"],
        )
        self.packed, self._delta_digest = cp["packed"], cp["digests"]
        self.store.deltas.clear()
        self.store.deltas.update(cp["deltas"])
        self.store.keys.clear()
        self.store.keys.update(cp["keys"])
        del self.log[cp["log"] :]
        self._router = None

    def _rebuild_router(self) -> None:
        self._router = None
        _ = self.router

    def _record(self, action: str, fid: str, **extra) -> dict:
        event = {
            "seq": len(self.log),
            "action": action,
            "entry": fid,
            **extra,
            "state_hash": self.state_hash(),
        }
        self.log.append(event)
        return event

    # -- transactions -----------------------------------------------------
    def commit(self, fact: dict, delta: dict | None = None, **extra) -> dict:
        """Apply one write atomically: insert, duplicate (new phrasing) or update.

        `delta` is an adapter ({module: {"A", "B"}}) or a stored record (pack_delta);
        an adapter with a non-finite value is refused. `extra` goes into the log record.
        """
        with self.lock:
            fid = fact["id"]
            self._check_new_id(fid)
            if delta is not None:
                self._check_finite(delta)
            old = self.slot(fact)
            if old is not None and self._same_answer(old, fact):
                added = fact["probe"] != self.entries[old]["probe"] and (
                    fact["probe"] not in self.phrasings[old]
                )
                if added:
                    self.phrasings[old].append(fact["probe"])
                    self.owners.setdefault(old, {})[fact["probe"]] = fid
                self.writes[old].append({"fact": fact, "added_phrasing": added})
                self._router = None
                return self._record("duplicate", old, written=fid, **extra)
            kept, stack = [], []
            if old is not None:
                # the replaced entry's prompts stay as phrasings, each with its owner
                inherited = self.owners.get(old, {})
                kept = [(self.entries[old]["probe"], old)]
                kept += [(p, inherited.get(p, old)) for p in self.phrasings[old]]
                if self.keep_history:
                    stack = self.history.pop(old, []) + [self._snapshot(old)]
                self._remove_active(old)
            kept = [(p, o) for p, o in kept if p != fact["probe"]]
            self.entries[fid] = fact
            self.phrasings[fid] = [p for p, _ in kept]
            self.owners[fid] = dict(kept)
            self.writes[fid] = [{"fact": fact, "added_phrasing": False}]
            if stack:
                self.history[fid] = stack
            self.store.keys[fid] = fact["probe"]
            if delta is not None:
                self._set_delta(fid, delta)
            self._router = None
            if old is None:
                return self._record("insert", fid, **extra)
            return self._record("update", fid, superseded=old, **extra)

    def _revoke_active(self, fid: str) -> str | None:
        """Remove an active entry; the entry it superseded, if any, becomes active again."""
        stack = self.history.pop(fid, [])
        self._remove_active(fid)
        if not (stack and self.keep_history):
            return None
        restored = self._activate(stack.pop())
        if stack:
            self.history[restored] = stack
        return restored

    def _history_owner(self, fid: str) -> tuple[str, int] | None:
        for owner in sorted(self.history):
            for i, snap in enumerate(self.history[owner]):
                if snap["entry"]["id"] == fid:
                    return owner, i
        return None

    def _purge(self, owner: str, i: int) -> None:
        """Erase a superseded entry: its snapshot, and every phrasing its writes gave."""
        snap = self.history[owner].pop(i)
        if not self.history[owner]:
            del self.history[owner]
        self._drop_owned(
            {w["fact"]["id"] for w in snap["writes"]} | {snap["entry"]["id"]}
        )

    def revoke(self, fid: str) -> dict:
        """Remove a fact. An active entry goes with all its writes, and the entry it
        superseded comes back; a superseded entry is erased from the history, with the
        phrasings its writes gave to the entries after it."""
        with self.lock:
            t0 = time.perf_counter()
            if fid in self.entries:
                restored = self._revoke_active(fid)
                self._rebuild_router()
                seconds = round(time.perf_counter() - t0, 6)
                return self._record("revoke", fid, restored=restored, seconds=seconds)
            found = self._history_owner(fid)
            if found is None:
                raise KeyError(
                    f"{fid}: no entry or superseded entry has this id"
                    " (a duplicate write is removed with retract)"
                )
            self._purge(*found)
            self._rebuild_router()
            seconds = round(time.perf_counter() - t0, 6)
            return self._record(
                "revoke", fid, purged_from=found[0], restored=None, seconds=seconds
            )

    def _can_train(self) -> bool:
        return getattr(self.store, "model", None) is not None

    def _find_write(self, write_id: str):
        """(active entry, None, index) or (active entry, history index, index)."""
        for fid in sorted(self.entries):
            for k, w in enumerate(self.writes.get(fid, [])):
                if w["fact"]["id"] == write_id:
                    return fid, None, k
            for h, snap in enumerate(self.history.get(fid, [])):
                for k, w in enumerate(snap["writes"]):
                    if w["fact"]["id"] == write_id:
                        return fid, h, k
        return None

    def retract(self, write_id: str) -> dict:
        """Remove one write, as if it had never been made.

        A duplicate is dropped with the phrasings it gave, unless another write of the
        same record also gives them. The own write of an active entry is replaced by the
        next write, written again (trained when the ledger has a model), so no adapter
        trained on the retracted write survives; the own write of a superseded entry is
        replaced likewise inside the history. The last write of an entry is a revoke. A
        failure halfway leaves the ledger as it was.
        """
        with self.lock:
            t0 = time.perf_counter()
            found = self._find_write(write_id)
            if found is None:
                raise KeyError(f"{write_id}: no entry or history holds this write")
            owner, h, i = found
            cp = self._checkpoint()
            try:
                if h is None:
                    extra = self._retract_active(owner, i, write_id)
                else:
                    extra = self._retract_history(owner, h, i, write_id)
            except BaseException:
                self._restore(cp)
                raise
            self._rebuild_router()
            extra["seconds"] = round(time.perf_counter() - t0, 6)
            return self._record("retract", owner, write=write_id, **extra)

    def _replay(self, facts: list[dict], cause: str) -> tuple[str | None, list[str]]:
        """Write the facts again in order: (the write stored as an entry, rejected)."""
        stored, rejected = None, []
        for f in facts:
            if self._can_train():
                event = self.write(f, cause=cause)
            else:
                event = self.commit(f, cause=cause)
            if event["action"] == "reject":
                rejected.append(f["id"])
            elif stored is None and event["action"] in ("insert", "update"):
                stored = f["id"]
        return stored, rejected

    def _retract_active(self, owner: str, i: int, write_id: str) -> dict:
        ws = self.writes[owner]
        if len(ws) == 1:
            return {"mode": "last", "restored": self._revoke_active(owner)}
        if i > 0:
            ws.pop(i)
            self._drop_owned({write_id})
            return {"mode": "duplicate"}
        rest = [w["fact"] for w in ws[1:]]
        self._revoke_active(owner)
        stored, rejected = self._replay(rest, f"retract:{write_id}")
        return {"mode": "rebuilt", "rebuilt_from": stored, "rejected_replays": rejected}

    def _retract_history(self, owner: str, h: int, i: int, write_id: str) -> dict:
        snap = self.history[owner][h]
        ws = snap["writes"]
        if len(ws) == 1:
            self._purge(owner, h)
            return {"mode": "history-last"}
        if i > 0:
            ws.pop(i)
            self._drop_owned({write_id})
            return {"mode": "history-duplicate"}
        # the superseded entry is rebuilt from its next write; its adapter is retrained
        # when the ledger has a model, and dropped otherwise
        rejected = []
        for k, w in enumerate(ws[1:], 1):
            packed = self._ladder(w["fact"])["packed"] if self._can_train() else None
            if self._can_train() and packed is None:
                rejected.append(w["fact"]["id"])
                continue
            heir = w["fact"]
            snap["entry"] = heir
            snap["writes"] = [{"fact": heir, "added_phrasing": False}] + [
                x for x in ws[k + 1 :]
            ]
            snap["packed"] = packed
            snap["digest"] = None if packed is None else packed_digest(packed)
            if heir["probe"] in snap["phrasings"]:  # now the entry's own prompt
                snap["phrasings"].remove(heir["probe"])
                snap["owners"].pop(heir["probe"], None)
            self._drop_owned({write_id, *rejected})
            return {
                "mode": "history-rebuilt",
                "rebuilt_from": heir["id"],
                "rejected_replays": rejected,
            }
        self._purge(owner, h)
        return {"mode": "history-last", "rejected_replays": rejected}

    def _rewrites(self, fact: dict) -> list[str]:
        """Phrasings by the base model, from the subject and the edit prompt alone."""
        store, full = self.store, self.store.deltas
        store.deltas = {}
        store.materialize()
        prompt = REWRITE_USER.format(subject=fact["subject"], probe=fact["probe"])
        text = chat(
            store.tok,
            store.model,
            [
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=96,
        )
        store.deltas = full
        store.materialize()
        return parse_rewrites(text, fact["probe"], fact["subject"])

    def _ladder(self, fact: dict) -> dict:
        """Train the rungs in order; the first finite adapter with D <= kappa is kept.

        An adapter with a non-finite value is rejected before its divergence is read.
        """
        short = fact_pairs(fact)
        full = short + [(r, fact["answer"]) for r in self._rewrites(fact)]
        store, weight = self.store, self.store.kl_lambda
        out = {"packed": None, "tried": {}, "rejected_rungs": {}}
        try:
            for name, seed, kl_lambda, rewrites in LADDER:
                store.kl_lambda = kl_lambda
                try:
                    packed = train_adapter_packed(
                        store, full if rewrites else short, seed
                    )
                    self._check_finite(packed)
                except NonFiniteAdapter:
                    out["rejected_rungs"][name] = "non-finite adapter"
                    continue
                except ValueError as e:
                    if "non-finite" not in str(e):
                        raise
                    out["rejected_rungs"][name] = "non-finite adapter"
                    continue
                kld = store._expert_kld(pad(unpack_delta(packed)), HELDOUT_CANARIES)
                out["tried"][name] = kld
                if kld <= self.kappa:  # a NaN divergence is rejected, not committed
                    out.update(packed=packed, rung=name, kld=kld, seed=seed)
                    return out
        finally:
            store.kl_lambda = weight
        return out

    def write(self, fact: dict, **extra) -> dict:
        """Train the fact's adapter: the first rung of the ladder under `self.kappa`.

        The store's one model is parked while a candidate trains, so a write holds the
        lock throughout. If no rung passes, the write commits nothing and the log records
        the rejection, with the reason of each rung. `extra` goes into the log record.
        """
        with self.lock:
            t0 = time.perf_counter()
            self._check_new_id(fact["id"])
            old = self.slot(fact)
            if old is not None and self._same_answer(old, fact):
                seconds = round(time.perf_counter() - t0, 3)
                return self.commit(fact, seconds=seconds, **extra)
            out = self._ladder(fact)
            seconds = round(time.perf_counter() - t0, 3)
            info = {"tried": out["tried"], "rejected_rungs": out["rejected_rungs"]}
            if out["packed"] is None:
                return self._record(
                    "reject", fact["id"], **info, seconds=seconds, **extra
                )
            return self.commit(
                fact,
                out["packed"],
                rung=out["rung"],
                kld=out["kld"],
                seed=out["seed"],
                **info,
                seconds=seconds,
                **extra,
            )

    # -- reads ------------------------------------------------------------
    @property
    def router(self):
        if self._router is None:
            facts = [self.entries[fid] for fid in sorted(self.entries)]
            extra = {
                f"{f['subject']}\t{f['probe']}": self.phrasings[f["id"]] for f in facts
            }
            router = RouterV7(facts, self.scorer, self.tau, extra, self.relation_tau)
            self._router = (router, facts)
        return self._router

    def ask(self, query: str, history=()) -> dict:
        """The answer and the entry that produced it (None: the unmodified base model).

        `history` holds the earlier user turns; a pronoun in the query is resolved against
        them, and the model sees the resolved question.
        """
        with self.lock:
            router, facts = self.router
            query = resolve_reference(query, history, router)
            j, score = router.route(query)
            fid = None if j is None else facts[j]["id"]
            if fid not in self.store.deltas:  # an entry acts only through its adapter
                return {
                    "answer": base_answer(self.store, query),
                    "entry": None,
                    "score": score,
                }
            answer = serve_expert(self.store, fid, query)
            return {"answer": answer, "entry": fid, "score": score}

    def _config(self) -> list[str]:
        """Everything outside the entries that decides how the store behaves."""
        st = self.store
        parts = ["state-hash-v2", st.base_hash, str(getattr(st, "revision", None))]
        parts.append(st.router_config)
        parts += [repr(self.tau), repr(self.slot_theta), repr(self.duplicate_theta)]
        parts += [f"{r}={t!r}" for r, t in sorted(self.relation_tau.items())]
        parts += [f"kappa={self.kappa!r}", f"keep_history={self.keep_history}"]
        parts += [_canon(LADDER), _canon(R2), f"b_mult={B_MULT}"]
        parts += [f"steps={getattr(st, 'steps', None)!r}"]
        parts += [f"lr={getattr(st, 'lr', None)!r}"]
        parts += [f"{k}={getattr(st, k, None)!r}" for k in PROMPT_FIELDS]
        parts.append(_digest_texts(HELDOUT_CANARIES))
        parts.append(_digest_texts(getattr(st, "kl_prompts", CANARIES)))
        return parts

    @staticmethod
    def _hash_record(h, fid, entry, phrasings, owners, writes, digest) -> None:
        _hash_text(h, fid)
        _hash_text(h, _canon(entry))
        phrasings = sorted(phrasings)
        _hash_text(h, str(len(phrasings)))
        for p in phrasings:
            _hash_text(h, p)
            _hash_text(h, str(owners.get(p)))
        ws = sorted(writes, key=lambda w: w["fact"]["id"])
        _hash_text(h, str(len(ws)))
        for w in ws:
            _hash_text(h, _canon(w["fact"]))
            _hash_text(h, str(w["added_phrasing"]))
        _hash_text(h, digest or "")

    def state_hash(self) -> str:
        """A function of the state: the configuration, and each entry with its writes,
        phrasings, adapter record and the entries it superseded.

        A delta is hashed once when it is committed; hashing every tensor of a large
        store on each commit would make a commit linear in the size of the store.
        """
        h = hashlib.sha256()
        for s in self._config():
            _hash_text(h, s)
        for fid in sorted(self.entries):
            self._hash_record(
                h,
                fid,
                self.entries[fid],
                self.phrasings[fid],
                self.owners.get(fid, {}),
                self.writes.get(fid, []),
                self._delta_digest.get(fid),
            )
            stack = self.history.get(fid, [])
            _hash_text(h, f"history:{len(stack)}")
            for snap in stack:
                self._hash_record(
                    h,
                    snap["entry"]["id"],
                    snap["entry"],
                    snap["phrasings"],
                    snap["owners"],
                    snap["writes"],
                    snap["digest"],
                )
        return h.hexdigest()

    # -- durability -------------------------------------------------------
    def save(self, path) -> None:
        """Write the state to a new file and rename it: a crash leaves the old file whole.

        Adapters are written as their stored records (int8 rows, 373 KB for the adopted
        rank-2 adapter), not as the padded tensors that serving swaps in.
        """
        path = Path(path)
        tmp = path.with_name(f"{path.name}.tmp")
        with self.lock:
            state = {
                "format": 2,
                "config": self.config(),
                "entries": self.entries,
                "phrasings": self.phrasings,
                "owners": self.owners,
                "writes": self.writes,
                "history": {
                    fid: [{**snap, "packed": _blob(snap["packed"])} for snap in stack]
                    for fid, stack in self.history.items()
                },
                "packed": {fid: _blob(p) for fid, p in self.packed.items()},
                "log": self.log,
                "router_config": self.store.router_config,
                "state_hash": self.state_hash(),
            }
            with tmp.open("wb") as f:
                torch.save(state, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

    def config(self) -> dict:
        """The settings a ledger is built with, written into its file (Ledger.from_file)."""
        return {
            "tau": self.tau,
            "slot_theta": self.slot_theta,
            "duplicate_theta": self.duplicate_theta,
            "relation_tau": dict(self.relation_tau),
            "kappa": self.kappa,
            "keep_history": self.keep_history,
            "prompts": {k: getattr(self.store, k, None) for k in PROMPT_FIELDS},
        }

    @classmethod
    def from_file(cls, path, store, scorer) -> Ledger:
        """The ledger a file was written by: its settings and prompts, then its state.

        The state hash covers these settings, so building the ledger with others and
        calling load() refuses the file; this reads them from it first.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "config" not in state:
            raise ValueError(f"{path}: the file does not record its ledger settings")
        cfg = state["config"]
        use_prompts(store, cfg["prompts"])
        ledger = cls(
            store,
            scorer,
            cfg["tau"],
            cfg["slot_theta"],
            cfg["duplicate_theta"],
            cfg["relation_tau"],
            kappa=cfg["kappa"],
            keep_history=cfg["keep_history"],
        )
        ledger.load(path, state)
        return ledger

    def load(self, path, state: dict | None = None) -> None:
        if state is None:
            state = torch.load(path, map_location="cpu", weights_only=False)
        with self.lock:
            if "format" not in state:
                self._load_v1(path, state)
                return
            if state["format"] != 2:
                raise ValueError(f"{path}: unknown state format {state['format']!r}")
            self.entries, self.phrasings = state["entries"], state["phrasings"]
            self.owners = state.get("owners") or {
                f: {p: f for p in ps} for f, ps in self.phrasings.items()
            }
            self.writes = state["writes"]
            self.history = {
                fid: [
                    {
                        "owners": {p: snap["entry"]["id"] for p in snap["phrasings"]},
                        **snap,
                        "packed": _record(snap["packed"]),
                    }
                    for snap in stack
                ]
                for fid, stack in state["history"].items()
            }
            self.log = state["log"]
            self.store.router_config = state["router_config"]
            self.store.keys = {f: e["probe"] for f, e in self.entries.items()}
            self.store.deltas, self.packed, self._delta_digest = {}, {}, {}
            for fid, stored in state["packed"].items():
                self._set_delta(fid, _record(stored))
            for stack in self.history.values():
                for snap in stack:
                    snap["digest"] = (
                        None
                        if snap["packed"] is None
                        else packed_digest(snap["packed"])
                    )
            self._router = None
            if self.state_hash() != state["state_hash"]:
                raise ValueError(f"{path}: state hash does not match its contents")

    def _load_v1(self, path, state) -> None:
        """A file written before format 2: verify its own hash, then convert it.

        Format 1 kept no superseded entries, no duplicate writes and no owners of
        phrasings, so the converted ledger has an empty history, one write per entry, and
        each phrasing owned by its entry.
        """
        digests = {f: _delta_digest(d) for f, d in state["deltas"].items()}
        legacy = _legacy_state_hash(
            getattr(self.store, "legacy_base_hash", self.store.base_hash),
            state["router_config"],
            [self.tau, self.slot_theta, self.duplicate_theta],
            self.relation_tau,
            state["entries"],
            state["phrasings"],
            digests,
        )
        if legacy != state["state_hash"]:
            raise ValueError(f"{path}: state hash does not match its contents")
        self.entries, self.phrasings = state["entries"], state["phrasings"]
        self.owners = {f: {p: f for p in ps} for f, ps in self.phrasings.items()}
        self.writes = {
            f: [{"fact": e, "added_phrasing": False}] for f, e in self.entries.items()
        }
        self.history = {}
        self.log = state["log"]
        self.store.router_config = state["router_config"]
        self.store.keys = {f: e["probe"] for f, e in self.entries.items()}
        self.store.deltas, self.packed, self._delta_digest = {}, {}, {}
        for fid, delta in state["deltas"].items():
            self._set_delta(fid, pack_delta(trim_rank(delta)))
        self._router = None
