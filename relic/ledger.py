"""Transactional delta store (slice-1, docs/LIVING_MODEL_SLICE_PREREG.md).

Each fact -> one LoRA delta trained from the same frozen base; the live state is the
sum of the active deltas in id order, materialised into a fixed-capacity adapter.
`revoke` drops a delta and rebuilds; `state_hash` is the pure function of the active
set (identifier, note, routing key and delta of each entry, plus the router
configuration), so revoking the last addition restores the recorded pre-add hash exactly.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path

import torch

from relic.adapters import fact_pairs, train_adapter
from relic.metrics import hit, sym_kl
from relic.model import chat, load_model, make_lora
from relic.router import (
    RouterV7,
    key_text,
    normalize,
    resolve_reference,
    usable_aliases,
)
from relic.serving import base_answer, rag_system, serve_expert

R = 16  # rank per fact
CAP = 64  # capacity of the merged adapter, in facts; routed serving has no such limit
CANARIES = ["What is the capital of France?", "What is 7 times 8?"]


class DeltaStore:
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
        self.base_hash = hashlib.sha256(
            f"{model_name}|r{R}|cap{capacity}".encode()
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
            [
                {"role": "system", "content": "Kısa ve net cevap ver."},
                {"role": "user", "content": q},
            ],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(model.device)
        return model(input_ids=ids).logits[0, -1].float()

    def _pair_step(self, model, opt, user: str, assistant: str) -> None:
        pair = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        prompt = self.tok.apply_chat_template(
            pair[:-1], add_generation_prompt=True, return_tensors="pt"
        )
        full = self.tok.apply_chat_template(pair, return_tensors="pt")
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
            [
                {"role": "system", "content": "Kısa ve net cevap ver."},
                {"role": "user", "content": text},
            ],
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

    def _expert_kld(self, delta: dict) -> float:
        """Canary KLD of a single candidate expert vs the zeroed base."""
        full = self.deltas
        with torch.no_grad():
            self.deltas = {"__tmp__": delta}
            self.materialize()
            a = [self._logits(self.model, q) for q in self.kl_prompts]
            self.deltas = {}
            self.materialize()
            b = [self._logits(self.model, q) for q in self.kl_prompts]
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

    def answer_routed(self, user: str, system: str = "Kısa ve net cevap ver.") -> str:
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
    def answer(self, user: str, system: str = "Kısa ve net cevap ver.") -> str:
        return chat(
            self.tok,
            self.model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_new_tokens=24,
        )


def _hash_text(h, s: str) -> None:
    b = s.encode()
    h.update(f"{len(b)}:".encode() + b)


def _delta_digest(delta: dict) -> str:
    h = hashlib.sha256()
    for n in sorted(delta):
        _hash_text(h, n)
        for t in (delta[n]["A"], delta[n]["B"]):
            h.update(t.float().contiguous().numpy().tobytes())
    return h.hexdigest()


class Ledger:
    """The entries of a store, with one active entry per (subject, relation) slot.

    A write whose subject is stored and whose relation the scorer cannot tell apart from
    a stored relation of that subject is not a new entry. With the same answer it adds its
    phrasing to the entry, with another answer it replaces the entry and keeps its
    phrasings. Removing an entry therefore removes the fact, however many times it was
    written (prereg SS96, SS104).

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
    ):
        self.store, self.scorer = store, scorer
        self.tau, self.slot_theta = tau, slot_theta
        self.duplicate_theta = (
            slot_theta if duplicate_theta is None else duplicate_theta
        )
        self.entries: dict[str, dict] = {}
        self.phrasings: dict[str, list[str]] = {}
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

    # -- transactions -----------------------------------------------------
    def _drop(self, fid: str) -> None:
        del self.entries[fid], self.phrasings[fid]
        self._delta_digest.pop(fid, None)
        self.store.deltas.pop(fid, None)
        self.store.keys.pop(fid, None)
        self.store.notes.pop(fid, None)
        self._router = None

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

    def commit(self, fact: dict, delta: dict | None = None) -> dict:
        """Apply one write atomically: insert, duplicate (new phrasing) or update."""
        with self.lock:
            old = self.slot(fact)
            if old is not None and self._same_answer(old, fact):
                if fact["probe"] != self.entries[old]["probe"] and (
                    fact["probe"] not in self.phrasings[old]
                ):
                    self.phrasings[old].append(fact["probe"])
                self._router = None
                return self._record("duplicate", old, written=fact["id"])
            kept = []
            if old is not None:
                kept = [self.entries[old]["probe"], *self.phrasings[old]]
                self._drop(old)
            self.entries[fact["id"]] = fact
            self.phrasings[fact["id"]] = [p for p in kept if p != fact["probe"]]
            self.store.keys[fact["id"]] = fact["probe"]
            self.store.notes[fact["id"]] = fact["teach"]
            if delta is not None:
                self.store.deltas[fact["id"]] = delta
                self._delta_digest[fact["id"]] = _delta_digest(delta)
            self._router = None
            if old is None:
                return self._record("insert", fact["id"])
            return self._record("update", fact["id"], superseded=old)

    def revoke(self, fid: str) -> dict:
        with self.lock:
            self._drop(fid)
            return self._record("revoke", fid)

    def write(self, fact: dict, kld_limit: float = 2.0) -> dict:
        """The tiered write path: a note if it answers the edit prompt, else an adapter.

        The store's one model is parked while a candidate trains, so a write holds the
        lock throughout. A candidate above the divergence limit commits nothing.
        """
        with self.lock:
            old = self.slot(fact)
            if old is not None and self._same_answer(old, fact):
                return self.commit(fact)
            note = rag_system("instruct", [fact], 0)
            if hit(base_answer(self.store, fact["probe"], system=note), fact["answer"]):
                return self.commit(fact)
            delta = train_adapter(self.store, fact_pairs(fact))
            kld = self.store._expert_kld(delta)
            if not kld <= kld_limit:  # a NaN divergence is rejected, not committed
                return {"action": "rejected", "entry": fact["id"], "kld": kld}
            return {**self.commit(fact, delta), "kld": kld}

    # -- reads ------------------------------------------------------------
    @property
    def router(self):
        if self._router is None:
            facts = [self.entries[fid] for fid in sorted(self.entries)]
            extra = {
                f"{f['subject']}\t{f['probe']}": self.phrasings[f["id"]] for f in facts
            }
            self._router = (RouterV7(facts, self.scorer, self.tau, extra), facts)
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
            if j is None:
                answer = base_answer(self.store, query)
                return {"answer": answer, "entry": None, "score": score}
            fact = facts[j]
            note = rag_system("instruct", facts, j)
            if fact["id"] in self.store.deltas:
                answer = serve_expert(self.store, fact["id"], query, system=note)
            else:
                answer = base_answer(self.store, query, system=note)
            return {"answer": answer, "entry": fact["id"], "score": score}

    def state_hash(self) -> str:
        """A function of the active set: the model, the router setting and each entry.

        A delta is hashed once when it is committed; hashing every tensor of a large
        store on each commit would make a commit linear in the size of the store.
        """
        h = hashlib.sha256()
        for s in (self.store.base_hash, self.store.router_config):
            _hash_text(h, s)
        for fid in sorted(self.entries):
            entry = self.entries[fid]
            fields = [fid, entry["subject"], entry["answer"], entry["teach"]]
            names = sorted(entry.get("aliases", ()))
            for s in [*fields, entry["probe"], *sorted(self.phrasings[fid]), *names]:
                _hash_text(h, s)
            _hash_text(h, self._delta_digest.get(fid, ""))
        return h.hexdigest()

    # -- durability -------------------------------------------------------
    def save(self, path) -> None:
        """Write the state to a new file and rename it: a crash leaves the old file whole."""
        path = Path(path)
        tmp = path.with_name(f"{path.name}.tmp")
        with self.lock:
            state = {
                "entries": self.entries,
                "phrasings": self.phrasings,
                "log": self.log,
                "deltas": self.store.deltas,
                "router_config": self.store.router_config,
                "state_hash": self.state_hash(),
            }
            with tmp.open("wb") as f:
                torch.save(state, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

    def load(self, path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        with self.lock:
            self.entries, self.phrasings = state["entries"], state["phrasings"]
            self.log = state["log"]
            self.store.deltas = state["deltas"]
            self.store.router_config = state["router_config"]
            self.store.keys = {f: e["probe"] for f, e in self.entries.items()}
            self.store.notes = {f: e["teach"] for f, e in self.entries.items()}
            self._delta_digest = {
                f: _delta_digest(d) for f, d in state["deltas"].items()
            }
            self._router = None
            if self.state_hash() != state["state_hash"]:
                raise ValueError(f"{path}: state hash does not match its contents")
