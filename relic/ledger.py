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

from relic.adapters import (
    LADDER,
    REWRITE_SYSTEM,
    REWRITE_USER,
    fact_pairs,
    parse_rewrites,
    train_adapter,
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
        relation_tau: dict | None = None,
    ):
        self.store, self.scorer = store, scorer
        self.tau, self.slot_theta = tau, slot_theta
        self.relation_tau = dict(relation_tau or {})
        # SS104: a same-answer write merges wherever a query would be routed to the entry
        self.duplicate_theta = tau if duplicate_theta is None else duplicate_theta
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
            if fact["id"] in self.entries and fact["id"] != old:
                # committing would serve the old entry's adapter with this fact's note
                raise ValueError(f"{fact['id']}: id is held by another fact")
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

    def write(self, fact: dict, kappa: float = KAPPA) -> dict:
        """Train the fact's adapter: the first rung of the ladder under the divergence limit.

        The store's one model is parked while a candidate trains, so a write holds the
        lock throughout. If no rung passes, the write commits nothing.
        """
        with self.lock:
            old = self.slot(fact)
            if old is not None and self._same_answer(old, fact):
                return self.commit(fact)
            short = fact_pairs(fact)
            full = short + [(r, fact["answer"]) for r in self._rewrites(fact)]
            store, weight = self.store, self.store.kl_lambda
            tried = {}
            try:
                for name, seed, kl_lambda, rewrites in LADDER:
                    store.kl_lambda = kl_lambda
                    delta = train_adapter(store, full if rewrites else short, seed)
                    kld = store._expert_kld(delta, HELDOUT_CANARIES)
                    tried[name] = kld
                    if kld <= kappa:  # a NaN divergence is rejected, not committed
                        return {**self.commit(fact, delta), "kld": kld, "rung": name}
            finally:
                store.kl_lambda = weight
            return {"action": "rejected", "entry": fact["id"], "tried": tried}

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

    def state_hash(self) -> str:
        """A function of the active set: the model, the router settings and each entry.

        A delta is hashed once when it is committed; hashing every tensor of a large
        store on each commit would make a commit linear in the size of the store.
        """
        h = hashlib.sha256()
        limits = [self.tau, self.slot_theta, self.duplicate_theta]
        limits += [f"{r}={t}" for r, t in sorted(self.relation_tau.items())]
        for s in (self.store.base_hash, self.store.router_config, *map(str, limits)):
            _hash_text(h, s)
        for fid in sorted(self.entries):
            entry = self.entries[fid]
            fields = [fid, entry["subject"], entry["answer"], entry["teach"]]
            fields.append(entry.get("relation", ""))
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
            self._delta_digest = {
                f: _delta_digest(d) for f, d in state["deltas"].items()
            }
            self._router = None
            if self.state_hash() != state["state_hash"]:
                raise ValueError(f"{path}: state hash does not match its contents")
