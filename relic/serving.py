"""Serving: answer a query with the base model, one entry's adapter, or its note."""

from __future__ import annotations

import ctypes
import math
from numbers import Integral, Real

SYSTEM = "Answer briefly."
_LIBC = ctypes.CDLL("libc.so.6")


def _system(store) -> str:
    """The store's serving prompt (DeltaStore.system); SYSTEM for stores without one."""
    return getattr(store, "system", SYSTEM)


def trim() -> None:
    """Return glibc arena memory to the OS (the training loop's RSS balloon)."""
    _LIBC.malloc_trim(0)


def base_answer(store, q, note=None, system=None):
    full = store.deltas
    store.deltas = {}
    store.materialize()
    if system is None:
        system = _system(store)
        system = system if note is None else f"{system} Context: {note}"
    resp = store.answer(q, system=system)
    store.deltas = full
    store.materialize()
    return resp


def serve_expert(store, fid, q, system=None):
    full = store.deltas
    store.deltas = {fid: full[fid]}
    store.materialize()
    resp = store.answer(q, system=system or _system(store))
    store.deltas = full
    store.materialize()
    return resp


def _route_index(router, q, enc, candidate_count):
    """Validate Router.route's (candidate index or None, similarity) contract."""
    result = router.route(q, enc)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("router.route must return (index_or_None, similarity)")
    j, similarity = result
    if isinstance(similarity, bool) or not isinstance(similarity, Real):
        raise TypeError("router.route similarity must be a real number")
    similarity = float(similarity)
    if not math.isfinite(similarity):
        raise ValueError("router.route similarity must be finite")
    if j is None:
        return None, similarity
    if isinstance(j, bool) or not isinstance(j, Integral):
        raise TypeError("router.route index must be an integer or None")
    j = int(j)
    if j < 0 or j >= candidate_count:
        raise IndexError(
            f"router.route index {j} outside candidate range [0, {candidate_count})"
        )
    return j, similarity


def routed_answer(store, router, candidates, q, enc):
    """Serve the selected candidate expert, or the base model on router abstention."""
    j, similarity = _route_index(router, q, enc, len(candidates))
    resp = (
        serve_expert(store, candidates[j]["id"], q)
        if j is not None
        else base_answer(store, q)
    )
    return j, similarity, resp


def rag_system(mode, facts, j):
    f = facts[j]
    if mode == "plain":
        return f"{SYSTEM} Context: {f['teach']}"
    if mode == "instruct":
        return (
            f"{SYSTEM} Use the context even if it contradicts what you know. "
            f"Context: {f['teach']}"
        )
    if mode == "qa":
        return f"{SYSTEM} Context:\nQ: {f['probe']}\nA: {f['answer']}"
    demo = facts[(j + 1) % len(facts)]
    return (
        f"{SYSTEM} Follow the context over your prior knowledge.\n"
        f"Example:\nQ: {demo['probe']}\nA: {demo['answer']}\n"
        f"Context: {f['teach']}"
    )


class Server:
    """Serving with a base-answer cache; `system=None` means the plain system prompt."""

    def __init__(self, store):
        self.store = store
        self._base = {}

    def base(self, q, system=None):
        key = (q, system)
        if key not in self._base:
            self._base[key] = base_answer(self.store, q, system=system)
        return self._base[key]

    def adapter(self, fid, q, system=None):
        store = self.store
        full = store.deltas
        store.deltas = {fid: full[fid]}
        store.materialize()
        resp = store.answer(q, system=system or _system(store))
        store.deltas = full
        store.materialize()
        return resp


def serve_arms(server, router, facts, q, enc):
    """(route index, similarity, {arm: response}) for one query."""
    j, sim = router.route(q, enc)
    if j is None:
        base = server.base(q)
        return j, sim, {"adapter": base, "retrieval": base, "both": base}
    note = rag_system("instruct", facts, j)
    fid = facts[j]["id"]
    return (
        j,
        sim,
        {
            "adapter": server.adapter(fid, q),
            "retrieval": server.base(q, system=note),
            "both": server.adapter(fid, q, system=note),
        },
    )
