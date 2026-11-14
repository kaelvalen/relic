# Changes on branch fix/ledger-integrity

These changes make the library do what the paper describes. No GPU job was run and no
recorded result was changed. Old entries stay reproducible: every behavior change has a switch,
and the runners of entries up to 159 set it.

## 1. One system prompt for training, anchor, divergence and serving

Before: training pairs had no system message (Qwen's template then inserts its default one), the
KL anchor and the divergence D used for kappa ran under "Kısa ve net cevap ver.", and serving
used "Answer briefly.".

Now: `DeltaStore.train_system`, `anchor_system` and `system` are class attributes, all
`serving.SYSTEM` ("Answer briefly.") by default. `_pair_step`, `_logits`, `embed`, `answer` and
`answer_routed` read them; `base_answer` and `serve_expert` read the store's `system`.
`hyper_spike.py` reads the anchor and training prompts from the store.

Switch: `use_prompts(store, LEGACY_PROMPTS)`. Every runner in `confirm/` that builds a
`DeltaStore` calls it right after construction (a test checks this). The slice-1 store in
`text/ledger.py` pins its own prompts, which also keeps `answer()` under the Turkish instruction
those records used. `rank2_confirm.store_logits` and `soft_prompt.py` keep their hardcoded
anchor prompt, which is the legacy one.

Changed default: `DeltaStore.answer()` with no `system` now uses `store.system`; it used the
Turkish instruction.

## 2. Adapters stored at the size the paper states

Before: 373 KB was computed (`adapter_size.stored_bytes`); `Ledger.save` wrote fp32 tensors
padded to rank 16 (about 8.7 MB per fact), and notless_eval's parts rank-2 fp32 (about 1.1 MB).

Now (`relic/adapters.py`): `quantize_rows` / `dequantize`, with `_int8_rows` defined as their
composition; `pack_delta` / `unpack_delta` / `packed_nbytes`; `train_adapter_packed`, which
packs the trained adapter directly; `train_adapter` is `pad(unpack_delta(train_adapter_packed))`
and gives the same values as before. `narrow` moved here from notless_eval; `trim_rank` drops
only zero padding. `pack_delta` never loses information: a delta goes into int8 only if an exact
int8 representation of every row exists (`exact_rows`, which also handles rows whose fp16 scale
is subnormal and does not requantize to itself), otherwise it stays fp32.

The ledger keeps `packed[fid]` beside the padded `store.deltas[fid]`. `Ledger.save` writes only
the records, each as one byte buffer with an index; `load` rebuilds the padded adapters. For the
Qwen2.5-1.5B shapes a record is 372,960 bytes; the file grows by about 388 KB per entry, the
rest being the index of module names and the entry's text.

Numerical note: the old `_int8_rows` returned -0.0 where a negative value rounds to zero; the
new one returns 0.0. The values are equal; only the sign bit of zeros differs.

## 3. Replacing a fact can be undone

Before: an update dropped the entry it replaced; revoking the update left the slot empty.

Now: `history[fid]` holds the entries an entry superseded. `revoke(fid)` of an active entry
removes it and brings back the last entry it superseded; `revoke(fid)` of a superseded id erases
that snapshot, delta included. The log records `restored` and, for an erasure, `purged_from`.

Switch: `Ledger(..., keep_history=False)` restores the old behavior. Set in `slot_eval.py`
(E103, E122), `slot_fresh.py` (E115), `ledger_smoke.py` (E121, E144), `ledger_parity.py` (E144)
and `notless_eval.py --system legacy` (E146, the default).

## 4. One write can be removed without the others

Before: a same-answer write added a phrasing but had no record of its own; revoking its id
raised KeyError.

Now: `writes[fid]` lists the writes of an entry, its own first, with `added_phrasing`.
`revoke(entry)` still removes the fact with all its writes (E122). New `retract(write_id)`:
a duplicate is dropped, and its phrasing with it unless another write has the same prompt (the
flag passes to that write); the entry's own write is replaced by the next write, which is written
again (trained when the ledger has a model, committed without an adapter otherwise); the last
write is a revoke. Tests check each case against a ledger in which the write never happened.
Ids are unique across entries, writes and history; a reuse raises ValueError and changes
nothing. Removing a duplicate write's id with `revoke` raises KeyError and names `retract`.

Changed behavior: committing an id that is already used, even for the same fact, now raises.

## 5. State hash v2

`h0` now also covers kappa (`Ledger.kappa`, which `write()` uses; `write()` no longer takes a
kappa argument), `keep_history`, LADDER, the R2 spec, the LoRA+ factor, steps, learning rate, the
three prompts, a digest of the 40 held-out canaries and of the anchor prompts, and the model
revision (`DeltaStore.revision`, read from `model.config._commit_hash`; `base_hash` includes
it). Each entry adds its whole fact, its writes (sorted by id) and its history; delta digests are
computed from the stored record. Files are `"format": 2`. A file without a format is checked with
the old hash (`_legacy_state_hash`, using `DeltaStore.legacy_base_hash`) and then converted, with
an empty history and one write per entry, since format 1 kept neither. A test loads a file
written by the code before this branch (`tests/data/ledger_v1.ledger`).

## 6. The log records what the paper says it records

`commit(fact, delta=None, **extra)` passes `extra` into the log record. `write()` logs the rung,
its D, the D of every rung tried, the seed and the seconds of the whole write. A rejection is
logged as `{"action": "reject", "entry", "tried", "seconds"}` with the unchanged state hash, and
`write()` returns that record (it returned `"rejected"` without logging). notless_eval takes the
seconds from these records instead of timing on its own, and maps `reject` to `rejected` in its
own events so its output schema is unchanged.

## 7. Removal timing includes the router rebuild

`revoke` and `retract` rebuild the router inside the lock and log their `seconds`. notless_eval
reads the removal time from the log.

## 8. Kappa from the adapters the store writes

New `confirm/kappa_calib.py` (draft entry 160): 300 first-rung rank-2 adapters on the
development split (checked disjoint from every earlier draw) give kappa as the p99 of D with a
bootstrap interval; 50 harmful adapters go through every rung of the ladder, and the share
stored is read under the new kappa and under 0.968. `--smoke`. Not run.

## 9. The fresh set through the library write path

`notless_eval.py` takes `--facts {main,fresh}` (fresh is the E137 draw), `--system
{legacy,current}`, `--kappa` and `--prereg`; anything other than the E146 run needs `--prereg`
and writes to its own files. Its summary adds the counts of insert, duplicate, update and reject,
the median and interquartile range of seconds per write, the median seconds per removal, and
the removed facts with a history. Under the current system a fact is removed by its own id. The
wrong-route reading counts and reports the questions it skips. Current-system runs save their
ledger for entry 163. `--report --variant` combines the two datasets of any run.

## 10. Time to first token

New `confirm/ttft_eval.py` (draft entry 163): routing, adapter swap, prefill and first token
against the base model alone, on routed and never-routed queries, with synchronize, warmup and
repeats; median and interquartile range per part. `--smoke`. Not run.

## 11. Pre-registration drafts

Entries 160 to 163 in `docs/LIVING_MODEL_CONFIRMATORY_PREREG.md`, marked as drafts, with
every prediction as TODO(author). None is pinned or tagged.

## Reproducing old entries

- Entries up to 159 in `confirm/`: run the runner as it is; it sets `LEGACY_PROMPTS`, and the
  Ledger runners set `keep_history=False`.
- E146: `notless_eval.py --dataset ... ` with the default `--system legacy`.
- Slice-1 records: `text/ledger.py`'s store keeps their prompts.

## Readings to rerun under the current system

- Kappa of E69: entry 160.
- The fresh-set readings of E137 and E139: entry 161.
- E146, and the unfinished zsRE half of entry 145: entry 162.
- Time to first token, never measured: entry 163.
- Readings that train or serve adapters (wrong route E114, capability E116, aliases E119 and
  E121) describe the old prompts; they need a rerun before the paper reports them for the
  current system. Routing-only readings (E76, E109, E139's routing, E156, E159) do not use the
  language model's prompts and are unaffected.
