# A hypernetwork that writes the adapter: design

Status: draft for the author, 2026-11-12. Nothing here is pinned; every run is pinned as a
prereg entry before it starts, with the margins this document leaves open.

## Intent

Today a write trains one rank-2 adapter per fact (21 s at the median, 373 KB stored). The
author's final system has five goals: adapters that do more than a weighted notebook, a
router that alone keeps locality, a system that keeps working when the base model changes,
strong scores on standard benchmarks, and low cost, storage above all. A hypernetwork, a
network that reads a fact and outputs its adapter in one forward pass, touches four of
them at once: storage becomes the fact's text, a write takes milliseconds, a new base model
needs one hypernetwork and not one training per fact, and an adapter learned across many
facts may generalise past the phrasings it was given.

This line of work does not have to win at once. Its first purpose is a comparison that
tells the author whether the hypernetwork's weakness against trained adapters is one that
can be optimised: a gap that shrinks with training data or a few steps of fine-tuning. If
it is, the project is configured around it; if the gap does not move, the line stops.

## Goals

1. Measure, on the same facts and through the same router, a generated adapter against the
   current trained adapter, on every reading of the main evaluation (entry 145).
2. Measure whether the gap is optimisable: how it changes with the number of training facts,
   and with a few steps of fine-tuning that start from the generated adapter.
3. Measure the cost side: bytes stored per fact, seconds per write, and whether a stored
   fact can be rebuilt bit for bit from its text.
4. Measure generalisation to relations and templates the hypernetwork never saw in training.
5. Later phases: a second base model, and readings beyond the notebook (reverse questions,
   two-hop questions).

## Non-goals

- Router changes. Evaluation goes through the default router with `RELATION_TAU`, as in
  entry 145; the router is a separate track.
- Paper, README and deck text. The paper is written last, for the final system.
- Multi-fact adapters (shards). The hypernetwork writes one adapter per fact.

## Anti-goals: a failure even if the numbers look good

- **The hypernetwork learns from the store.** If stored facts update the hypernetwork, a
  revoked fact keeps its influence and exact removal is lost. The hypernetwork is trained
  offline on facts disjoint from every store and frozen while serving.
- **Template leakage.** Entry 79's lesson: CounterFact relations share templates, so a split
  by fact is not a held-out test. Evaluation facts are disjoint by subject, and one reading
  holds out whole relations and their templates.
- **Hidden per-fact storage.** An arm that silently stores a trained delta per fact does not
  count as low storage. Every arm reports its bytes per fact.
- **Hidden per-model tuning.** Constants tuned by hand for Qwen2.5-1.5B do not count towards
  "universal". What changes per model is written down.

## The reference

The current default (entries 136 to 146), trained on fresh facts with the ladder: CounterFact
edit .995, paraphrase .844 (no note), E3 leak .038, wrong route kept .558, forced MMLU and ARC
cost -.017, 18 percent of facts need a retry, 3 of 1000 not stored, 373 KB and 21 s per fact.

## What is built

**Output.** The same adapter the store already serves: rank 2 on `q_proj` and `v_proj` in all
28 layers of Qwen2.5-1.5B-Instruct, about 272 thousand parameters, padded into the store's
layout as `train_adapter` does. Serving, routing, the divergence gate and removal stay as
they are; only the source of the delta changes.

**Input.** The fact as the write path sees it: subject, edit prompt, answer.

**Training.** Offline, on CounterFact and zsRE facts disjoint from every evaluation set.
The loss is the per-fact objective of today's writes, amortised: the answer under the edit
prompt and its rewrites, plus the KL anchor on the canaries, with the base model frozen and
the gradient reaching only the hypernetwork.

**Arms on the evaluation facts.**

| Arm | What it is | Stored per fact |
|---|---|---|
| T | today's trained adapter with the ladder | 373 KB |
| H0 | the generated adapter, no further training | text |
| Hk | the generated adapter plus k steps of today's training (k pinned, e.g. 2 and 4) | text if the steps rebuild bit for bit, else 373 KB |
| Rk | a random start plus the same k steps: the control for Hk | as Hk |

Generated adapters pass through the same divergence gate. A rejected H0 adapter goes on to
H2, then H4; if those are rejected too the fact is not stored. There is no fall back to T, so
each arm stays one recipe and its rejection count is a reading.

**Scaling.** H0 and Hk are trained at three sizes of the training pool (for example 2, 8
and about 30 thousand facts) so that the gap to T can be read as a curve, not a point.

**Held-out relations.** One training run leaves out a set of relations with all their
templates; H0 is read on facts of those relations beside facts of seen ones.

## Evaluation facts

800 fresh facts, 400 CounterFact and 400 zsRE, drawn from facts no decision has seen and
disjoint by subject from the training pool, as in entry 136, whose N was chosen so that a
true gap of zero can pass a margin of -.03. T is trained on them with today's ladder.

## Readings

Full readings, for T and for each arm at the largest pool: every reading of entry 145, on
the evaluation facts and through the router: edit prompts, paraphrases, E3 and E2 leak,
natural near and uniform leak, removal and retention, forced wrong route, forced MMLU and
ARC, facts rejected by the gate. Added: seconds per write, bytes per fact, the hypernetwork's
own size, bitwise rebuild of Hk from text in a new process.

Short readings, for the smaller pools and the held-out relations: edit prompts, paraphrases
and E3 leak. They are enough to draw the scaling curve, and they keep the GPU time to a
fraction of running the full set eight times.

## Decision, by comparison

Three outcomes, each with margins pinned before the run:

- **Configure the project around it**: Hk with small k reaches T within the margins on edit
  and paraphrase and is not worse on leak and wrong route, with the cost of text storage;
  or H0 does so outright.
- **Optimisable**: H0 trails T, but the gap falls with each step of training data and Hk
  beats Rk. The next entry is a larger pool or a larger hypernetwork, not a new idea.
- **Stop**: the gap is flat across pool sizes and Hk is no better than Rk. The hypernetwork
  learned nothing a random start does not give.

## Phases

0. **Feasibility**: a few hundred training facts; does a generated adapter make the edit
   prompt answered on its own training facts at all, and on a few unseen ones? Decides
   whether phase 1 is worth its GPU time. Pinned and reported like any run (author,
   2026-11-12).
1. **Main comparison** on Qwen2.5-1.5B: all arms, three pool sizes, held-out relations.
   Pinned.
2. **Second base model**: the same recipe, with what had to change written down. Pinned.
3. **Beyond the notebook**: reverse questions and two-hop questions (benchmarks such as
   MQuAKE), for the best arm against T. Pinned.

## Constraints

- One 8 GB GPU, shared: check before each run, never stop another process, pause when the
  author asks. Training must fit beside nothing else; serving must fit as today.
- Prereg first; predictions recorded; failed predictions are not patched.
- No Claude traces in git. Many small commits.
- `Ledger` keeps its interface; the hypernetwork is another way to make the delta in
  `Ledger.write`.

## Decided by the author

- The path is explored through a comparison with today's trained adapter, and the decision
  follows the comparison (2026-11-12).
- A weakness that can be optimised is enough to continue; the first result need not win.
- The paper waits for the final system.
- Phase 0 is pinned and reported too, not a throwaway spike (2026-11-12).
- The GPU budget of phase 1 (a guess of 1.5 to 3 days) is decided after phase 0.

## My calls, open to the author

- The output keeps today's adapter shape (rank 2, `q_proj` and `v_proj`), so that only the
  source of the delta changes and T is a fair reference.
- The training loss is today's per-fact objective, amortised, rather than a regression onto
  saved trained adapters: two adapters with the same behaviour can have very different
  weights.
- Hk is in scope as the bridge arm, because it is the likely middle ground between storage
  and quality.
- Training facts get the same base-model rewrites as a write does today, generated in
  batches, not the dataset's paraphrases: the paraphrases of the evaluation come from the
  datasets, and training on their style would favour the hypernetwork over T.
- A rejected generated adapter climbs H2 and H4 and is otherwise not stored; no fall back
  to T.
- Full readings only at the largest pool; short readings elsewhere.

## Left to the builder

- The hypernetwork's architecture (for example a fact embedding plus layer and module
  embeddings, decoded by a shared network into each A and B), its size, and the fact
  encoder (the frozen base model's own hidden states or a separate encoder).
- Batch size, learning rate, steps, memory techniques for the 8 GB GPU.
- Which relations are held out, as long as their templates go with them.
