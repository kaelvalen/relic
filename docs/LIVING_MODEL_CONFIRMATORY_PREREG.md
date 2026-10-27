# Confirmatory study: do the guarantees and the router survive scale? (pre-registration)

Status: **proposed 2026-10-03, to be committed before any confirmatory run.** The
slice-1 record (`docs/LIVING_MODEL_SLICE_PREREG.md`) is an exploratory sequence, each
entry honest alone but written after seeing the previous result; this document is the
single confirmatory run it points to. Position and audit:
`docs/LIVING_MODEL_POSITIONING.md`.

## 1. Question

1. At N in {50, 200, 1000} facts, does the parameter-free router keep precision,
   recall and abstention, and do the measured guarantees (behavioural return after
   revoke, locality, provenance) hold, with confidence intervals?
2. Is there a task family where per-fact deltas beat retrieval at equal cost? The
   candidate measured first is VLM image-conditioned recognition (the contrast-pair
   result); then behaviour/format facts; then retrieval-missed paraphrases.

## 2. Facts

- **Nonce facts**: procedurally generated (subject/relation/object over a small
  grammar) in English, with held-out paraphrases (templated, unseen wordings) and
  distractor questions; N facts per size, fresh seeds. The generator is pinned by
  commit hash before the run.
- **CounterFact / zsRE subsets**: intended sources are the ROME dumps
  (`counterfact.json`, `zsre_mend_eval.json`); a dry-download check pins the exact
  URLs, item filters (single-token targets, balanced relation mix) and the held-out
  paraphrase policy (the dataset's own alternate prompts where present, else a
  template) at execution time, before any arm runs.
- Every fact carries: one teach pair, one probe, one held-out paraphrase, one
  distractor.

## 3. Arms

| arm | what | implementation |
| :-- | :-- | :-- |
| memory-only (RAG) | MiniLM retrieval + note injection; no weights | this repo |
| shared LoRA sequential | one adapter trained on all facts in order | this repo |
| summed deltas | per-fact deltas added into one adapter (slice-1: 0/18) | this repo |
| GRACE | key-value adaptors + deferral radius | reimplementation from the paper (official code if it runs on Qwen2.5; provenance recorded either way) |
| MELO | neuron-indexed dynamic LoRA blocks activated by a vector index | same |
| WISE | side memory + router | same |
| ours | per-fact deltas + MiniLM router + ledger + repair controller | this repo |

## 4. Metrics (all with bootstrap 95% CIs over facts)

Efficacy (probe hit), paraphrase generalisation (held-out), locality (neighbours'
recall after an edit), router precision/recall, abstention on distractors, behavioural
return after revoke (canary KL vs the pre-add state, retain-set accuracy), per-fact
time (train / serve / revoke) and storage.

## 5. Primary claims and thresholds (pinned before the run)

- **Router at N = 1000**: precision >= 0.95 and abstention >= 0.95; the curve
  50 -> 200 -> 1000 is reported (Stage-1's F18/F19 warning: routers degrade with
  scale, so this is the primary risk).
- **Behavioural return**: after revoke, canary KL <= 0.05 nats and retain accuracy
  within 1 point of the pre-add value.
- **Provenance**: leave-one-out attribution >= 0.95.
- **Delta > memory**: at least one family with a CI-separated gap; the VLM
  image-conditioned family is measured first.

## 6. Falsification

If router precision drops below 0.90 at N = 1000, or behavioural return fails, or no
delta > memory family separates, the thesis narrows to the measured parts and the
failure itself is the result. No mid-run tuning: a bug fix after a run restarts that
arm at all sizes.

## 7. Order of execution

1. Nonce generator + held-out paraphrases + distractors (commit hash pinned).
2. Dry-download and filter the external subsets; pin the exact selection.
3. Pilot N = 50, all arms; fix bugs; freeze the harness.
4. N = 200, then N = 1000; one run per arm per size.
5. VLM image-conditioned family: extend the contrast result to N = 50 scenes with
   held-out variants; the fair memory baseline is CLIP retrieval + note injection.

## 8. What this study is not

Not a new architecture claim (the per-fact LoRA + retrieval-router family is prior
art: MELO, GRACE, WISE, LoraRetriever, T-Patcher); not a full-transformer exactness
claim; not a claim that memory cannot revoke (it can, trivially). The claim under
test is the transaction ledger, provenance and the measured guarantee set - and the
delta's existence proof against retrieval.

## 9. Execution pin (2026-10-03)

- **Nonce generator**: `experiments/sandbox/live_learning/confirm/facts.py`; seed policy
  seed = n (50 / 200 / 1000 are independent draws); outputs regenerable at
  `results/live_learning/confirm/facts_n{50,200,1000}.json` (untracked). Validation on
  generation: unique subjects, unique word answers, paraphrase != probe. The generator
  commit is the pin.
- **External subsets**: dry check passed with an explicit certifi CA bundle (the bare
  urllib path fails on this host's CA setup): `counterfact.json` 45,108,470 bytes and
  `zsre_mend_eval.json` 8,091,864 bytes at `https://rome.baulab.info/data/dsets/`.
  Adapters and filters are committed before the pilot: CounterFact -> teach = prompt +
  target_new, probe = prompt, held-out paraphrase = first paraphrase_prompt, distractor
  = another item's probe; zsRE -> probe = src, held-out paraphrase = alt, answer =
  answers[0]; filter: single-token target with the Qwen2.5 tokenizer, balanced across
  the top relations/subjects; the sampling seed lives in the adapter. Downloads use
  certifi's CA bundle explicitly.

## 10. Staged pilot (2026-10-03, before the pilot)

The harness is built in two stages so bugs are fixed cheaply: **pilot A** (this run)
covers the in-house arms on the nonce set - rag, sequential (shared LoRA), summed,
ours - with the full metric set and CIs at N = 50. **Pilot B** adds the external
adapters (CounterFact/zsRE) and the GRACE/MELO/WISE arms before N = 200; the freeze
happens after pilot B. Harness pins for pilot A: answer system "Answer briefly.";
router tau = 0.656 (the slice-1 MiniLM value); anchors = two English canaries; ours =
propose_and_commit (cap 2.0) with 16 steps x 2 pairs per fact; sequential = one shared
LoRA, 8 steps x 2 pairs per fact (stated: half the per-fact budget); summed = the
per-fact deltas materialised together; revoke sample = the first 10 facts (token gone
+ retain on the rest); bootstrap 95% CIs over facts.

## 11. Pilot A results (2026-10-03, N = 50 nonce, wall 676 s)

| arm | efficacy | paraphrase | distractor no-leak | route |
| :-- | :-- | :-- | :-- | :-- |
| rag (memory-only) | 0.80 [0.68, 0.92] | 0.82 [0.70, 0.92] | 1.00 | 1.00 |
| ours | 0.82 [0.70, 0.92] | 0.76 [0.64, 0.88] | 0.60 | 1.00 |
| summed | 0.00 [0.00, 0.00] | 0.00 | 1.00 | - |
| sequential (shared LoRA) | 0.08 [0.02, 0.16] | 0.08 [0.02, 0.16] | 0.40 | - |
| revoke (10 of 50) | gone 1.00 [1.00, 1.00] | retain 0.875 [0.775, 0.975] | | |

Ours extras: add 7.19 s/fact, serve 0.36 s, storage 435.8 MB for 50 facts (~8.7 MB/
fact), paraphrase route accuracy 1.00, **router abstention on distractors 0.00**.

Readings, honest:
1. **Memory-only matches the delta system on this fact family** (CIs overlap; RAG is
   ahead on paraphrase). The reviewer's first question stands: on nonce facts the
   vector DB is the baseline to beat, and pilot A does not beat it. The delta's
   existence proof must come from another family (VLM image-conditioned first).
2. Composition replicates: summed 0/50; sequential 0.08 - the per-fact delta + routing
   architecture is the only working delta design here, at 7.2 s/fact and 8.7 MB/fact
   (~2 h and ~8.7 GB at N = 1000 - the scale cost is real and reported).
3. **The router abstention fails on template-identical distractors** (0.00): the
   distractor asks about an unseen subject with the same question template, so the
   semantic key matches; the router needs subject-aware keys (entity in the key or a
   verify step), not template similarity alone. The predicted router-at-scale failure
   is visible already at N = 50.
4. Revoke is behaviourally clean on this set (token gone 1.00; retain within efficacy).

Pilot B (external adapters + GRACE/MELO/WISE) and the subject-aware router fix come
before N = 200; the freeze happens after pilot B.

## 12. Router v2 (entity-aware) + pilot A2 (pinned 2026-10-03, before the run)

Pilot A's abstention failure (0.00) comes from template similarity: the distractor
asks about an unseen subject with the same question template. Router v2 requires a
**known subject** in the question (longest substring match over the fact base's subject
strings; no match -> abstain), then the semantic key picks among that subject's facts
(tau 0.656 unchanged). The same router serves every arm (RAG and ours) so the
comparison stays fair. Pilot A2 reruns N = 50 with the fixed router; predictions:
distractor abstention ~1.00 for RAG and ours, no-leak ~1.00, efficacy/paraphrase
unchanged within CI.

## 13. Pilot A2 blocked by the shared GPU; external adapters built (2026-10-03)

The rerun with router v2 hit OOM: a Minecraft Java process holds ~1.7 GB of the 8 GB
card, so the store model plus the sequential arm's second base do not fit. Per the
house rule the process is not touched; pilot A2 reruns when the GPU frees (no result
changed). Meanwhile the external adapters were built and sampled: counterfact 21,919
adapted (target <= 1 token: 10,034; 50 sampled, seed 50); zsRE 19,086 adapted (<= 1
token: 2,287; 50 sampled). The zsRE adapter uses `rephrase` as the held-out paraphrase
(`alt` is an alternative answer, not a question) and both adapters require the subject
in the paraphrase.

## 14. Router v2 results (CPU-only, N = 50, 2026-10-03)

| router | probe route | paraphrase route | distractor abstention |
| :-- | :-- | :-- | :-- |
| v1 semantic (pilot A) | 1.00 | 1.00 | **0.00** |
| v2 entity-aware | 1.00 | 1.00 | **1.00** |

The predicted fix holds with no routing loss: the entity gate turns template-identical
distractor abstention from 0.00 to 1.00 while probe and paraphrase routing stay 1.00.
The serving half of pilot A2 (behavioural abstention through the model, no-leak) still
waits for free VRAM; this measurement is CPU-only (MiniLM) and needs no model.

## 15. Pilot A2 results (router v2, N = 50 nonce, wall 829 s)

| arm | efficacy | paraphrase | distractor no-leak | route |
| :-- | :-- | :-- | :-- | :-- |
| rag | 0.80 [0.68, 0.92] | 0.82 [0.70, 0.92] | 1.00 | 1.00 |
| ours | 0.82 [0.70, 0.92] | 0.76 [0.64, 0.88] | **1.00** | 1.00 |
| summed | 0.00 [0.00, 0.00] | 0.00 | 1.00 | - |
| sequential | 0.08 [0.02, 0.16] | 0.08 | 0.40 | - |
| revoke (10/50) | gone 1.00 [1.00, 1.00] | retain 0.875 [0.775, 0.975] | | |

Ours extras: add 9.27 s/fact, serve 0.43 s, storage 435.8 MB, distractor abstention
1.00 (was 0.00), paraphrase route 1.00. The fix holds through the model: no-leak 0.60 ->
1.00 and abstention 0.00 -> 1.00 with efficacy/paraphrase unchanged within CI. The
pilot-A finding stands: on nonce facts memory-only matches the delta system; the
delta's existence proof must come from another family (VLM image-conditioned first).
Next: external sets (CounterFact/zsRE) and the GRACE/MELO/WISE-style arms before the
freeze.

## 16. External results (pilot B, N = 50 per set, router v2, 2026-10-03)

CounterFact (wall 691 s): rag eff 0.76 [0.64, 0.88], para 0.34; **ours eff 1.00
[1.00, 1.00]**, para 0.34; summed 0.00; sequential 0.34; revoke gone 0.90, retain
1.00; ours add 7.0 s, abstention 1.00, para_route 0.42.
zsRE (wall 859 s): rag eff 0.82 [0.72, 0.92], para 0.84 [0.74, 0.94]; **ours eff 1.00
[1.00, 1.00], para 0.98 [0.94, 1.00]**; summed 0.02; sequential 0.80; revoke gone 0.10,
retain 1.00; ours add 9.3 s, abstention 1.00, para_route 0.98.

Readings:
1. **The delta > memory existence proof appears here**: on both external sets ours
   reaches 1.00 efficacy with CI separation from RAG (0.76 / 0.82), and on zsRE also
   paraphrase 0.98 vs 0.84. The mechanism is visible: these are counterfactual or
   parametric facts - context injection (RAG) loses to the model's prior, a weight
   edit overrides it. This is the measured answer to "why not a vector DB".
2. Two measurement artifacts, pinned for the fix: (a) zsRE no-leak 0.04 for *both*
   arms - the distractor asks a planet question and the base answers a planet that
   coincides with a taught answer; the leak metric must exclude answer-vocabulary
   collisions (or use off-template distractors); (b) zsRE revoke gone 0.10 - the base
   already answers these true facts parametrically, so token absence is the wrong
   return metric; the confirmatory metric is behavioural return to the pre-add state
   (response/KL), not token removal (CounterFact, where the base does not know the
   counterfactual, shows gone 0.90 - the contrast confirms the explanation).
3. CounterFact paraphrase 0.34 and para_route 0.42: the dump's paraphrase prompts are
   noisy; the held-out paraphrase policy needs a quality filter (or zsRE-style
   rephrase). Recorded, not fixed.
4. Composition and sequential fail on external sets too (summed 0.00 / 0.02;
   sequential 0.34 / 0.80 with no_leak 0.06 / 0.26).

## 17. Scale step 1: CounterFact N = 200 (pinned 2026-10-03, before the run)

The existence proof scales first: same harness, router v2, cap 2.0; arms rag + ours
only (summed/sequential return in stage 2); CounterFact sampled at N = 200 with the
same adapter (seed 200, max_tokens 1). Prediction: ours efficacy stays at or near 1.00
with CI separation from RAG; add cost ~7 s/fact (~25 min); abstention 1.00; revoke gone
high (the base does not know the counterfactuals).

## 18. Capacity guard (pinned 2026-10-03, before the N = 200 rerun)

The N = 200 run hit the merged-adapter capacity (CAP = 64): `materialize` writes all
active deltas into one adapter, which cannot hold more than 64. Routed serving only
needs one delta at a time, so the pilot harness now uses a `PilotStore` whose
`materialize` materialises sets <= CAP as before and zeroes above CAP. Consequence,
recorded as a design limit: the **summed arm is CAP-bound by construction** (it needs
all deltas merged) and cannot run at N > 64; the routed arm is unaffected. Stage-1
N = 200 keeps arms rag + ours.

## 19. Scale step 1 results: CounterFact N = 200 (2026-10-03, wall 2242 s)

| arm | efficacy | paraphrase | no-leak | route |
| :-- | :-- | :-- | :-- | :-- |
| rag | 0.85 [0.80, 0.90] | 0.31 [0.25, 0.38] | 0.915 | 1.00 |
| ours | **0.995 [0.985, 1.00]** | 0.295 [0.235, 0.36] | 0.915 | 1.00 |
| revoke (10/200) | gone 0.90 [0.70, 1.00] | retain 0.995 [0.984, 1.00] | | |

Ours extras: add 9.2 s/fact, serve 0.07 s, storage 1743 MB (~8.7 MB/fact, ~8.7 GB at
N = 1000), abstention 1.00, paraphrase route 0.355. The existence proof scales: the
efficacy gap survives (0.995 vs 0.85, CIs separated); abstention holds at 1.00; revoke
stays behavioural (gone 0.90, retain 0.995). CounterFact paraphrase remains noisy for
both arms (~0.3). Storage and training time are the scale costs to report.

## 20. Scale step 2: zsRE N = 200 (pinned 2026-10-03, before the run)

Same harness, router v2, cap 2.0; arms rag + ours; zsRE sampled at N = 200 (seed 200,
max_tokens 1). Prediction: ours efficacy near 1.00 with CI separation from RAG (the
N=50 gap was 1.00 vs 0.82); abstention 1.00; the no-leak artifact (answer-vocabulary
collision) and the revoke-gone confound (parametric knowledge) are expected to persist
and stay recorded as metric fixes, not arm failures.

## 21. Scale step 2 results: zsRE N = 200 (2026-10-03, wall 1945 s)

| arm | efficacy | paraphrase | no-leak | route |
| :-- | :-- | :-- | :-- | :-- |
| rag | 0.79 [0.73, 0.845] | 0.825 [0.77, 0.88] | 0.03 | 0.955 |
| ours | **0.99 [0.975, 1.00]** | **0.98 [0.96, 0.995]** | 0.03 | 0.955 |
| revoke (10/200) | gone 0.50 [0.20, 0.80] | retain 0.9895 [0.974, 1.00] | | |

Ours extras: add 7.8 s/fact, serve 0.06 s, storage 1743 MB, abstention 1.00, paraphrase
route 0.93. The gap scales on both external sets (efficacy and paraphrase CI-separated
from RAG); abstention holds; the two known metric artifacts persist exactly as pinned
(no-leak 0.03 = answer-vocabulary collision; revoke gone 0.50 = parametric knowledge
confound, CounterFact 0.90 for contrast). Remaining before the freeze: the GRACE/MELO/
WISE-style arms, the two metric fixes (collision-excluded leak; behavioural return),
then N = 1000.

## 22. Behavioural-return metric (pinned 2026-10-03, before the verification run)

The revoke metric becomes behavioural: for the first 10 facts the pre-add base response
is recorded before any training; after revoke the served response must have the same
token status as pre-add (return_match). Token absence (gone) stays reported but is only
meaningful where the base does not already know the fact. Verification run: zsRE N = 50,
arms rag + ours.

## 23. Behavioural-return verification (zsRE N = 50, wall 588 s, 2026-10-03)

The pre-add/post-revoke token-status match is **0.9 [0.7, 1.0]** (return_match), replacing token
absence as the return metric: token gone stays 0.10 (the parametric confound), while
the behaviour returns to the pre-add state. Efficacy/paraphrase/no-leak unchanged
(ours 1.00 / 0.98 / 0.04; rag 0.82 / 0.84 / 0.04). The confirmatory return metric is
now behavioural; the no-leak collision fix (base-referenced leak) remains pinned for
the freeze pass.

## 24. Base-referenced no-leak (pinned 2026-10-03, before the verification run)

The leak metric becomes base-referenced: the base response for every distractor is
recorded before any training, and a response counts as a leak only when it contains a
taught word the pre-edit base did not produce (the zsRE collision artifact: planet
answers coincide with taught answers). Verification: zsRE N = 50, arms rag + ours.

## 25. Base-referenced no-leak verified (zsRE N = 50, wall 517 s, 2026-10-03)

**no_leak 0.04 -> 1.00 for both arms** with efficacy (ours 1.00 / rag 0.82) and
paraphrase (0.98 / 0.84) unchanged: the collision was a metric artifact, not a leak.
Both pinned metric fixes are now in the harness (behavioural return; base-referenced
leak). Remaining before the freeze: the GRACE/MELO/WISE-style arms, then N = 1000.

## 26. MELO-style arm (pinned 2026-10-03, before the verification run)

MELO (AAAI 2024) is per-edit dynamic LoRA blocks activated by a vector index - the same
architecture as ours minus the transaction protocol. The arm `melo_like` trains the
identical per-fact deltas with the same anchor and router but commits via `add` without
the KLD cap and without the ledger checks. Prediction: efficacy equals ours within CI
(the cap is a guarantee, not an efficacy mechanism); the comparison is for the
guarantee table, not accuracy. Verification: zsRE N = 50, arm melo_like.

## 27. MELO-style arm verified (zsRE N = 50, wall 412 s, 2026-10-03)

melo_like: efficacy 1.00 [1.00, 1.00], paraphrase 0.98 [0.94, 1.00], no-leak 1.00,
route 0.98 - identical to ours within CI, as predicted. The KLD cap is a guarantee,
not an efficacy mechanism; the MELO comparison belongs to the guarantee table, not the
accuracy table. Remaining before the freeze: GRACE/WISE-style arms (logit-boost
codebook with deferral radius; side-memory router), then N = 1000.

## 28. GRACE- and WISE-style arms (pinned 2026-10-04, before the verification run)

`grace_like`: codebook (MiniLM key -> single-token answer id) + deferral radius = the
same entity router; when routed, the value token is forced in logit space (bias +10 on
the answer id during generation), else the base answers. No weights change.
`wise_like`: side memory only for facts the base does not already answer (conflicts);
the same entity router selects the side entry; non-conflict facts stay with the base
model. Both arms share the harness router (fairness) and the base-referenced leak
metric. Verification: zsRE N = 50, arms grace_like + wise_like. Predictions:
grace_like efficacy high but paraphrase weaker (logit forcing does not teach); wise_like
efficacy equals the base on non-conflicts and gains on conflicts - the side-memory
ceiling.

## 29. GRACE- and WISE-style results (zsRE N = 50, wall 137 s, 2026-10-04)

| arm | efficacy | paraphrase | no-leak | route |
| :-- | :-- | :-- | :-- | :-- |
| grace_like | 0.70 [0.56, 0.82] | 0.72 [0.60, 0.84] | 1.00 | 0.98 |
| wise_like | 0.82 [0.72, 0.92] | 0.74 [0.62, 0.84] | 1.00 | 1.00 |

Both below ours (1.00 / 0.98) and at or below RAG (0.82 / 0.84): logit-space forcing
does not teach (grace), and the side-memory ceiling equals RAG on zsRE (the base
already knows most facts; the conflicts are exactly where a weight edit wins). Every
arm in the prereg's table now has a measured reading. Next: N = 1000.

## 30. N = 1000 (pinned 2026-10-04, before the run)

The primary scale claim: CounterFact N = 1000 (seed 1000, max_tokens 1), arms rag +
ours, same harness (router v2, cap 2.0, behavioural return, base-referenced leak).
Predictions: ours efficacy >= 0.98 with CI separation from RAG; abstention >= 0.99;
return_match >= 0.8; add cost ~7-9 s/fact (~2.5 h); storage ~8.7 GB. The zsRE N = 1000
run follows the same pin (same arms) if the session allows.

## 31. N = 1000 results: CounterFact (2026-10-04, chunked; eval wall 1815 s)

| arm | efficacy | paraphrase | no-leak | route |
| :-- | :-- | :-- | :-- | :-- |
| rag | 0.834 [0.812, 0.858] | 0.326 [0.298, 0.356] | 1.00 | 0.997 |
| ours | **0.999 [0.997, 1.000]** | 0.323 [0.295, 0.352] | 1.00 | 0.997 |
| revoke (10/1000) | gone 1.00 [1.00, 1.00] | retain 0.999 [0.997, 1.00] | | |

Ours extras: add 8.45 s/fact, serve 0.04 s, storage 8716 MB, abstention 1.00,
paraphrase route 0.362. **The primary scale claim holds**: the efficacy gap is
CI-separated at N = 1000 (0.999 vs 0.834), abstention stays 1.00, the router route
0.997, revoke behavioural (gone 1.00, retain 0.999). Paraphrase stays noisy for both
arms (CounterFact dump). Execution note: the run was chunked (host-RSS glibc arena
retention; malloc_trim per fact + idempotent ranges) - the protocol is unchanged; the
chunking is an execution detail.

## 32. Review round (2026-10-04): accepted limits and the gating order

The review received after `77c5f70` is accepted in full and recorded in
`docs/LIVING_MODEL_REVIEW_2026-10-04.md` (six findings, point by point) and in
Positioning Appendix B.4. Consequences for this prereg: the N=1000 table stands only as
"delta > plain-prompt RAG" until the strong-RAG arms report; the router metrics are
dictionary-like and the paraphrase score is a router failure; the GRACE/WISE/MELO
columns are approximations, not official implementations; the §5 thresholds (canary
KL, retain, provenance, router precision) and a revoke sample of n >= 100 are owed;
the cost-scaling question becomes the multi-fact delta capacity curve; the paper
identity (editing vs systems/guarantee) is an open choice. **No runs were started for
this round; each item pins before its run, and item 1 is the gate: if the gap closes,
stop and narrow the thesis.**

## 33. Strong-RAG arms (pinned 2026-10-04, before the run; review item 1 - the gate)

CounterFact N=200 first (confirmed at N=1000 only if the gap survives). Arms:
`rag_instruct` ("use the context even if it contradicts what you know"), `rag_qa`
(context as a Q/A pair), `rag_fewshot` (a note-following demonstration from another
fact). Same entity router, same facts, same base-referenced metrics. The plain arm's
N=200 numbers are the reference (ours 0.995 vs rag 0.85). Predictions per the review:
the gap narrows; the gate is whether it stays CI-separated from ours. If it closes,
stop and narrow the thesis to the guaranteed expert bank.

## 34. Strong-RAG results (CounterFact N = 200, wall 490 s, 2026-10-04)

| arm | efficacy | paraphrase |
| :-- | :-- | :-- |
| rag_instruct | 0.865 [0.815, 0.91] | 0.315 |
| rag_qa | 0.71 [0.645, 0.77] | 0.275 |
| rag_fewshot | 0.52 [0.45, 0.59] | 0.20 |
| (plain rag, N=200) | 0.85 [0.80, 0.90] | 0.31 |
| (ours, N=200) | 0.995 [0.985, 1.0] | 0.295 |

**The gate passes**: the strongest strong-RAG variant (explicit instruction) reaches
0.865 and stays CI-separated from ours (0.995); Q/A formatting and few-shot hurt
(0.71 / 0.52). The headline stands as "delta > strong RAG" at N=200; the N=1000
confirmation for rag_instruct is launched.

## 35. Router stress results (review item 2; N=50 nonce, CPU, 2026-10-06)

| perturbation | route acc | abstain | semantic-only acc |
| :-- | :-- | :-- | :-- |
| typo_swap | 0.00 | 1.00 | 0.88 |
| typo_drop | 0.00 | 1.00 | 1.00 |
| lower | 1.00 | 0.00 | 1.00 |
| space (hyphen -> space) | 0.00 | 1.00 | 1.00 |
| partial (first token) | 0.00 | 1.00 | 1.00 |
| pronoun (subject -> "it") | 0.00 | 1.00 | 0.10 |

The review's item-2 prediction is exact: the entity gate is a **dictionary** - any
subject-string deviation abstains (fails safe, but recall 0), and **semantic-only
routing recovers 88-100 % of the typo/space/partial cases**. The pinned fix: replace
the substring entity match with an embedding-based (or fuzzy) entity match; pronoun
follow-ups need dialogue coreference and stay out of scope for a stateless router
(semantic-only accuracy 0.10 - there is nothing to match).

## 36. Metric-completion pin (review item 4; 2026-10-06)

The revoke sample default is raised to **100** (min(100, n//2) in practice), per the
review. The reportable pinned metrics at the next full `ours` run: canary KLD per
committed expert (mean/max, from the commit records - the cap keeps each <= 2.0),
behavioural return (`return_match`), retain accuracy, router precision and
abstention. Provenance (leave-one-out) at N=1000 is expensive and stays a separate
pinned run; N=200 used the old metrics, so the final table is built on N=1000 only.

## 37. Strong-RAG N=1000 results (review item 1 complete; 2026-10-06, wall 1280 s)

rag_instruct at N = 1000: **eff 0.837 [0.815, 0.86]**, para 0.306, no_leak 1.00,
route 0.997. Plain RAG: 0.834 [0.812, 0.858]. Ours: 0.999 [0.997, 1.0].
**The gate passes at full scale**: the strongest external-evidence prompt does not
close the gap (0.837 vs 0.834, statistically the same); ours stays CI-separated.
The editing claim stands; the review's contingency (stop and narrow) does not fire.

## 38. Multi-fact delta capacity (review item 5; pinned 2026-10-06, before the run)

One delta trained on the union of k facts' pairs (nonce N=100, group = facts[:k]),
k in {1, 4, 16, 64}; readings per k: in-group efficacy and paraphrase (the delta
active alone), leakage onto 10 held-out facts, canary KLD of the group delta, training
seconds. Predictions: efficacy stays high for small k and degrades with k (a single
delta must hold conflicting pairs); KLD grows with k (the interference budget);
leakage stays ~0 (the facts are unrelated). The reading informs expert granularity:
if a delta carries k* facts, experts can group by subject instead of per-fact.

## 39. Multi-fact delta capacity results, nonce (review item 5; 2026-10-06)

| k | efficacy | paraphrase | leakage | KLD | train s |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | 1.00 | 0.00 | 0.00 | 0.038 | 16.7 |
| 4 | 0.75 | 0.75 | 0.00 | 0.007 | 22.4 |
| 16 | 1.00 | 0.875 | 0.00 | 0.007 | 76.6 |
| 64 | **1.00** | **0.875** | 0.00 | 0.002 | 308.5 |

A **single jointly-trained delta carries 64 facts** at full efficacy with a tiny
canary footprint - the opposite of the summed-independent-deltas collapse (0/1000).
The distinction is joint vs additive training: a shared low-rank solution exists;
summing independently trained solutions interferes. Consequences: per-fact deltas are
not necessary; experts can group ~k* facts (cheaper storage, fewer router keys - the
review's "real MoE" direction). Caveats: nonce facts are templated (5 relations), one
prefix group per k, no seeds/CIs yet; the external-fact check follows.

## 40. Multi-fact delta capacity results, external (CounterFact N=200; 2026-10-06)

| k | efficacy | paraphrase | leakage | KLD | train s |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | 1.00 | 0.00 | 0.10 | 0.014 | 33.8 |
| 4 | 1.00 | 0.25 | 0.00 | 0.011 | 40.1 |
| 16 | 0.938 | 0.438 | 0.00 | 0.003 | 112.8 |
| 64 | **1.00** | **0.656** | 0.00 | 0.007 | 368.2 |

The finding holds on real counterfactual facts: one jointly-trained delta carries 64
facts at full efficacy (paraphrase 0.656 - real paraphrases are harder than nonce;
leakage 0; KLD 0.007). Scale consequences: effective storage per fact drops ~64x (one
~11 MB delta instead of 64 per-fact deltas), router keys drop from facts to groups,
and the expert becomes a grouped jointly-trained organ - the review's real-MoE
direction. Next: group-by-subject experts with a coarse router (which group), then
in-group serve; capacity with seeds/CIs for the paper.

## 41. EasyEdit official baselines (review item 3; pinned 2026-10-06)

Feasibility (researched): EasyEdit main pins transformers 5.5.4 / torch 2.9.1 and
loads Qwen2.5 by name match (no registry change); official qwen2.5-7b hparams exist
for GRACE, WISE, MEMIT and AlphaEdit (adapt to 28 layers for 1.5B); on the gpt2-xl
(1.5B) class GRACE ~6 GB and WISE ~7 GB fit the 8 GB card, MEMIT ~11 GB and AlphaEdit
are tight, and MELO is gpt2-xl-only out of the box (a faithful Qwen port is a research
project - the `melo_like` arm stays the recorded approximation). Plan: a separate venv
(`.venv-easyedit`) and a clone under `.deps/` (both ignored); a CounterFact N=200
subset adapted to the KnowEdit format; run GRACE then WISE on the same facts;
evaluate efficacy/paraphrase/locality with our metric definitions; MEMIT/AlphaEdit
only if VRAM allows. No run starts before zsRE N=1000 finishes.

## 42. EasyEdit setup status (2026-10-06)

Venv `.venv-easyedit` is ready after two pin fixes: `PyYAML==6.0` and
`sentencepiece==0.2.0` have no cp313 wheels and fail to build, replaced with
`PyYAML>=6.0.2` and `sentencepiece>=0.2.2` (requirements otherwise installed as
pinned: transformers 5.5.4, torch 2.9.1). EasyEdit has no pyproject/setup - it is
used via `PYTHONPATH=.deps/EasyEdit`, not installed. Dataset schemas read from the
clone: **zsRE uses `alt` as target_new** (our zsRE runs used answers[0]), so the
comparable set is **CounterFact** (its loader reads the ROME dump fields we used:
requested_rewrite prompt/target_new, paraphrase_prompts). GRACE and WISE have
official qwen2.5-7b hparams (adapt layers to 28 for 1.5B; GRACE ~6 GB, WISE ~7 GB).
Next (GPU after zsRE N=1000): run GRACE then WISE on the CounterFact N=200 subset
with PYTHONPATH=.deps/EasyEdit; evaluate efficacy/paraphrase/locality with our
metric definitions; MEMIT/AlphaEdit only if VRAM allows; MELO has no Qwen support.

## 43. zsRE N=1000 stopped on request (2026-10-06)

Training stopped at the 100/1000 checkpoint (the chunk 100:200 in progress was
abandoned; the checkpoint holds the first 100 facts). Resumable with the same chunked
wrapper; the CounterFact N=1000 result is unaffected.

## 44. LoRA delta improvement ablation (pinned 2026-10-08, before the run)

CounterFact N=50, per-fact deltas, 16 steps, KL anchor (against the true zeroed
base for every variant); variants: baseline (q/v, AdamW 3e-4), olora (orthogonal
init - PiSSA was rejected because it mutates the base weights in place, which is
incompatible with the shared frozen base + independent deltas), lora_plus (B lr x8),
mlp (q/v + gate/up/down at r=16), dora. Readings per variant: efficacy, paraphrase
(delta active alone), mean canary KLD (10-delta sample, cap 2.0), training seconds.
Predictions: OLoRA and LoRA+ improve at equal size (few-step regime); MLP targets
improve paraphrase at 2-3x size; DoRA neutral-to-better; all KLDs stay under the cap
(the guarantee must survive any recipe change).

## 45. LoRA ablation partial results + findings (2026-10-08; mlp/dora running)

| variant | efficacy | paraphrase | KLD mean/max | train s |
| :-- | :-- | :-- | :-- | :-- |
| baseline | 1.00 | 0.86 | 0.036 / 0.062 | 358 |
| olora | 0.08 | 0.00 | 22.35 / 28.16 | 358 |
| lora_plus | 1.00 | 0.90 | 0.014 / 0.021 | 343 |

Findings:
1. **LoRA+ is a free win**: equal size, paraphrase 0.86 -> 0.90, KLD even lower
   (0.014 vs 0.036).
2. **OLoRA fails catastrophically** (efficacy 0.08, KLD 22.4 = 11x the cap): an init
   far from the base is incompatible with the KL-anchored few-step recipe - the same
   family as PiSSA, which was rejected for mutating the shared base.
3. **The pilot's paraphrase ~0.30 was routing, not the delta**: with each fact's own
   delta activated directly (no routing), the baseline delta scores 0.86 on
   paraphrases - confirming the review's item-2 diagnosis. The delta generalises;
   the router did not.

mlp and dora run with a variant-aware evaluation path (the store-based materialisation
cannot host MLP-target deltas); their results append below.

## 46. LoRA ablation complete (2026-10-08)

| variant | efficacy | paraphrase | KLD mean/max | size | verdict |
| :-- | :-- | :-- | :-- | :-- | :-- |
| baseline | 1.00 | 0.86 | 0.036 / 0.062 | 8.7 MB | ok |
| olora | 0.08 | 0.00 | 22.35 / 28.16 | 8.7 MB | **fails cap + quality** |
| **lora_plus** | **1.00** | **0.90** | **0.014 / 0.021** | 8.7 MB | **adopted** |
| mlp | 1.00 | 0.78 | 0.017 / 0.041 | ~2-3x | no gain, worse paraphrase |
| dora | 0.98 | 0.70 | 26.75 / 27.84 | ~same | **fails cap** |

Readings: LoRA+ is the free win (equal size, best paraphrase, lowest KLD). The
base-distant init/form changes (OLoRA, DoRA) break the KLD cap by 11-13x - they are
incompatible with the KL-anchored few-step recipe as-is (would be refused by the
ledger). MLP targets stay under the cap but cost 2-3x and hurt paraphrase. Adopt
LoRA+; keep the baseline otherwise; OLoRA/DoRA rejected (retuning possible but not
worth it at this scale). The paraphrase-vs-routing finding from §45 stands.

## 47. Capacity with the adopted LoRA+ recipe (pinned 2026-10-08, before the run)

CounterFact k in {64, 128, 256}: one grouped delta trained with the adopted LoRA+
recipe (B lr x8, 16 steps, KL anchor vs the true base); readings per k: in-group
efficacy and paraphrase (delta active), leakage on 10 held-out facts, KLD mean/max
(cap 2.0 - the guarantee), train seconds. Baseline k=64 reference: 1.00 / 0.656 /
leak 0 / KLD 0.007. Predictions: LoRA+ holds k=64 and extends to 128; if capacity
binds it shows at 256; KLD stays under the cap throughout; leakage ~0.

## 48. Capacity with LoRA+ results - instability (2026-10-08)

| k | efficacy | paraphrase | KLD | leakage | train s |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 64 | **0.031** | 0.047 | 0.24 | 0.00 | 292 |
| 128 | 0.875 | 0.445 | 0.026 | 0.30 | 571 |
| 256 | **0.035** | 0.039 | 0.025 | 0.00 | 1138 |

The curve is non-monotonic and the k=64/256 cells collapse: **LoRA+ at 8x B-lr is
unstable for grouped training**. Diagnosis: the grouped recipe performs 16 x 2k
individual optimizer steps (the per-pair step structure of _pair_step), so k=64 is
2048 steps at B-lr 2.4e-3 - a 64x amplification vs the per-fact case (32 steps) where
LoRA+ was adopted. The k=128 cell (4096 steps) succeeded by luck (no seed was set).
Baseline grouped (lr 3e-4) was stable (k=64: 1.00). Pinned fix for the next run:
a fixed seed, and either B-lr scaled by 1/sqrt(k) or the step budget normalised
(16 total passes instead of 16 per pair). Until then, **LoRA+ is adopted for
per-fact deltas only; grouped training keeps the baseline recipe**. Leakage 0.30 at
k=128 (template generalisation to held-out subjects) is recorded as a secondary
reading.

## 49. Capacity limit of the adopted grouped recipe, seeded (pinned 2026-10-08, before the run)

CounterFact N=1000, the adopted baseline grouped recipe (lr 3e-4, 16 x 2k per-pair
steps), now with `torch.manual_seed(0)`; k in {64, 128, 256}. Readings as in §40:
in-group efficacy/paraphrase, leakage (10 held-out), KLD mean/max (cap), train
seconds. Reference: k=64 was 1.00 / 0.656 / leak 0 / KLD 0.007 (unseeded).
Predictions: k=64 replicates; 128 degrades mildly; 256 shows the capacity limit;
KLD under cap throughout. LoRA+ stays per-fact-only until its grouped instability
is fixed (§48).

## 50. Capacity limit of the adopted grouped recipe, seeded - results (2026-10-08)

| k | efficacy | paraphrase | leakage | KLD | train s |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 64 | 0.984 | 0.578 | 0.30 | 0.002 | 288 |
| 128 | **1.00** | 0.703 | 1.00 | 0.002 | 696 |
| 256 | **1.00** | 0.66 | 1.00 | 0.007 | 1768 |

**A single grouped delta carries 256 facts at full efficacy** with a negligible canary
footprint (KLD 0.007; cap 2.0) - the seeded run replicates k=64 (0.984, one fact) and
extends the capacity far beyond it. Paraphrase improves with group size (0.578 -> 0.703
-> 0.66). Cost scales linearly (16 x 2k per-pair steps; ~30 min for k=256).

**Leakage caveat (recorded, suspected artifact):** leakage on 10 held-out facts reads
0.30 (k=64) and 1.00 (k=128/256) - but capacity.py's leakage check is NOT
base-referenced, and CounterFact targets concentrate on common languages, so a group
delta with a template prior ("The language of X is English") can hit held-out answers
by collision (the same artifact the pilot fixed with the base-referenced rule). The
pinned follow-up: re-run k=256 with the base-referenced leakage (delta-vs-base on the
held-out probes) before any leakage claim.

**Decision:** the adopted grouped recipe (baseline, lr 3e-4) is stable, seeded and
scales to 256 facts/delta; LoRA+ stays per-fact-only.

## 51. Router v3: fuzzy entity match - built and CPU-validated (2026-10-08)

Pinned fix from review item 2: replace the substring dictionary gate with a fuzzy
entity match. v3: normalize (lower, -/_ -> space); exact normalized substring (longest
wins, prefix safety) else token-level match (edit distance <=1, <=2 for tokens of
length >=6, adjacent transposition = 1); candidates -> semantic key with tau; no
candidate -> abstain.

CPU validation (N=50, same set as the stress; results/router_stress_v2v3_n50.json,
router_eval_v3_n50.json):

| kind | v2 route | v3 route | v3 abstain | semantic-only |
| :-- | :-- | :-- | :-- | :-- |
| typo_swap | 0.00 | 1.00 | 0.00 | 0.88 |
| typo_drop | 0.00 | 1.00 | 0.00 | 1.00 |
| lower | 1.00 | 1.00 | 0.00 | 1.00 |
| space | 0.00 | 1.00 | 0.00 | 1.00 |
| partial | 0.00 | 0.80 | 0.20 | 1.00 |
| pronoun | 0.00 | 0.00 | 1.00 | 0.10 |

Normal path (router_eval): probe 1.00, paraphrase 1.00, distractor abstention 1.00 -
identical to v2 (no regression). Pronoun abstention is correct behaviour (no entity
signal). Owed: pilot re-validation with v3 (route + efficacy unchanged) at the next
GPU run; confirmatory route readings will be reported with the v3 label.

## 52. Router v3 pilot re-validation + revoke-arm bug fix (2026-10-08)

The revoke arm crashed on f0010: pre-revoke references covered only 10 facts while
run_revoke samples min(100, n//2); the mismatch appeared when the revoke sample was
raised to 100 for the pinned metric completion. Fixed: pre covers exactly the revoke
sample and is only built when the ours arm runs. Recorded as a bug fix, not patched
silently.

N=50 nonce re-run with router v3 (seeded, --arms ours; results/pilot_v3_n50.json):
efficacy 0.82 [0.70, 0.92], paraphrase 0.76 [0.64, 0.88], route 1.00, paraphrase
route 1.00, distractor abstention 1.00, no-leak 1.00 - identical to the v2 record
(pilot_n50_v2.json, same seed: the N=50 harness value is 0.82, not 1.00). v3 changes
nothing on the normal path and fixes the perturbation cases (SS51). Revoke (sample 25,
the pinned rule): token gone 1.00 [1.00, 1.00], return match 1.00, retain 0.92
[0.80, 1.00]. The revoke n=100 metric now runs at the next N=1000 ours run.

## 53. Official EasyEdit baselines: GRACE and WISE on CounterFact N=200 (pinned 2026-10-08, before the runs)

Purpose: review item 3 - close the "strawman baselines" gap with the official
implementations. Model: Qwen2.5-1.5B-Instruct (same as ours). Data: the same
CounterFact N=200 subset (external_counterfact_n200.json; aligned to
raw/counterfact.json). Hparams: EasyEdit GRACE/WISE defaults adapted to the 1.5B
(28 layers, hidden 1536): GRACE layer 18 / n_iter 50 / lr 1.0; WISE layer 23 /
n_iter 70 / retrieve / chat template. Sequential editing of all 200 facts (the
setting our protocol uses). Metrics: (a) EasyEdit's own post/rephrase/locality;
(b) our protocol where the edited model is evaluable (greedy + contains on
probe/paraphrase/locality; canary KLD if the edited model is directly usable).
Predictions: strong first-edit efficacy with decay over 200 sequential edits for
both; KLD not protected (no anchor) and likely above our cap (2.0) for at least one
method; locality gaps reported as-is. Runs are detached; failures recorded, not
patched silently.

## 54. Official EasyEdit GRACE/WISE on CounterFact N=200 - results (2026-10-09)

Both official implementations ran end-to-end on the pinned subset (Qwen2.5-1.5B-Instruct,
sequential 200 edits, fp16 - the fp32 official default OOMs on the shared 8 GB GPU;
recorded as an adaptation). EasyEdit's own metrics, our protocol (chat serving) and a
raw-prompt diagnostic:

| method | EasyEdit post | ours-protocol eff (chat) | raw-prompt eff | paraphrase | no-leak | canary KLD | wall |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| GRACE | rewrite_acc 0.0* (token-EM artifact) | 0.025 | **1.00** | 0.02 | 1.00 | 0.0 | 44m |
| WISE | rewrite_acc 1.0 (per-step) | 0.07 | 0.04 | 0.06 | 0.64 | 0.0 | 49m |

Honest findings (all in the logs/JSON quirks):
1. EasyEdit GRACE token-EM is space-sensitive (Qwen emits " Russian" id 8522 vs
   tok.encode("Russian") 47707) -> 0.0 for all targets; uninformative, our diagnostic
   is the meaningful measure.
2. GRACE edits are perfect on raw prompts (200/200) but do not fire under
   chat-templated serving: keys are built on the raw prompt's last-token activation,
   and the chat template changes that token. GRACE works in its intended format; the
   like-for-like comparison must label formats.
3. WISE's official per-step metric (1.0) hides final-model forgetting (0.07): with
   save_freq=500/merge_freq=1000 and only 200 edits, WISE-Retrieve never merges, so
   all edits share one new_weight.
4. canary KLD 0.0 for both (the prediction in SS53 did not fire): the edited paths
   fall back to the original layer for canary prompts (GRACE keys / WISE activation
   gate). Recorded as-is.
5. WISEHyperParams has no sequential_edit field although WISE.__init__ reads it; set
   at runtime so the adapter accumulates edits.

Review item 3 is closed with these caveats. Pinned follow-ups (optional): WISE with
merge_freq <= 200 (its intended lifecycle) and a format-matched GRACE-vs-ours
comparison.

## 55. Metric completion at N=1000 on the existing ckpt (pinned 2026-10-09, before the run)

Review item 4: eval-only rerun of the complete CounterFact N=1000 store
(cf_n1000.ckpt) with the v3 router and the fixed revoke arm (sample 100). No
retraining: the deltas are the ones measured in the strong-RAG gate run, so the
readings are directly comparable except for the router version. Readings: the
standard ours block (efficacy/paraphrase/no-leak/route/abstention) plus revoke
(token gone, return match, retain) and canary KLD. Predictions: efficacy ~0.999
replicates; distractor abstention stays 1.00 under v3 (watch for false candidates
from fuzzy token matches on real names); revoke gone 1.00, return match 1.00,
retain >=0.98; canary KLD <=0.05.

## 56. Metric completion at N=1000 - results (2026-10-09)

Eval-only rerun of cf_n1000.ckpt with router v3 and the fixed revoke arm (wall 1211 s):

| metric | value | CI | threshold | verdict |
| :-- | :-- | :-- | :-- | :-- |
| efficacy | 0.999 | [0.997, 1.0] | - | replicates the gate run |
| paraphrase (routed) | 0.323 | [0.295, 0.352] | - | known router-bound limitation (para route 0.362) |
| distractor no-leak | 0.992 | [0.986, 0.997] | >=0.95 | pass |
| route accuracy | 0.997 | - | >=0.95 (precision proxy) | pass |
| distractor abstention | 0.992 | - | >=0.95 | pass (prediction 1.00; 8 false candidates - the v3 watch item fired slightly) |
| revoke gone (n=100) | 0.96 | [0.92, 0.99] | - | prediction 1.00 missed: token collisions (coinciding language targets), 4/100 |
| return match | 0.96 | [0.92, 0.99] | - | follows gone |
| retain | 0.9989 | [0.9967, 1.0] | >=0.98 | pass |
| revoke latency | 1.4 ms | - | - | exact-revoke speed |
| storage | 8.7 GB | - | - | 8.7 MB/fact |

Owed from the review list: canary KLD at N=1000 (not in the eval-only JSON; measured
from a sampled pass over the ckpt - pinned follow-up) and provenance (separate run).

## 57. zsRE N=1000 resume (pinned 2026-10-09, before the run)

The zsRE ckpt is at 300/1000 (earlier note said 100; verified by loading). Resume the
same chunked wrapper over chunks 300:400 ... 900:1000, then the standard eval-only
block with router v3 and revoke n=100 (same readings as SS56). Predictions: chunk
completion without failures; final efficacy in the range of the N=200 zsRE run,
revoke/retain per SS56. Failures recorded, not patched silently.

## 58. zsRE N=1000 - final results (2026-10-09)

Resumed 300->1000 (7 chunks, ~23 min each; final eval wall 1558 s) with the standard
eval-only block (router v3, revoke n=100):

| metric | value | CI | threshold | verdict |
| :-- | :-- | :-- | :-- | :-- |
| efficacy | 0.996 | [0.992, 0.999] | - | replicates the N=200 level |
| paraphrase | 0.973 | [0.962, 0.982] | - | zsRE paraphrases keep the subject -> routing works (para route 0.776) |
| distractor no-leak | 0.975 | [0.965, 0.984] | >=0.95 | pass |
| distractor abstention | 0.967 | - | >=0.95 | pass |
| route accuracy | 0.784 | - | - | many facts per subject: the gate fires but the semantic pick among siblings is 78.4% |
| revoke gone (n=100) | 0.36 | [0.26, 0.45] | - | base-confounded: zsRE answers are often known by the base, so the token is present after revoke anyway |
| return match | 0.80 | [0.72, 0.88] | - | base-referenced reading; the 20% gap is sibling-routing interference |
| retain | 0.9944 | [0.9889, 0.9989] | >=0.98 | pass |
| storage | 8.7 GB | - | - | same recipe |

Reading notes (harness semantics, verified in code):
1. Probe serving: when the entity gate fires (any route), the fact's OWN expert is
   served (eval_ours line 282), so probe efficacy is robust to sibling misroutes;
   route_accuracy measures the semantic pick among candidates.
2. Paraphrase serving uses the ROUTED fact's expert (line 289) - with sibling subjects
   the misroutes would hurt, but paraphrases keep the subject and 0.973 holds.
3. Revoke: gone is absolute (token absent) and confounded by base knowledge; the
   base-referenced return_match (0.80) is the meaningful reading. The 20% mismatch is
   sibling-routing interference (router_keep routes to remaining siblings whose deltas
   change the response), a zsRE-specific caveat.

## 59. Canary KLD sample over the N=1000 ckpts (pinned 2026-10-09, before the run)

Review item 4 residual: canary KLD at N=1000 was not in the eval-only JSON. Measure
_expert_kld (the commit-gate measure) on the first 50 deltas of cf_n1000.ckpt and
zsre_n1000.ckpt. Threshold: mean <= 0.05 (cap 2.0). Prediction: <= 0.05 (the commit
gate enforces <= 2.0; the N=50 variants measured 0.014-0.036).

Results (first 50 deltas of each ckpt):
- cf_n1000: mean 0.0215, max 0.0498, p95 0.0447 - pass (mean <= 0.05; max just under).
- zsre_n1000: mean 0.0171, max 0.0529, p95 0.0376 - mean pass; one delta marginally
  above 0.05 at max (0.0529), well under the 2.0 cap.

Verdict: the review threshold holds as a mean over sampled deltas; recorded with the
single marginal max at zsRE. The commit gate (2.0) was enforced during training by
construction; the sample confirms the realized footprint.

## 60. Base-referenced leakage re-check at k=256 (pinned 2026-10-09, before the run)

SS50 follow-up: the absolute leakage at k=256 (1.00) is suspected to be an artifact
(not base-referenced + coinciding language targets). capacity.py now also reports
leakage_base_ref (delta hit AND base miss on the 10 held-out facts) and the base hit
rate; the k=256 run is repeated with the same seed. Predictions: base hit rate > 0
(the collision mechanism), leakage_base_ref much lower than 1.00 - likely 0-0.3;
efficacy/KLD replicate SS50.

### SS60 addendum (2026-10-09): the first re-check exposed a bug

The first k=256-only re-run gave efficacy 0.707 (SS50: 1.00), leakage 0.7,
leakage_base_ref 0.7, base_outside_hit_rate 0.0, KLD 0.032. Two findings:
1. The leakage slice is broken: outside = facts[64:74] lies INSIDE the group for
   k > 64 - the SS50 leakage 1.00 at k=128/256 was in-group memorization, not
   leakage (and not a base-collision artifact either; base hit rate is 0.0).
2. The k=256-only run consumed different RNG draws for the delta init (the SS50
   sequence trains k=64 and k=128 first) - efficacy 0.707 vs 1.00 shows the
   baseline grouped training at k=256 is RNG-position sensitive. Recorded.

## 61. Capacity leakage, corrected (pinned 2026-10-09, before the run)

outside = facts[k:k+10] per k (genuinely outside), base-referenced leakage, the
SS50 sequence --ks 64,128,256 (same seed; greedy generations consume no RNG, so
the training sequence matches SS50). Predictions: k=64 replicates (~0.3 leakage,
which is genuine); k=128/256 leakage drops far below 1.00 (real leakage only);
efficacy/KLD replicate SS50 (1.00 / 0.66 / 0.007) if training is deterministic.

Results (sequence 64,128,256; deterministic replication of SS50's eff/para/KLD to
the digit):

| k | efficacy | paraphrase | leakage (corrected) | base hit | KLD | SS50 leakage |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 64 | 0.984 | 0.578 | 0.3 | 0.0 | 0.002 | 0.30 (valid) |
| 128 | 1.00 | 0.703 | 0.1 | 0.0 | 0.002 | 1.00 (slice bug) |
| 256 | 1.00 | 0.66 | 0.3 | 0.0 | 0.007 | 1.00 (slice bug) |

Verdict: the grouped delta's genuine leakage (base-referenced; base hit rate 0.0
throughout) is 0.1-0.3, not 1.00 - the SS50 1.00 readings were the in-group slice
bug. Template generalisation to held-out subjects is real but modest (10-30% of
held-out answers produced). Efficacy/paraphrase/KLD replicate SS50 exactly,
confirming the training is deterministic given the RNG sequence (the k=256-only
0.707 in the SS60 addendum was RNG-position sensitivity, not a capacity effect).

## 62. Provenance (G5 leave-one-out) at N=1000 (pinned 2026-10-09, before the run)

First 100 facts of the cf_n1000 store: serve with the fact's expert, suspend it
(remove + materialize), serve again, restore; attribution = the answer is present
with the expert and absent without it; the store state hash must return to its
value. Threshold from the review list: provenance >= 0.95. Prediction: ~1.00 (the
N=50 c6-v2 check measured 1.00; each fact's expert is independent).

Results: attribution 1.00 (100/100), state hash stable (true), mean 0.434 s/fact -
the threshold (>=0.95) passes with margin; the N=50 c6-v2 reading scaled to N=1000.
With this, all review item 4 readings exist at N=1000: canary KLD mean <=0.05 (SS59),
retain 0.9989, provenance 1.00, router precision 0.997 / abstention 0.992, revoke
n=100 (gone 0.96 / return 0.96 / retain 0.9989, SS56).

## 63. WISE with its intended lifecycle (merge within 200 edits) - pinned 2026-10-09, before the run

SS54 caveat: with the official save_freq=500/merge_freq=1000, 200 edits never
merge, so the final model forgets (0.07). Variant: save_freq=100, merge_freq=200
(everything else official), same N=200 subset and runner; a 2-edit smoke with
merge_freq=2 validates the merge path first. Readings: EasyEdit metrics + our
protocol + canary KLD, same as SS54. Prediction: the final model improves
substantially over 0.07 (the merge consolidates the edits), but per-edit decay is
still expected; report as-is.

Results (N=200, save_freq=100, merge_freq=200; the merge triggered at edit 200):
- EasyEdit: post rewrite_acc 0.66 (per-step 1.00 in SS54), rephrase 0.615, locality 1.00.
- Our protocol: efficacy 0.125 [0.08, 0.17] (SS54: 0.07), paraphrase 0.105,
  distractor_no_leak 0.27 [0.21, 0.335] (SS54: 0.64), raw diagnostic 0.05,
  canary KLD 0.0.

The intended lifecycle improves the final model over the never-merged run
(0.07 -> 0.125) but per-edit decay persists and locality collapses: no-leak 0.27
means the merged weights fire on unrelated prompts (73% leak). Even with merge,
WISE stays far from ours (eff 0.999 / no-leak 0.992 at N=1000; SS56/58). Caveat as
in SS54: chat-templated serving vs EasyEdit's raw protocol; the raw diagnostic is
0.05, so the gap is not a serving-format artifact alone.

## 64. Routed (non-oracle) evaluation of the N=1000 stores (pinned 2026-10-10, before the run)

Harness-semantics correction found while drafting the paper: in `eval_ours` (and the
SS56/SS58 eval-only reruns) the probe is served by the fact's OWN expert whenever the
entity gate fires anywhere (SS58 reading note 1), so the reported efficacy (0.999 /
0.996) is conditioned on the gold expert and is not end-to-end routed efficacy; the
retain reading has the same oracle substitution. Paraphrase, distractor and revoke-gone
already used the routed expert. `confirm/routed_eval.py` (eval-only, no training, inputs
verified against `routed_eval_manifest.json` SHA-256 pins) serves every query - probe,
paraphrase, distractor, post-revoke probe and retained probe - with the router-selected
expert (base on abstention) and logs every route and response. Runs: cf_n1000.ckpt and
zsre_n1000.ckpt, router v3, revoke sample 100; outputs `*_n1000_routed_eval.json`.
Predictions: CounterFact routed probe efficacy ~0.99 (route 0.997); zsRE routed probe
efficacy well below 0.996, near the route accuracy (0.784) plus base hits on sibling
misroutes; paraphrase / no-leak / abstention replicate SS56/SS58; routed retain on zsRE
below 0.9944. Whatever the numbers, the paper reports the routed readings as the
primary efficacy and the oracle-conditioned ones as an upper bound labelled as such.

### SS64 results (2026-10-10): routed end-to-end readings

Eval-only, inputs verified against the manifest SHA-256 pins; walls 1826 s (cf) and
1573 s (zsRE); outputs `results/live_learning/confirm/{cf,zsre}_n1000_routed_eval.json`.

| metric (routed) | CounterFact | zsRE | earlier (gold-expert / SS56, SS58) |
| :-- | :-- | :-- | :-- |
| probe efficacy | 0.997 [0.993, 1.0] | 0.994 [0.989, 0.998] | 0.999 / 0.996 |
| paraphrase efficacy | 0.323 [0.295, 0.352] | 0.973 [0.962, 0.982] | replicates |
| probe route accuracy | 0.997 | 0.784 | replicates |
| distractor abstention / no-leak | 0.992 / 0.992 | 0.967 / 0.975 | replicates |
| revoke gone / return match (n=100) | 0.96 / 0.96 | 0.36 / 0.80 | replicates |
| retain (900 kept, routed) | 0.9967 [0.9922, 1.0] | 0.9922 [0.9856, 0.9978] | 0.9989 / 0.9944 (gold) |
| revoke latency | 2.6 ms | 2.4 ms | 1.4 ms |

Readings against the predictions: CounterFact as predicted (~0.99). **zsRE prediction
missed in the favourable direction**: routed efficacy 0.994, not near 0.78. Diagnosis
from the per-query logs: 360/1000 zsRE facts share subject AND answer with another fact
posed as a different question (rephrased duplicates); all 210 probe misroutes land on
such a duplicate and 208 are answered correctly; no subject in the sample has two facts
with different answers, so conflicting-sibling disambiguation is untested. The
pre-registered router-precision threshold (>= 0.95, same fact id) therefore fails
nominally on zsRE (0.784) and is reported as a miss. Further diagnoses: (1) CounterFact
paraphrase - all 636 unrouted paraphrases passed the entity gate and fell below tau
(median sim 0.53 vs 0.656; misroutes 0.2%); routed paraphrases that reach their own
expert hit 0.865 (= the direct-serving 0.86). (2) CounterFact revoke: base hit 0/100; the
4 not-gone cases are fuzzy-gate routes to another stored fact sharing a subject token
and the same target (e.g. Chevrolet Caprice -> Chevrolet Chevette, "IBM"). (3) zsRE
revoke: base hit 44/100; all 20 return mismatches are base-miss -> still-hit, 15 routed
to a stored rephrased duplicate of the revoked fact (revocation is per entry). (4) Every
distractor leak is a fuzzy match of a synthetic name to a real subject (Muku-75 -> Mukul
Roy; Fapa-91 -> Scott Fava; Veho-65 -> Morro Velho). Paper 1 reports the routed readings
as primary (paper/numbers.tex is generated by paper/make_numbers.py).

## 65. Paraphrase-conditional retrieval comparison and harder locality (pinned 2026-10-10, before the run)

Review of the paper draft (external, 2026-10-10): (1) edit-prompt efficacy is near
tautological (the adapter trains on the edit prompt and the router key is its embedding),
and on paraphrases RELIC and retrieval are equal (0.323 vs 0.306) because both share a
router that declines 64% of CounterFact paraphrases - the adapter-vs-retrieval claim is
untested where it matters; (2) locality is measured only on synthetic-name distractors;
the risky case (a stored subject asked about another relation) and the standard
CounterFact neighborhood prompts are unmeasured. Eval-only run on cf_n1000.ckpt (same
manifest verification as SS64), `confirm/locality_eval.py`, router v3, tau 0.656:

- **E1 (retrieval on routed paraphrases):** every paraphrase is routed once; retrieval
  (instructed prompt, the routed fact's statement) and RELIC (the routed adapter) answer
  it; abstentions are served by the base for both. Readings: accuracy of both on all
  1000 and on the route-correct subset (362 in SS64), paired bootstrap CI of the
  difference. Prediction: on the route-correct subset RELIC ~0.86, retrieval clearly
  lower (60-80%), CI of the difference excludes 0.
- **E2 (CounterFact neighborhood):** the first neighborhood prompt of each fact (other
  subject, same relation, true answer = target_true). Readings: abstention, output
  identical to the base output, leak (stored target_new present, absent from base).
  Prediction: abstention >= 0.98, identical >= 0.98, leak <= 0.01.
- **E3 (same subject, other relation):** for each fact, a prompt template of another
  CounterFact item with a different relation_id (drawn with random.Random(65)), filled
  with the fact's subject. Readings as E2, plus the routed rate. Prediction: the entity
  filter passes all; tau declines most (routed 10-40%); among routed queries the adapter
  often changes the answer; leak of the stored target 5-20% overall. Reported as-is.

### SS65 results (2026-10-10)

Eval-only, wall 1892 s, `results/live_learning/confirm/cf_n1000_locality_eval.json`.

| reading | value | prediction | verdict |
| :-- | :-- | :-- | :-- |
| E1 all paraphrases: ours / retrieval | 0.323 / 0.306, diff +0.017 [-0.001, 0.034] | - | equal |
| E1 route-correct (n=362): ours / retrieval | 0.865 / 0.818, diff +0.047 [-0.003, 0.097] | CI excludes 0, retrieval 0.60-0.80 | **missed**: not separated |
| E2 neighborhood: routed / identical to base / leak | 0.098 / 0.902 / 0.030 [0.020, 0.041] | routed <= 0.02, leak <= 0.01 | **missed** |
| E3 same subject, other relation: routed / identical / leak | 0.782 / 0.222 / 0.346 [0.316, 0.374] | routed 0.10-0.40, leak 0.05-0.20 | **missed** (worse) |
| synthetic distractors (from SS64): routed / identical | 0.008 / 0.992 | - | reference |

Readings: (1) every abstained query (E2 902/902, E3 218/218, synthetic 992/992) returned
exactly the base output, and nearly every routed one did not (E2 0/98, E3 4/782
identical): locality equals the router's abstention rate per query type. (2) E2 routes
come from the approximate entity match on shared first names or common tokens
("Jean-Baptiste Say" -> "Jean Galland"; "Antarctic Treaty System" -> "Indian Antarctic
Program"). (3) E3: the entity filter passes by construction and tau does not separate
relations - the threshold was calibrated without same-subject negatives; routed outputs
contain the stored target in 44% and are sometimes degenerate ("Noted: Minnesota
Glacier", the teach-pair echo). (4) The adapter-over-retrieval advantage is established
only on the trained edit prompts (0.997 vs 0.837); on held-out paraphrases that reach the
right fact it is +4.7 points and not significant. Consequences for paper 1: the locality
claim is narrowed to unknown subjects; the retrieval claim is narrowed to trained
prompts. Pinned candidates for paper 2: relation-aware keys or a threshold calibrated on
same-subject negatives; contrast pairs for other relations of the stored subject (the
VLM recipe); token-level fuzzy matching restricted to full-subject alignment.

## 66-69. Fixing the router, the check and the concept (pinned 2026-10-11, before any run)

Diagnosis from SS64/65: the router compares the embedding of the WHOLE query with the
embedding of the edit prompt, so subject and relation are entangled. This one decision
explains four failures: other relations of a stored subject pass tau (E3, 78% routed),
CounterFact paraphrases with an unrelated leading sentence fall below tau (64%
declined), a shared first name or common token produces an entity candidate (E2, 9.8%),
and same-subject siblings cannot be separated. The adapters themselves generalise (0.86
on paraphrases that reach them). The goal of this block is to fix the system, not to
defend v3; whatever the outcome, the paper reports it.

### 66. Router v4 (CPU, no retraining)

1. Entity filter: exact normalized subject match with word boundaries; otherwise an
   approximate match must align ALL subject tokens to consecutive query tokens (tokens
   < 4 characters exact, OSA <= 1, <= 2 for >= 6 characters). No single-token partial
   matches. All facts whose subject equals the matched subject are candidates
   (siblings are kept, not reduced to the first one).
2. Clause scoping: only the sentence of the query that contains the matched subject is
   used (split on . ! ? ;).
3. Relation key: the subject span is replaced by a placeholder in both the clause and the
   fact's edit prompt; the frozen MiniLM encoder embeds these relation texts; cosine
   similarity picks among the candidates.
4. tau_v4 is calibrated on a DEV split disjoint from the test facts (500 CounterFact
   and 500 zsRE facts, different subjects, seed 66): positives = held-out paraphrases
   (both CounterFact paraphrase prompts; zsRE rephrase), negatives = same-subject
   other-relation prompts (templates of other CounterFact relations, seed 66). Rule:
   tau maximising Youden's J (TPR - FPR) on the pooled dev set. Fixed before the test.

CPU readings on the test sets (routing only): probe route, paraphrase route, E3 routed
rate, E2 / synthetic-distractor routed rate, the nonce perturbation stress test, and a
sibling test (for each stored CounterFact subject, a second synthetic fact on another
relation is added to the router; the original probe/paraphrase must still select the
original fact and the sibling's probe the sibling). Predictions: probe route >= 0.99;
CF paraphrase route >= 0.75 (from 0.36); E3 routed <= 0.10 (from 0.78); E2 routed
<= 0.01 (from 0.098); partial-subject stress cases drop (accepted trade-off); sibling
selection >= 0.95.

### 67. GPU re-evaluation with router v4 (eval-only, saved stores)

`confirm/v4_eval.py` on cf_n1000.ckpt and zsre_n1000.ckpt, manifest-verified inputs.
Readings per set: routed efficacy, paraphrase accuracy, synthetic-distractor leak,
removal of the first 100 (gone / return / retention); CounterFact E2 and E3 locality;
zsRE locality on the dataset's `loc` questions (same-as-base, leak); zsRE removal by
content (remove every entry with the subject and answer of the removed fact). Arms on
probes and paraphrases: adapter, instructed retrieval, adapter + note (both). Predictions:
CF paraphrase 0.65-0.80 (from 0.32); E3 leak <= 0.05 (from 0.346); E2 leak <= 0.01;
efficacy unchanged within 0.01; zsRE return with removal by content >= 0.95 (from 0.80);
the paired adapter-minus-retrieval difference on routed paraphrases is reported as-is.

### 68. Tiered store (concept change, computed from the 67 logs)

Retrieval and adapters are both entries of the store with the same transaction
semantics (a note is removed exactly by deleting it). Policy fixed now: a fact is
stored as a note if instructed retrieval answers its edit prompt at write time;
otherwise as an adapter (the decision uses only the edit prompt, which is available at
write time). Readings: fraction of facts that need an adapter (storage), efficacy and
paraphrase accuracy of the tiered store vs adapter-only and retrieval-only, locality.
Prediction: about 15-20% of CounterFact facts need adapters (5x storage cut); tiered
efficacy >= 0.99; tiered paraphrase >= adapter-only - 0.03.

### 69. A divergence check that can bind, and the cost of a wrong route (eval-only)

(a) A held-out canary battery of 40 general prompts (facts, arithmetic, instructions,
other languages) disjoint from the 2 training canaries; D_heldout for the first 100
CounterFact adapters next to the in-sample D. (b) Damage of a wrong route: 200 zsRE dev
questions the base model answers correctly, each served with a randomly drawn stored
CounterFact adapter; retained accuracy. (c) kappa is recalibrated from (a) as the 99th
percentile of D_heldout over benign adapters (a data-derived budget instead of 2.0).
Predictions: D_heldout mean <= 0.1 with a long tail; retained accuracy under a wrong
route 0.6-0.9 (wrong routes are harmful, which is why routing precision matters).

### SS66 results (2026-10-11, CPU, `router_v4_calibration.json`)

Calibration on the pooled dev split: tau_v4 = 0.6308 (Youden J 0.725, TPR 0.868, FPR
0.143; 1500 positives, 1000 negatives). Test routing, v3 -> v4:

| reading | CounterFact v3 | CounterFact v4 | zsRE v3 | zsRE v4 | prediction |
| :-- | :-- | :-- | :-- | :-- | :-- |
| probe route correct | 0.997 | 0.998 | 0.784 | 1.000 | >= 0.99: met |
| paraphrase route correct | 0.362 | 0.814 | 0.776 | 0.740 | CF >= 0.75: met |
| synthetic distractor routed | 0.008 | 0.000 | 0.033 | 0.000 | - |
| E2 neighborhood routed | 0.098 | 0.018 | - | - | <= 0.01: missed (close) |
| E3 other relation routed | 0.782 | 0.290 | - | - | <= 0.10: **missed** |
| zsRE loc questions routed | - | - | 0.000 | 0.000 | - |

The paraphrase failure is fixed and E3 drops 2.7x, but relation texts of different
CounterFact relations are often closer than tau; the router alone cannot close E3. Note:
the pooled calibration under-estimates the CounterFact FPR (the zsRE negatives are easy);
tau stays as pinned, the per-dataset sensitivity is reported as secondary.

## 70. Locality-aware adapter training (pinned 2026-10-11, before the run)

Adapter-side fix for E3: the KL term of Eq. (1) is applied, in addition to the two
generic canaries, to three prompts about the fact's OWN subject with templates of other
CounterFact relations (seed 70), so the adapter learns to leave other questions about
its subject unchanged (the text analogue of the VLM contrast pairs). The relation used by
the SS65 E3 test prompt and the fact's own relation are excluded from these training
prompts. Everything else as the confirmatory recipe (r=16 q,v, lr 3e-4, 16 x 2 steps,
lambda 1, commit check). Facts: the first 200 CounterFact test facts; comparison with
their existing adapters (cf_n1000.ckpt) under the same router (v4, tau_v4) and v3.
Readings: efficacy, paraphrase accuracy, E3 same-as-base / leak (SS65 prompts), E2,
in-sample and held-out D (SS69 battery), training seconds. Predictions: E3 leak with
router v4 drops from the existing adapters' value to <= 0.03; efficacy >= 0.98;
paraphrase within 0.05 of the existing adapters; D unchanged or lower.

### SS69 results (2026-10-11, `canary_eval.json`)

| reading | value | prediction |
| :-- | :-- | :-- |
| benign D, 2 training canaries (n=100) | mean 0.022, p95 0.049, max 0.078 | - |
| benign D, 40 held-out canaries | mean 0.364, p95 0.72, max 1.07 | mean <= 0.1: **missed** |
| harmful (no KL, lr 1e-3, 32 steps; n=10) in-sample / held-out | 2.03-32.9 / 3.46-28.8 | - |
| kappa_heldout (p99 of benign held-out D) | 0.968 | - |
| harmful rejected: held-out kappa / in-sample kappa 2.0 | 10/10 / 10/10 | - |
| wrong route: base-correct zsRE dev questions retained (n=200) | 0.72 | 0.6-0.9: met |

Readings: the in-sample D under-states the footprint ~17x because the KL term is trained
on the same two prompts; the held-out battery gives a measure the training cannot fit.
Both checks reject the 10 harmful candidates, but the in-sample margin is nil (min 2.03
vs 2.0) while the held-out margin is 3.6x (min 3.46 vs 0.968). A wrong route destroys
28% of correct base answers, so routing precision is a safety property, not only an
accuracy one. Decision: the paper's check becomes D on the held-out battery with the
data-derived kappa; the 2-prompt D remains the training anchor only.

### SS67/68 results (2026-10-11, `{cf,zsre}_n1000_v4_eval.json`, walls 1565 s / 1647 s)

| reading (router v4, existing adapters) | CounterFact | zsRE | v3 reference | prediction |
| :-- | :-- | :-- | :-- | :-- |
| efficacy, adapter | 0.997 | 0.999 | 0.997 / 0.994 | unchanged: met |
| paraphrase, adapter | 0.669 | 0.959 | 0.323 / 0.973 | CF 0.65-0.80: met |
| paraphrase, instructed retrieval | 0.629 | 0.817 | 0.306 / - | - |
| paraphrase, adapter + note | 0.803 | 0.979 | - | - |
| paraphrase adapter - retrieval (paired) | +0.040 [0.011, 0.068] | +0.142 [0.117, 0.167] | +0.017 n.s. | reported |
| paraphrase on route-correct: adapter / retrieval / both | 0.815 / 0.765 / 0.979 | - | 0.865 / 0.818 / - | - |
| synthetic distractor no-leak | 1.000 | 1.000 | 0.992 / 0.975 | - |
| E2 neighborhood: routed / same / leak | 0.018 / 0.982 / 0.014 | - | 0.098 / 0.902 / 0.030 | leak <= 0.01: missed (close) |
| E3 other relation: routed / same / leak | 0.290 / 0.711 / 0.186 | - | 0.782 / 0.222 / 0.346 | leak <= 0.05: **missed** |
| zsRE loc (NQ) questions: routed / same / leak | - | 0.000 / 1.000 / 0.000 | - | - |
| removal: gone / return (n=100) | 1.00 / 1.00 | 0.42 / 0.86 | 0.96 / 0.96, 0.36 / 0.80 | - |
| removal by content (zsRE, +38 entries): return | - | 1.00 | - | >= 0.95: met |
| retention (routed) | 0.9967 | 0.9989 | 0.9967 / 0.9922 | - |

SS68 tiered store (policy pinned): adapter fraction 0.162 (CF) / 0.206 (zsRE)
[prediction 0.15-0.20: met]; efficacy 0.998 / 1.000; paraphrase 0.664 / 0.931
[>= adapter-only - 0.03: met]. Post-hoc variants (labelled as such, computed from the
same logs): note for every fact plus an adapter where the note fails, served with both:
CF efficacy 0.998, paraphrase 0.677; adapter and note for every fact: CF paraphrase
0.803 [0.778, 0.827], +0.134 [0.113, 0.157] over adapter alone.

Readings: the router was the main cause of the v3 failures. With v4, removal on
CounterFact is complete (the four SS64 residues were fuzzy routes), the paraphrase gap to
the direct-serving ceiling closes from 0.54 to 0.19, and serving the adapter together
with its note is the strongest arm (0.979 on CounterFact paraphrases that reach the right
fact). E3 remains open: when an other-relation query is routed (29%), the adapter emits
the stored answer in 64% of cases - addressed on the adapter side by SS70.

### SS70 results (2026-10-11, `local_anchor_n200.json`, 200 CounterFact facts)

All 200 local-anchor adapters passed the training check (D on the 2 canaries: mean
0.014, max 0.053); mean training time 18.5 s.

| reading (router v4) | existing adapters | local-anchor adapters | prediction |
| :-- | :-- | :-- | :-- |
| efficacy | 1.000 | 0.930 [0.89, 0.965] | >= 0.98: **missed** |
| paraphrase | 0.690 | 0.550 [0.48, 0.62] | within 0.05: **missed** |
| E3 routed / same / leak | 0.265 / 0.740 / 0.160 | 0.265 / 0.745 / 0.090 | leak <= 0.03: **missed** |
| E2 routed / same / leak | 0.010 / 0.990 / 0.010 | 0.010 / 0.990 / 0.010 | - |
| D held-out battery (mean, 50 adapters) | 0.343 | 0.146 | unchanged or lower: met |
| under router v3: efficacy / paraphrase / E3 leak | 1.000 / 0.355 / 0.435 | 0.930 / 0.290 / 0.210 | - |

Readings: subject-local KL prompts halve the other-relation leak (0.160 -> 0.090 under
v4, 0.435 -> 0.210 under v3) and cut the held-out divergence 2.4x, but they cost 7 points
of efficacy and 14 points of paraphrase accuracy: the adapter cannot separate "change the
answer to this relation of this subject" from "leave this subject alone" with three
anchor prompts and 32 steps. It is a trade-off, not a fix. Decision: not adopted as the
default recipe; the existing adapters stay. The remaining lever is the serving arm (SS71).

## 71. Locality of the serving arms (pinned 2026-10-11, before the run; eval-only)

SS65-SS70 measured locality only with the adapter arm, but the system the paper proposes
serves notes, adapters, or both (SS67/68). (a) CounterFact n=1000, existing adapters,
router v4 (tau_v4): for the E3 (SS65 other-relation) and E2 (neighborhood) prompts,
routed / same-as-base / leak under four serving policies: adapter, note (instructed
retrieval), both, and the SS68 tiered store (note-tier facts served by their note,
adapter-tier facts by their adapter; tiers from `cf_n1000_v4_eval.json`). Leak as
before: any stored non-numeric answer appears that the base answer did not contain.
(b) The 200 SS70 local-anchor adapters served with their note (both): efficacy,
paraphrase, E3 leak, to test whether the note recovers what the local KL costs.
Predictions: E3 leak note <= 0.05; both between note and adapter; tiered <= 0.06;
local + note: efficacy >= 0.98, paraphrase >= 0.75, E3 leak <= 0.08.
Decision rule fixed now: the paper's default serving policy is the one with the highest
CounterFact paraphrase accuracy among the policies whose E3 leak is <= 0.10, with storage
(adapter fraction) as the tie-break; the others are reported.

## 72. Router v5: the store's own keys as contrast (pinned 2026-10-11, before the run)

SS71(a) partial reading (E3 leak: adapter 0.186, note 0.140, both 0.237, tiered 0.155)
shows that every serving arm leaks once an other-relation query is routed; the lever is
the router. v4 accepts the best candidate if its relation similarity s_i >= tau, so a
query about another relation of a stored subject passes whenever that relation text is
close enough to the stored one. v5 adds one test without new trainable parameters: the
relation text of the query is compared with the relation keys of ALL stored facts (any
subject), and the candidate is accepted only if s_i >= max_j cos(r(q), k_j) - m. Keys of
the same relation as the candidate have nearly the same text, so they do not block a
correct route; a query whose relation is stored for some other subject is closer to that
key and is rejected. The contrast set is the store itself (no external relation list).
Calibration: tau stays 0.6308; m is chosen on the SS66 dev split (dev store = the 1000
dev facts, their keys are the contrast set) to maximise Youden's J, ties to the larger m.
Evaluation is exact from the saved logs: v5 routes are a subset of v4 routes with the
same selected fact, so a still-routed query has its logged response and a newly
abstained query receives the base answer; base answers for newly abstained paraphrases
are generated with the base model only (no adapters). Readings: CF/zsRE route-correct
for probes and paraphrases, E2/E3 routed and leak under the four SS71 policies,
synthetic distractors, zsRE loc, sibling test, removal return, paraphrase accuracy per
arm, tiered store. Secondary: E3 leak split by whether the E3 relation occurs among the
stored facts' relations. Predictions: E3 routed <= 0.10 and adapter-arm E3 leak <= 0.06;
CF paraphrase route-correct >= 0.75; zsRE paraphrase route-correct >= 0.70; probes
route-correct >= 0.99; sibling test unchanged. SS71's decision rule is then applied to
the v5 readings.

### SS71 results (2026-10-11, `serving_locality.json`, wall 1307 s)

(a) CounterFact n=1000, existing adapters, router v4 (E3 routed 0.290, E2 routed 0.018):

| policy | E3 leak [95% CI] | E3 same | E2 leak | prediction |
| :-- | :-- | :-- | :-- | :-- |
| adapter | 0.186 [0.162, 0.209] | 0.711 | 0.014 | - |
| note | 0.140 [0.118, 0.161] | 0.720 | 0.013 | <= 0.05: **missed** |
| both | 0.237 [0.210, 0.262] | 0.710 | 0.016 | between note and adapter: **missed** (highest) |
| tiered | 0.155 [0.133, 0.176] | 0.716 | 0.014 | <= 0.06: **missed** |

(b) The 200 SS70 facts served with adapter and note: existing adapters efficacy 1.000,
paraphrase 0.805, E3 leak 0.200; local-anchor adapters 1.000 / 0.805 / 0.185 (paired
difference in E3 leak -0.015 [-0.045, 0.015]) [predictions: efficacy >= 0.98 met,
paraphrase >= 0.75 met, E3 leak <= 0.08 **missed**].

Readings: the note recovers the efficacy and paraphrase loss of the local-anchor adapters
completely, but it also carries the leak: an instructed note about one relation of a
subject is used by the model to answer other questions about that subject (14% of E3
prompts, about half of the routed ones). Combining note and adapter adds the two leak
sources. No policy meets the SS71 decision rule (E3 leak <= 0.10); the leak is a router
property (every leak occurs on a routed query), which SS72 addresses. The decision rule
is applied to the SS72 readings.

### SS72 results (2026-10-11, `router_v5_eval.json`)

Calibration: m = 0.3345 (Youden J 0.729 vs 0.725 for v4; dev TPR 0.835 vs 0.867, FPR
0.106 vs 0.143). Base answers generated for 42 CounterFact and 3 zsRE newly abstained
paraphrases.

| reading | v4 | v5 | prediction |
| :-- | :-- | :-- | :-- |
| CF probe route-correct / zsRE | 0.998 / 1.000 | 0.998 / 1.000 | >= 0.99: met |
| CF paraphrase route-correct | 0.814 | 0.772 | >= 0.75: met |
| zsRE paraphrase route-correct | 0.740 | 0.737 | >= 0.70: met |
| CF paraphrase adapter / note / both / tiered | 0.669 / 0.629 / 0.803 / 0.664 | 0.639 / 0.606 / 0.764 / 0.638 | - |
| E3 routed | 0.290 | 0.208 | <= 0.10: **missed** |
| E3 leak adapter / note / both / tiered | 0.186 / 0.140 / 0.237 / 0.155 | 0.143 / 0.111 / 0.177 / 0.125 | adapter <= 0.06: **missed** |
| E2 routed / adapter leak | 0.018 / 0.014 | 0.018 / 0.014 | - |
| distractors, zsRE loc routed | 0 / 0 | 0 / 0 | - |
| sibling test original / sibling | 0.998 / 0.995 | 0.998 / 0.995 | unchanged: met |
| removal return CF / zsRE | 1.00 / 0.86 | 1.00 / 0.87 | - |

E3 relation occurs among the stored relations for 97.5% of the prompts; the 2.5% with an
unstored relation do not leak. Readings: the store-contrast test removes 28% of the E3
routes at a cost of 4 points of CounterFact paraphrase accuracy; the calibrated margin is
large (0.33) because the MiniLM similarities of different CounterFact relations overlap
with those of paraphrases of the same relation. The sentence encoder, not the routing
rule, limits relation discrimination. SS71 decision rule: still no policy with E3 leak
<= 0.10.

## 73. Router v6: the frozen base model verifies relevance (pinned 2026-10-11, before the run)

A route of v4 is accepted only if the frozen base model, asked whether the selected
fact's statement answers the question, prefers "Yes": score = logit(Yes) - logit(No) at
the first answer position for the prompt (system "You check whether a stored fact
answers a question.", user "Fact: <statement>\nQuestion: <clause of the query that
contains the subject>\nDoes the fact answer the question? Answer Yes or No."). No
trainable parameters; one forward pass per routed query. Threshold theta: Youden's J on
the SS66 dev split over v4-routed queries (positives: paraphrases routed to their own
fact, negatives: other-relation prompts routed); everything v4 does not route stays
unrouted. Evaluation exact from the SS67/SS71 logs as in SS72 (v6 routes are a subset of
v4 routes; newly abstained probes and paraphrases receive generated base answers).
Readings as SS72, plus v6 on the zsRE loc questions and the removal probes. Predictions:
E3 routed <= 0.10, adapter-arm E3 leak <= 0.06, note-arm E3 leak <= 0.05; CF paraphrase
route-correct >= 0.75; probe route-correct >= 0.99; zsRE paraphrase route-correct >= 0.70.
SS71's decision rule is applied to the v6 readings.

### SS73 results (2026-10-11, `router_v6_eval.json`)

Calibration: theta = -7.25 (Youden J 0.729; dev TPR 0.865 vs v4 0.867, FPR 0.136 vs
0.143). The judge separates the routed dev queries (median score 4.1 for correct
paraphrases, -1.4 for other-relation prompts; 5th-95th percentiles -4.3..8.3 and
-7.2..4.2), but Youden's J on the pooled split, where 1500 positives face only 143 routed
negatives out of 1000, puts theta below almost every score.

| reading | v4 | v6 | prediction |
| :-- | :-- | :-- | :-- |
| CF probe / paraphrase route-correct | 0.998 / 0.814 | 0.997 / 0.809 | met / met |
| CF paraphrase adapter / both | 0.669 / 0.803 | 0.665 / 0.798 | - |
| E3 routed | 0.290 | 0.281 | <= 0.10: **missed** |
| E3 leak adapter / note / both / tiered | 0.186 / 0.140 / 0.237 / 0.155 | 0.183 / 0.138 / 0.232 / 0.153 | **missed** |
| E2 routed | 0.018 | 0.015 | - |
| zsRE paraphrase route-correct | 0.740 | 0.740 | met |

Readings: with the pinned calibration rule v6 is v4. The rule, not only the score, is
the problem: Youden's J weighs a missed paraphrase and a wrong route equally, while SS69
shows that a wrong route destroys 28% of correct answers and SS71 that it leaks the
stored answer in about half of the cases.

## 74. Relevance scores compared on the dev split, and operating points (pinned 2026-10-11, before the run)

(a) Dev-only selection. On the SS66 dev split, for the queries that v4 routes (positives:
paraphrases routed to their own fact; negatives: other-relation prompts), compare the
ROC AUC of three relevance scores: s_rel (v4 MiniLM relation similarity), s_judge (SS73)
and s_type = log p_base(" " + answer | query clause) - log p_base(" " + answer | edit
prompt), plain completion without chat template, summed over answer tokens (does the
stored answer fit the question as well as it fits the stored edit prompt). AUC per
dataset part and pooled; the score with the highest pooled AUC is selected. No test data
is touched in (a).
(b) Operating points for the selected score, both fixed now: (i) Youden's J on the pooled
dev split (the SS66 rule); (ii) the lowest threshold whose FPR on the CounterFact dev
negatives is <= 0.05 (the cost argument of SS69/SS71). Test readings as SS72/73 for both
operating points, computed exactly from the logs; plus the test trade-off curve
(CF paraphrase accuracy with adapter and note vs E3 leak) over thresholds, reported as a
curve with the two dev-chosen points marked.
Predictions: s_type has the highest AUC (it tests the answer's type, which is what a
leak emits); at (ii) E3 routed <= 0.10 and both-arm E3 leak <= 0.08 with CF paraphrase
(both) >= 0.70. SS71's decision rule is applied to the selected operating point.

### SS74 results (2026-10-11, `relevance_eval.json`)

(a) Dev AUC over v4-routed queries (CF part / zsRE part / pooled): s_rel 0.790 / 0.949 /
0.848; s_judge 0.775 / 0.938 / 0.827; s_type 0.649 / 0.632 / 0.646. Selected: s_rel
[prediction s_type highest: **missed**; s_type is the weakest]. The MiniLM relation
similarity is already the best of the three relevance scores; the base-model judge adds
nothing and the answer-type score is close to chance on CounterFact.

(b) Operating points for s_rel: (i) Youden theta = 0.6603 (dev TPR 0.835, FPR 0.110; J is
flat between 0.631 and 0.660); (ii) CF dev FPR <= 0.05: theta = 0.757 (dev TPR 0.677).

| reading (test) | v4 (tau 0.631) | (i) 0.660 | (ii) 0.757 | prediction for (ii) |
| :-- | :-- | :-- | :-- | :-- |
| CF edit prompt, both | 0.996 | 0.996 | 0.996 | - |
| CF paraphrase route-correct | 0.814 | 0.776 | 0.595 | - |
| CF paraphrase adapter / note / both / tiered | 0.669 / 0.629 / 0.803 / 0.664 | 0.643 / 0.607 / 0.768 / 0.642 | 0.513 / 0.488 / 0.589 [0.559, 0.618] / 0.512 | both >= 0.70: **missed** |
| zsRE paraphrase adapter / both | 0.959 / 0.979 | 0.955 / 0.975 | 0.938 / 0.956 | - |
| E3 routed | 0.290 | 0.207 | 0.064 | <= 0.10: met |
| E3 leak adapter / note / both / tiered | 0.186 / 0.140 / 0.237 / 0.155 | 0.140 / 0.109 / 0.171 / 0.121 | 0.058 / 0.045 / 0.063 [0.048, 0.078] / 0.053 | both <= 0.08: met |
| E2 routed / both leak | 0.018 / 0.016 | 0.017 / 0.015 | 0.012 / 0.011 | - |

Trade-off curve (test, CF): leak and paraphrase accuracy fall together along the
threshold, e.g. theta 0.699: paraphrase (both) 0.705, E3 leak (both) 0.116; 0.719: 0.661 /
0.092; 0.757: 0.589 / 0.063; 0.806: 0.506 / 0.036. Above theta ~0.72 nearly every routed
E3 prompt leaks: the survivors ask a near-identical question. Post-hoc reading (labelled
as such): 1.3% of the E3 prompts use a template that CounterFact also uses for the stored
fact's own relation (e.g. "X is located in" for continent and country); all 13 leak, and
they are 11 of the 63 leaks at (ii); other survivors are near-synonymous templates ("The
location of X is" / "X is located in", "produced by" / "a product developed by").
166 of the 186 adapter-arm leaks under v4 contain the routed fact's own answer.

SS71 decision rule at the selected operating point (ii): every policy has E3 leak <= 0.10;
the highest CF paraphrase accuracy is adapter and note (0.589). Default configuration for
the paper: router v4 with theta 0.757, adapter and note served together; the Youden point
and the curve are reported as the recall-oriented alternative.

## 75. Relation scorers and generated keys (pinned 2026-10-11, before the run)

SS72-74 locate the remaining leakage in the router's relation comparison; the threshold
moves along one trade-off curve. Two ways to move the curve itself, both without
trainable parameters in the store:
(1) the relation scorer, compared on the subject-masked relation texts of query clause
and key: MiniLM multilingual (current), sentence-transformers/all-mpnet-base-v2,
BAAI/bge-large-en-v1.5, intfloat/e5-large-v2 ("query: " prefix on both sides), all
frozen bi-encoders with cosine similarity, and the frozen cross-encoder
cross-encoder/quora-roberta-large (duplicate-question score on the pair);
(2) generated keys: at write time the frozen base model rewrites the edit prompt in three
other ways that keep the subject (greedy; prompt fixed in the script); lines without the
subject are dropped; the key score is the maximum over the original and generated keys.
10 configurations (5 scorers x {no, generated keys}). Selection on the SS66 dev split
only: the configuration with the highest dev TPR (paraphrases routed to their own fact)
at its threshold theta = the lowest value with CounterFact dev FPR <= 0.05 (the SS74
default rule); dev AUC and Youden J reported. Test routing readings (CPU) are reported
for all 10 configurations; the selected one at its dev theta is evaluated end to end on
both N=1000 stores (eval-only, existing adapters): serving arms, tiered store, E2/E3
under the four SS71 policies, zsRE loc, removal, sibling test.
Predictions: the selected configuration beats MiniLM's dev TPR at FPR 0.05 (0.677) by
>= 0.05; on the test set, CF paraphrase (adapter and note) >= 0.65 with E3 leak (both)
<= 0.07 (SS74 default: 0.589 / 0.063). Decision rule: if both test predictions hold, the
selected configuration replaces the SS74 default; otherwise the SS74 default stays and
the result is reported.

### SS75 (a) results (2026-10-11, `router_v7_select.json`, `generated_keys.json`)

Generated keys: 3000 edit prompts, mean 2.83 kept rewrites, 97 prompts with none; some
rewrites drift (a year instead of a place, invented details). Dev TPR at the threshold
with CF dev FPR <= 0.05, and test routing at that threshold:

| config | theta | dev TPR | CF paraphrase route-correct | E3 routed | E2 routed | zsRE paraphrase route-correct |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| minilm (SS74 default) | 0.757 | 0.677 | 0.595 | 0.064 | 0.012 | 0.694 |
| minilm+keys | 0.775 | 0.707 | 0.628 | 0.067 | 0.011 | 0.714 |
| mpnet | 0.669 | 0.692 | 0.607 | 0.052 | 0.011 | 0.714 |
| **mpnet+keys (selected)** | 0.693 | 0.716 | 0.623 | 0.057 | 0.010 | 0.721 |
| bge | 0.838 | 0.668 | 0.583 | 0.059 | 0.006 | 0.678 |
| bge+keys | 0.838 | 0.703 | 0.629 | 0.060 | 0.006 | 0.693 |
| e5 | 0.909 | 0.676 | 0.568 | 0.064 | 0.008 | 0.729 |
| e5+keys | 0.911 | 0.704 | 0.597 | 0.068 | 0.008 | 0.735 |
| quora_ce | 0.355 | 0.567 | 0.583 | 0.061 | 0.012 | 0.582 |
| quora_ce+keys | 0.623 | 0.585 | 0.561 | 0.061 | 0.010 | 0.630 |

Prediction "selected beats MiniLM dev TPR by >= 0.05": **missed** (+0.039). Generated
keys help every bi-encoder by 0.03-0.035 dev TPR; the encoder choice matters less
(0.668-0.692); the duplicate-question cross-encoder is worst. (b) the end-to-end
evaluation of mpnet+keys at theta 0.693 is pending (GPU paused on request).

### SS75 (b) results (2026-10-11, `{cf,zsre}_n1000_v7_eval.json`, walls 1628 s / 1670 s)

Router v7 = all-mpnet-base-v2 with generated keys at theta 0.693, existing adapters.

| reading | SS74 default (MiniLM, 0.757) | v7 (mpnet+keys, 0.693) | prediction |
| :-- | :-- | :-- | :-- |
| CF edit prompt, both | 0.996 | 0.996 | - |
| CF paraphrase route-correct | 0.595 | 0.623 | - |
| CF paraphrase adapter / note / both / tiered | 0.513 / 0.488 / 0.589 / 0.512 | 0.530 / 0.507 / 0.620 [0.591, 0.649] / 0.525 | both >= 0.65: **missed** |
| E3 routed | 0.064 | 0.057 | - |
| E3 leak adapter / note / both / tiered | 0.058 / 0.045 / 0.063 / 0.053 | 0.050 / 0.038 / 0.050 [0.037, 0.063] / 0.043 | both <= 0.07: met |
| E2 routed / both leak | 0.012 / 0.011 | 0.010 / 0.009 | - |
| CF distractors routed; removal return / retention | 0 ; >= 1.00 / - | 0 ; 1.00 / 0.997 | - |
| zsRE paraphrase route-correct / adapter / both | 0.694 / 0.938 / 0.956 | 0.721 / 0.947 / 0.960 | - |
| zsRE edit prompt both; loc routed | 0.989 ; 0 | 0.989 ; 0 | - |
| zsRE removal return / by content | >= 0.86 / 1.00 | 0.88 / 1.00 | - |

Readings: v7 is better than the SS74 default on both axes (CF paraphrase +3.1 points with
adapter and note, E3 leak -1.3 points), i.e. it moves the trade-off curve rather than
moving along it, but by less than the pinned bar. Decision rule: the paraphrase
prediction failed, so the SS74 default (MiniLM, theta 0.757) remains the paper's default
configuration; v7 is reported as the best measured configuration. The encoder accounts
for little of the gain; generated keys account for most of it (SS75a).

## 76. A relation encoder trained offline (pinned 2026-10-11, before the run)

SS75 shows that frozen general-purpose encoders differ little in separating relations.
Here the encoder is trained once, offline, for exactly this comparison, and then frozen;
the store still has no trainable state, so Propositions 1-3 are unaffected (keys are
fixed once computed). This is a change of concept: the router now contains a learned
component, trained on data disjoint from both the dev split and the test stores.
Training split (seed 76): CounterFact and zsRE records whose subjects occur in neither
the test stores nor the SS66 dev split; up to 3000 CounterFact and 1500 zsRE facts.
Held-out relations: 11 of the 34 CounterFact relation ids, drawn with seed 76, never
appear in training (as anchor, positive or negative), so generalization to unseen
relations is measured. Triplets on subject-masked relation texts (the router's own
processing): anchor = edit prompt; positive = each paraphrase prompt that contains the
subject (CounterFact: both paraphrase prompts; zsRE: the rephrased question);
hard negative = a template of another training relation. Model: all-mpnet-base-v2,
multiple-negatives ranking loss with the hard negatives (scale 20), batch 32, lr 2e-5,
1 epoch, AdamW, seed 76. Configurations: trained encoder with and without the SS75
generated keys; theta and selection exactly as SS75 (lowest theta with CF dev FPR <= 0.05,
highest dev TPR). The selected configuration is evaluated end to end on both stores
(eval-only, v7_eval) as in SS75(b). Additional reading: E3 routed and leak split by
whether the stored relation and the E3 relation are training relations.
Predictions: dev TPR at the threshold >= 0.80 (SS75 best 0.716); test CF paraphrase
(adapter and note) >= 0.70 with E3 leak (both) <= 0.06; on E3 prompts whose relations
were both held out, the routed fraction is higher than on seen relations.
Decision rule: if both test predictions hold, the trained encoder becomes the paper's
default; otherwise the SS74 default stays and the result is reported.

### SS76 results (2026-10-11, `relation_encoder_mpnet/`, `router_v8_select.json`, `{cf,zsre}_n1000_v8_eval.json`, `relenc_split.json`)

Training: 7500 triplets (3000 CF facts, 1500 zsRE facts), 235 steps, 45 s; loss 2.24 ->
1.24. Held-out relations: P103, P131, P136, P140, P176, P364, P37, P449, P463, P641, P740.
Selection: trained encoder without generated keys (dev TPR 0.936 at theta 0.750; with
keys 0.923) [prediction dev TPR >= 0.80: met].

| reading | SS74 default (MiniLM 0.757) | trained encoder (0.750) | prediction |
| :-- | :-- | :-- | :-- |
| CF edit prompt, both | 0.996 | 0.996 | - |
| CF paraphrase route-correct | 0.595 | 0.908 | - |
| CF paraphrase adapter / note / both / tiered | 0.513 / 0.488 / 0.589 / 0.512 | 0.730 / 0.684 / 0.891 [0.872, 0.910] / 0.723 | both >= 0.70: met |
| E3 routed | 0.064 | 0.044 | - |
| E3 leak adapter / note / both / tiered | 0.058 / 0.045 / 0.063 / 0.053 | 0.042 / 0.031 / 0.043 [0.031, 0.056] / 0.036 | both <= 0.06: met |
| E2 routed / both leak | 0.012 / 0.011 | 0.018 / 0.017 | - |
| CF distractors routed; removal return / retention | 0 | 0 ; 1.00 / 0.997 | - |
| zsRE paraphrase route-correct / adapter / both | 0.694 / 0.938 / 0.956 | 0.745 / 0.967 / 0.982 [0.973, 0.990] | - |
| zsRE loc routed; removal return / by content | 0 | 0 ; 0.84 / 1.00 | - |

Split by relation (CF test; seen = training relation):

| reading | default | trained |
| :-- | :-- | :-- |
| paraphrase route-correct, seen relations (n=652) | 0.575 | 0.994 (both arm 0.972) |
| paraphrase route-correct, held-out relations (n=348) | 0.632 | 0.747 (both arm 0.739) |
| E3 leak (both), stored and E3 relation seen (n=441) | 0.059 | 0.023 |
| E3 leak, stored relation held out (n=248) | 0.057 | 0.057 |
| E3 leak, E3 relation held out (n=211) | 0.095 | 0.081 |
| E3 leak, both held out (n=100) | 0.030 | 0.020 |

Prediction "both-held-out E3 routed higher than seen": **missed** (0.020 vs 0.025; n=100).
Readings: the trained encoder breaks the threshold trade-off on CounterFact, but most of
the gain is on relations it was trained on, where it has learned CounterFact's template
inventory: paraphrases of seen relations are routed almost perfectly and leakage falls
2.6x. On held-out relations paraphrase routing still improves (0.632 -> 0.747), while
leakage does not fall. The result is therefore a statement about a deployment whose
relation inventory is known in advance, with partial transfer of recall and none of
locality to new relations. Decision rule: both test predictions hold, so the trained
encoder (router v8) becomes the paper's default; the held-out split is reported with it.
E2 routing rises slightly (0.012 -> 0.018).

## 77. A relation encoder trained on a broad relation inventory (pinned 2026-10-11, before the run)

SS76 transfers recall but not locality to the 11 held-out CounterFact relations; with 23
training relations the encoder learns CounterFact's templates. Here the training
relations come from Wikidata: all properties whose values are items (fetched once by
SPARQL: id, English label, description), minus the 11 SS76 held-out property ids (exact,
since CounterFact relation ids are Wikidata property ids). For each property the frozen
base model writes four incomplete sentences or questions about an entity "X" that ask
for that property (greedy, prompt fixed in the script; lines without "X" dropped).
Triplets: anchor and positive = two templates of the same property; hard negative = the
most similar template of another property under the untrained all-mpnet-base-v2 (mined
once). Two encoders, trained exactly as SS76 (all-mpnet-base-v2, MNRL scale 20, batch
32, lr 2e-5, one epoch, seed 77):
  A = Wikidata triplets with all 34 CounterFact relations also removed (pure transfer:
      every CounterFact relation is unseen);
  B = SS76 triplets + Wikidata triplets without the 11 held-out properties.
Selection and threshold as SS75/76 (CF dev FPR <= 0.05, highest dev TPR); test routing
for both; end-to-end evaluation (v7_eval) of the selected one; relation split as SS76.
Predictions for the selected encoder: held-out-relation paraphrase route-correct >= 0.80
(SS76 0.747); E3 leak (both) pooled over prompts with a held-out stored or query relation
<= 0.04 (SS76 0.059); overall CF paraphrase (both) >= 0.88 and E3 leak (both) <= 0.045.
For A: CF paraphrase route-correct higher than the frozen mpnet (SS75: 0.607).
Decision rule: the selected encoder replaces the SS76 default only if all four
predictions for it hold.

### SS77 results (2026-10-11, `wikidata/`, `relation_encoder_wiki_{A,B}/`, `router_v9_select.json`, `{cf,zsre}_n1000_v9_eval.json`, `relenc_split_v9.json`)

Wikidata: 1785 item-valued properties, 3.89 templates each (1747 with >= 2). A: 1714
properties, 6807 triplets, 213 steps (63 s). B: 7500 SS76 + 6895 Wikidata triplets (1736
properties), 450 steps (128 s). Selection: B without keys (dev TPR 0.917 at theta 0.741;
A 0.744 at 0.689).

| reading | SS76 default | B (selected) | prediction for B |
| :-- | :-- | :-- | :-- |
| CF paraphrase route-correct, held-out relations | 0.747 | 0.750 | >= 0.80: **missed** |
| E3 leak (both), prompts with a held-out relation (pooled, n=559) | 0.059 | 0.059 | <= 0.04: **missed** |
| CF paraphrase (both) | 0.891 | 0.890 [0.870, 0.909] | >= 0.88: met |
| E3 leak (both) | 0.043 | 0.047 [0.034, 0.060] | <= 0.045: **missed** |
| CF paraphrase route-correct, seen relations | 0.994 | 0.994 | - |
| E3 leak (both), both relations seen | 0.023 | 0.032 | - |
| zsRE paraphrase (both); removal return / by content | 0.982; 0.84 / 1.00 | 0.957; 0.86 / 1.00 | - |

A (pure transfer, routing only, CF test): paraphrase route-correct 0.648 vs frozen
all-mpnet-base-v2 0.607 [prediction "higher": met]; split: seen-by-SS76 relations 0.655
vs 0.587, held-out relations 0.635 vs 0.644; E3 routed 0.045 vs 0.052.
Post-hoc reading (labelled): of the 33 leaking E3 outputs under B that involve a held-out
relation, 20 come from pairs of closely related relations (P495/P740 country of origin vs
location of formation 7; P364/P407 original language vs language of work 4; P103 with
P407/P37/P1412 native vs other languages 4; P176/P178 manufacturer vs developer 3;
P276/P131 location vs located in the administrative entity 2).
Readings: a broad relation inventory does not transfer to the held-out CounterFact
relations; training on 1714 Wikidata relations transfers recall to the 23 other
CounterFact relations but not to the 11 held-out ones, which are mostly near neighbours of
training relations. Where two relations differ in nuance, the boundary has to be learned
from examples of that pair. Decision rule: three of four predictions failed; the SS76
encoder stays the default.

## 78. Write-time negative keys (pinned 2026-10-11, before the run)

SS76/77: the remaining leakage on new relations comes from near-neighbour relations,
whose boundary an encoder does not learn from other relations. Instead of retraining the
encoder, each fact gets negative keys at write time: the frozen base model is shown the
edit prompt and writes five questions about the same subject that ask for different but
related information (greedy; prompt fixed in gen_negkeys.py; lines without the subject
dropped; relation texts masked as for keys). Router v10 = router v8 (SS76 encoder) with
one extra test: the selected candidate is accepted only if s_pos >= tau and
s_pos - max(similarity to its negative keys) >= m. The encoder stays the SS76 one, so the
11 held-out relations stay unseen by the encoder; the negatives are generated per fact
from the edit prompt only (no test prompt, no relation id). Calibration on the SS66 dev
split (negative keys generated for the dev facts the same way): (tau, m) over a grid
(tau: 200 quantiles of the dev candidate scores; m: -inf and -0.30..0.30 in steps of
0.02), choosing the highest pooled dev TPR with CounterFact dev FPR <= 0.05, ties to the
larger m. m = -inf reproduces SS76. Readings: test routing for CF and zsRE, the SS76
relation split, and the end-to-end evaluation (v7_eval) of v10.
Predictions: E3 leak (both) pooled over prompts with a held-out relation <= 0.04 (SS76
0.059) with held-out paraphrase route-correct >= 0.72 (SS76 0.747); overall CF paraphrase
(both) >= 0.88 and E3 leak (both) <= 0.035. Decision rule: v10 becomes the default only
if all four hold.

### SS78 results (2026-10-11, `generated_negkeys.json`, `router_v10_select.json`)

Negative keys: 3000 edit prompts, mean 4.82 kept, 31 with none. Some "negatives" are
paraphrases of the stored question ("What was the original language used in X?"), most
ask for unrelated attributes (year, founders, notable people).
Calibration: tau 0.747, m = -0.20, dev TPR 0.942 at CF dev FPR 0.05; without negatives the
same grid gives the same TPR (0.942 at tau 0.747), so m = -0.20 was chosen only by the
tie rule and is inactive in practice. Dev diagnostic (CF part, routes above tau): margin
s_pos - s_neg for correct paraphrases median 0.295 (5th-75th percentile 0.081-0.417),
for the 25 other-relation prompts that pass median 0.25 (0.037-0.60): the generated
negatives do not cover the confusable neighbour relation.
Test routing (CF): paraphrase route-correct 0.922 (held-out relations 0.787, seen 0.994),
E3 routed 0.046 (any held-out relation 0.059, both seen 0.030), E2 0.019; zsRE paraphrase
0.744, loc 0.000. The held-out E3 routed fraction equals SS76's (0.059); the held-out
paraphrase gain (0.747 -> 0.787) comes from the lower grid threshold (0.747 vs 0.750),
not from the negatives.
Predictions: held-out E3 leak <= 0.04 cannot hold (routed 0.059 and, as in SS76, every
routed held-out prompt leaks in all groups) -> **missed**; the decision rule cannot be
met. Deviation from the pin (recorded): the end-to-end evaluation of v10 was not run,
since its routes differ from v8 only by the threshold. The SS76 default stays.
Reading: write-time negatives from the base model do not supply the near-neighbour
contrast; the model does not know which neighbouring relation a query will confuse.

## 79. Template-disjoint split for the relation encoder (pinned 2026-10-12, before the run)

Motivation (external review, 2026-10-12, checked against `results/`). The router masks the
subject, so the SS76 subject-disjoint split does not separate the relation texts the
encoder sees. CounterFact has 218 distinct edit templates (243 counted per relation; 19
strings belong to more than one relation). For the SS76 encoder, every seen-relation test
key and 95.6% of seen-relation test paraphrase relation texts (after clause scoping and
masking) occur verbatim in its training triplets, as do all E3 queries whose two relations
are seen; 18 of the 66 held-out-relation templates are also templates of training relations
with another label. The SS76 seen-relation figures are therefore in-sample in CounterFact's
template space. The SS66 dev split, on which every threshold since SS66 was calibrated,
draws from the same strings, so the dev FPR of 0.05 holds only inside that space, for the
frozen encoders as well.

Strings. The unit of the split is the subject-masked relation text produced by the
router (`key_text` / `relation_text`), collected over all CounterFact records: edit
prompts and paraphrase prompts (E3 queries are edit templates and are included). 2767
strings. A string that occurs under more than one relation id is ambiguous (46): it is
never used for training or calibration. Strings of the 11 SS76 held-out relations (seed
76, unchanged) are held. The remaining strings of the 23 seen relations are split at
random into S_train and S_test, half each. Three split seeds: 79, 80, 81; for each seed
the dev split, the training set, the encoder and the test store below are rebuilt.

Training. As SS76 (all-mpnet-base-v2, multiple-negatives ranking loss with hard negatives,
scale 20, batch 32, lr 2e-5, one epoch, AdamW, training seed = split seed; CPU allowed),
with every triplet string required to lie in S_train: anchor = edit prompt with key in
S_train; positive = paraphrase with relation text in S_train; hard negative = an S_train
edit template of another seen relation. Up to 3000 CounterFact and 1500 zsRE facts; a zsRE
item is dropped if its probe or paraphrase relation text is in S_test, held or ambiguous.
Subjects of the dev split and of the test store are excluded.

Calibration. A dev split built like SS66 (500 CF + 500 zsRE facts), restricted to CF facts
whose key is in S_train, keeping only paraphrases in S_train (a fact without one is
skipped), with negatives drawn from S_train edit templates of other seen relations. For
each encoder separately, theta = the lowest value with CF dev FPR <= 0.05 (SS74 rule;
FPR = share of other-relation negatives routed). The frozen baseline is recalibrated the
same way, so trained and frozen thresholds are set on the same strings.

Test store (routing only, CPU, no adapters). The fixed N=1000 store is too small for this
split: its (key, paraphrase) pairs fall 148 in (train, train), 87 in (test, test), 233 in
(held, held), the rest mixed or ambiguous. A new routing store is drawn per seed from
CounterFact records whose subjects are disjoint from training and dev, 400 facts in each
group. For every fact the key, the evaluated paraphrase and the E3 query all lie on the
group's side:
  A  seen relation, seen strings: key, paraphrase and E3 query in S_train;
  B  seen relation, unseen strings: key, paraphrase and E3 query in S_test;
  C  held relation: key, paraphrase and E3 query held.
The evaluated paraphrase is the first paraphrase of the record on the group's side; the E3
query is an edit template of another relation from the group's side (A, B: another seen
relation; C: another held relation), drawn with the split seed and filled with the fact's
subject. Pools of records with key and a paraphrase on the side (E3 templates are drawn,
so they do not restrict the pool): seed 79 A 5034, B 3521, C 4910; seed 80 4489 / 3552 /
4910; seed 81 4758 / 3289 / 4910; at least 2869 distinct subjects per group.

Encoders. (1) the SS79 trained encoder of the seed; (2) frozen all-mpnet-base-v2 (its
base); (3) the SS76 encoder as a contaminated reference (it was trained on strings of all
groups except C). Each at its own SS79 dev theta of the seed. Generated keys are not used
(they exist only for the old store and need the GPU); in SS75 they added 0.016
route-correct to the frozen mpnet on the old store. The router is v4/v7 with the
tie-breaks of commits 2862294 and 45c000a (on the old stores they change no route).

Primary metric: paraphrase route-correct at the dev theta. Gain per group:
Delta_g = route-correct(SS79 trained) - route-correct(frozen mpnet), on the same facts.
Second reading: E3 FPR_g = share of the group's E3 queries routed at the dev theta.
Leakage is not measured (no GPU). E3 FPR is its proxy and an upper bound: the share of
routed E3 prompts that leaked rises with the threshold, 0.64 (v4, tau 0.631, adapter arm,
n = 290 routed), 0.88 (v7, theta 0.693, adapter and note, n = 57), 0.98 (v8, theta 0.750,
adapter and note, n = 44).
Statistics: predictions are evaluated on the mean over the three seeds; the per-seed
values and their range are reported. Within a seed, paired bootstrap 95% CIs (10000
resamples over facts) for differences between encoders. With n = 400 per group the CI
half-width of a single rate near 0.7 is about 0.045.

Provenance of the predictions. The thresholds below were proposed by the external reviewer
and adopted by the author after looking at SS76 (seen relations: trained minus frozen
mpnet route-correct about +0.41; held-out relations about +0.10; E3 routed on held-out
relation groups 0.057-0.081 against a dev FPR of 0.05). No SS79 data had been produced or
seen when they were written. Reviewer's expectation: Delta_A about 0.35, Delta_B
0.15-0.20; E3 inflation +0.02 to +0.04 for the trained encoder and 0 +/- 0.02 for the
frozen one, so P3 may fail.

Predictions.
  P1 (primary, three outcomes; untestable, and reported as such, if Delta_A < 0.05):
     template memorisation    Delta_B <= 0.5 * Delta_A;
     generalisation           Delta_B >= 0.8 * Delta_A;
     partial                  otherwise.
     Predicted outcome: template memorisation.
  P3 (primary, contrast) Inflation of the E3 FPR on group B, I = E3 FPR_B - dev FPR (dev
     FPR = the realised CF dev FPR at the encoder's theta), is at least 0.02 larger for
     the SS79 trained encoder than for the frozen mpnet: I_trained - I_frozen >= 0.02,
     paired over the same E3 queries, pooled over the three seeds (n = 1200; standard error
     of a rate near 0.05 about 0.006, of the difference about 0.009). I_frozen is reported
     descriptively (expected |I_frozen| <= 0.03), not as a prediction.
  P2 (secondary) route-correct(SS79 trained, B) <= route-correct(frozen mpnet, B) + 0.05.
Decision rule.
  P1 template memorisation: the paper reports the SS76 seen-relation figures as in-sample
  in CounterFact's template space, states that relation separation outside that space
  remains open, and reports SS79 alongside; the encoder may remain the configured default,
  but no claim of generalisation within a relation is made.
  P1 generalisation: the encoder generalises to unseen phrasings of known relations, the
  SS76 seen-relation claim stands, and the paper says so with the SS79 group B figures.
  P1 partial: the paper reports Delta_A and Delta_B side by side and claims partial
  generalisation within a relation, with the share Delta_B / Delta_A.
  P3 holds: the paper states that the trained encoder's threshold keeps its dev FPR only
  within the template space it was calibrated on, more so than a frozen encoder. P3
  fails: the paper reports the inflation of both encoders without that claim.
  In every case the old split (relenc_split.json) is relabelled as a relation split only.

Deviation (2026-10-18, before any SS79 run). The router's subject ranking is changed from
(subject length, exact) to (length of the matched query text, exact): an approximate match
of a longer subject no longer displaces an exact match of the same words. On the N=1000
stores this changes 7 of the 6000 candidate sets used for the reported results, all on
CounterFact and all corrections: queries about Nokia Lumia 520 went to Nokia Lumia 1520
(edit prompt, paraphrase, E2, E3) and queries about Limca went to Loimaa (edit prompt,
paraphrase, E3). SS79 runs with the new ranking. The reported SS64 to SS78 figures were
computed with the old ranking and are not recomputed here.

## 80. Natural other-relation queries (DRAFT 2026-10-12, not yet pinned; not run before SS79 is reported)

E3 draws its queries from the same template pool the encoders train and calibrate on.
Here the queries are written by a model of another family than the base model (Qwen2.5),
so neither the generated keys (written by Qwen) nor the templates share their style.
Generator: Llama-3.2-3B-Instruct, else Gemma-2-2B-it (fits the shared 8 GB GPU; greedy;
prompt fixed in the script). For each stored fact the generator receives the subject and
the English label and description of a target relation R' != R and writes one natural
question about the subject asking for R'; questions without the subject are dropped.
R' is sampled with deliberate near-neighbour pairs, half of the queries: P495/P740,
P176/P178, P17/P30/P131, P364/P407, P103/P1412/P37, P276/P131, P19/P20/P937; the other
half uniform over the remaining relations.
Leak is redefined: a routed output leaks only if it contains the stored target and the
stored target is not a correct answer to the generated question under the counterfactual
(for example, a continent stored for a "which region" question is not a leak). A manual
correctness label on at least 100 routed outputs, sampled before any rates are computed,
estimates how often the automatic leak counts a legitimate answer; rates are reported
with and without the correction.
Encoders and thresholds: chosen after SS79 is reported (at least the SS74 default, the
SS76 encoder and the frozen mpnet). Predictions are pinned with them.

## 81. Adapter size (pinned 2026-10-19, before the run)

Motivation. A stored adapter is a rank-16 LoRA update of q_proj and v_proj in all 28
layers, saved in fp32: 2,179,072 parameters, 8.7 MB per fact and 8.8 GB for the N=1000
store, for a single fact. On Qwen2.5-1.5B a rank-r q/v update costs r x 4864 parameters
per layer (q 1536 -> 1536, v 1536 -> 256). Two independent levers are measured: the
precision the adapter is stored in (A, no training) and the form of the update (B).

A. Storage precision (eval only, no training). The first 200 facts of the CounterFact
N=1000 store (cf_n1000.ckpt, baseline recipe, verified against routed_eval_manifest.json).
Each stored A and B tensor is rounded to the storage format and cast back to fp32 before
it is written into the LoRA slot: fp32 (reference), bf16, fp16, int8 (symmetric, one
absmax scale per row of A and per row of B, scales stored in fp16). Size per fact: 8.7,
4.4, 4.4 and 2.3 MB. Each fact is served by its own adapter (no router; the router does
not read the adapter). Readings per format: efficacy (edit prompt), paraphrase (adapter
alone, "Answer briefly."), paraphrase with adapter and note (the instruct note system of
the default serving mode), held-out D on the 40 SS69 canaries (kappa = 0.968), and the
share of greedy outputs (edit prompt, paraphrase, paraphrase with note) identical to the
fp32 outputs.
Predictions: bf16 and fp16 give identical outputs for at least 97% of the generations,
every hit rate within 0.01 of fp32 and |delta D| < 0.01 per adapter; int8 gives identical
outputs for at least 90%, hit rates within 0.02 of fp32, no adapter crosses kappa.

B. Update form (training). The SS44 set (CounterFact N=50, external_counterfact_n50.json),
per-fact adapters, the adopted per-fact LoRA+ recipe (A lr 3e-4, B lr 2.4e-3, AdamW),
16 passes over the two pairs, KL anchor on the two training canaries against the base,
lora_alpha = 2r, B = 0 at init. torch.manual_seed(81) before each adapter is created, so
every adapter starts from the same A. Variants:
  r16_qv_all   rank 16, q+v, all 28 layers (reference)
  r4_qv_all, r2_qv_all, r1_qv_all
  r4_v_all, r1_v_all          v_proj only
  r4_qv_mid, r1_qv_mid        q+v, layers 8 to 15 only
  fa_r4_qv_all, fa_r1_qv_all  A frozen at its seeded init and shared by all facts (it is
                              regenerated from the seed and stored once); only B trained
                              and stored
Size per fact in bf16 (A + B, or B only for fa): 4.36 MB, 1.09 MB, 545 KB, 272 KB,
401 KB, 100 KB, 311 KB, 78 KB, 401 KB, 100 KB. No learning rate or step count is tuned
per variant; a variant that fails is recorded, not retuned.
Evaluation per variant on a fresh base with that variant's LoRA layout, each fact served
by its own adapter: efficacy, paraphrase (adapter alone), paraphrase with adapter and
note, held-out D (40 canaries) mean / max and share above kappa = 0.968, in-sample D (the
2 training canaries, as SS46), stored size, training seconds.
Acceptance rule (fixed now): efficacy >= 0.98, paraphrase (adapter alone) >= 0.86 (the
SS46 baseline), held-out D max <= kappa. The smallest variant that passes is the
candidate recipe; it replaces the stored recipe only after an N=1000 confirmation pinned
in a later section (store, router results, tiered store, removal), not on this N=50 run.
Predictions: r16_qv_all reproduces SS46 LoRA+ (efficacy 1.00, paraphrase 0.86 to 0.94);
r4_qv_all passes; r1_qv_all keeps efficacy >= 0.98 with paraphrase 0.80 to 0.90
(borderline); at equal rank, v-only and mid-layer variants keep efficacy but lose
paraphrase against q+v all layers; fa variants keep efficacy >= 0.96 and lose some
paraphrase against the trained-A variant of the same rank; no variant crosses kappa
(B = 0 at init and the KL anchor); the smallest passing variant is at most 300 KB in bf16
(at least 29x smaller than the stored 8.7 MB).

## 82. SS79 results: template-disjoint split (2026-10-19)

Run as pinned in SS79 with the deviation of 2026-10-18 (relenc_template_split.py; results
relenc_template_split_s{79,80,81}.json and relenc_template_split.json). The split
reproduces the pinned counts exactly (2767 strings, 46 ambiguous, 887 held, 917 + 917
seen; pools 5034 / 3521 / 4910, 4489 / 3552 / 4910, 4758 / 3289 / 4910). Per seed: dev
500 CF + 500 zsRE facts, store 400 facts per group, 5175 to 5408 triplets from 3000 CF and
1500 zsRE facts, one epoch on CPU (383 to 452 s; loss 1.9-2.1 -> 1.10-1.15).
Implementation choices not fixed by SS79: a CF fact counts toward the 3000 only if it
contributes at least one triplet; theta is used unrounded; the store has one fact per
subject, so route-correct is the share of paraphrases that reach their own fact.

Paraphrase route-correct (seeds 79 / 80 / 81) at each encoder's own dev theta (dev CF
FPR 0.048 to 0.050 for every encoder):

| group | SS79 trained | frozen mpnet | SS76 (contaminated) | Delta (trained - frozen) |
| :-- | :-- | :-- | :-- | :-- |
| A seen strings | 0.983 / 0.985 / 0.995 | 0.765 / 0.632 / 0.547 | 0.990 / 0.988 / 1.000 | 0.218 / 0.353 / 0.448, mean 0.339 |
| B unseen strings | 0.920 / 0.963 / 0.998 | 0.620 / 0.580 / 0.703 | 0.998 / 0.993 / 0.998 | 0.300 / 0.383 / 0.295, mean 0.326 |
| C held relations | 0.897 / 0.820 / 0.797 | 0.530 / 0.580 / 0.520 | 0.892 / 0.912 / 0.890 | 0.368 / 0.240 / 0.278, mean 0.295 |

Paired bootstrap 95% CIs of Delta exclude 0 in every group and seed (lowest bound 0.18).
E3 FPR (seeds 79 / 80 / 81): trained A 0.040 / 0.050 / 0.033, B 0.077 / 0.080 / 0.075,
C 0.110 / 0.085 / 0.083; frozen A 0.058 / 0.033 / 0.022, B 0.048 / 0.025 / 0.018,
C 0.030 / 0.037 / 0.035.

Predictions.
  P1 failed: template memorisation was predicted; the outcome is generalisation
     (Delta_B / Delta_A = 0.326 / 0.339 = 0.96 >= 0.8).
  P2 failed: route-correct on B is 0.960 for the trained encoder against 0.634 for the
     frozen one (+0.33, not <= +0.05).
  P3 holds: I_trained = +0.028, I_frozen = -0.020, I_trained - I_frozen = +0.048 (95% CI
     0.032 to 0.063, n = 1200 pooled E3 queries) >= 0.02.
Readings. The trained encoder's gain does not depend on having seen the phrasing: it is
as large on unseen phrasings of seen relations (B) as on seen ones (A), and nearly as
large on relations never trained on (C). The external review's concern (SS76 gains
in-sample in the template space) is answered in favour of the encoder for routing
accuracy. The cost is on the locality side: its threshold, calibrated on S_train
negatives, routes 7.5 to 11% of other-relation queries outside that space (B, C), where
the frozen encoder stays at or below its dev FPR.
Decision (SS79 rule). P1 generalisation: the SS76 seen-relation claim stands and the
paper reports it with the SS79 group B figures. P3 holds: the paper states that the
trained encoder's threshold keeps its dev FPR only within the template space it was
calibrated on, more so than a frozen encoder. The old split (relenc_split.json) is
relabelled as a relation split only. SS80 can now be pinned.

## 83. SS81 results: adapter size (2026-10-19)

Run as pinned (adapter_size.py; adapter_size_precision.json, adapter_size_sweep.json).

A. Storage precision, first 200 CounterFact N=1000 adapters (baseline recipe), each
served by its own adapter; the LoRA slot is fp32, so every format is cast back to fp32.

| format | size | identical outputs | efficacy | paraphrase | para. + note | held-out D mean / max | above kappa | max abs D change |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| fp32 | 8.7 MB | 1.000 | 1.000 | 0.835 | 0.985 | 0.371 / 1.448 | 4 | 0 |
| bf16 | 4.4 MB | 0.985 | 0.995 | 0.845 | 0.985 | 0.372 / 1.414 | 4 | 0.058 |
| fp16 | 4.4 MB | 0.978 | 0.995 | 0.835 | 0.980 | 0.371 / 1.452 | 4 | 0.054 |
| int8 | 2.3 MB | 0.978 | 0.995 | 0.845 | 0.985 | 0.371 / 1.436 | 5 | 0.041 |

Predictions: bf16 and fp16 identical outputs >= 97% holds (98.5%, 97.8%); hit rates
within 0.01 of fp32 holds; |delta D| < 0.01 per adapter fails (up to 0.058). int8
identical >= 90% holds (97.8%); hit rates within 0.02 holds; no adapter crossing kappa
fails (one adapter crosses that fp32 keeps below; 4 of 200 fp32 adapters are above kappa
already). Reading: int8 storage is lossless at the level of answers (the 13 of 600 changed
outputs do not change any hit rate by more than 0.01) and quarters the size; D moves by
up to 0.06, so an adapter near kappa must be checked in the stored format.

B. Update form, SS44 set (N=50), per-fact LoRA+ recipe, seed 81. Held-out D on the 40
SS69 canaries, kappa 0.968; in-sample D on the 2 training canaries.

| variant | size (bf16) | efficacy | paraphrase | para. + note | held-out D mean / max | above kappa | in-sample D max | pass |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| r16_qv_all | 4.36 MB | 1.00 | 0.90 | 0.94 | 1.248 / 3.829 | 0.70 | 0.049 | no (D) |
| r4_qv_all | 1.09 MB | 1.00 | 0.78 | 0.98 | 0.588 / 1.129 | 0.06 | 0.063 | no |
| r2_qv_all | 545 KB | 0.98 | 0.88 | 1.00 | 0.447 / 0.923 | 0 | 0.077 | yes |
| r1_qv_all | 272 KB | 0.98 | 0.72 | 1.00 | 0.266 / 0.657 | 0 | 0.092 | no (para.) |
| r4_v_all | 401 KB | 1.00 | 0.84 | 1.00 | 0.557 / 0.960 | 0 | 0.113 | no (para.) |
| r1_v_all | 100 KB | 0.94 | 0.62 | 1.00 | 0.186 / 0.414 | 0 | 0.078 | no |
| r4_qv_mid | 311 KB | 0.76 | 0.10 | 0.94 | 0.591 / 1.387 | 0.16 | 0.097 | no |
| r1_qv_mid | 78 KB | 0.24 | 0.04 | 0.90 | 0.292 / 0.722 | 0 | 0.075 | no |
| fa_r4_qv_all | 401 KB | 0.92 | 0.58 | 1.00 | 0.353 / 0.851 | 0 | 0.078 | no |
| fa_r1_qv_all | 100 KB | 0.34 | 0.12 | 0.94 | 0.058 / 0.144 | 0 | 0.055 | no |

The fa variants used the same A for every fact (checked). One fact is 0.02 of a rate.
Predictions: r16_qv_all reproduces SS46 on efficacy and paraphrase (1.00 / 0.90) holds;
r4_qv_all passes fails (paraphrase 0.78, D max 1.13); r1_qv_all efficacy >= 0.98 holds,
its paraphrase 0.80 to 0.90 fails (0.72); v-only loses paraphrase against q+v at equal
rank holds at r1 (0.62 / 0.72) and fails at r4 (0.84 / 0.78); mid-layer variants keep
efficacy fails (0.76, 0.24); fa variants keep efficacy >= 0.96 fails (0.92, 0.34); no
variant crosses kappa fails (r16 0.70, r4 0.06, r4_mid 0.16); smallest passing variant at
most 300 KB fails (545 KB).
Readings.
1. LoRA+ at rank 16 fails the held-out check: 70% of its adapters lie above kappa
   (held-out D mean 1.25), while its in-sample D is the lowest of all variants. SS46
   adopted LoRA+ on the in-sample D, the reading that SS69 showed to understate an
   adapter's effect; under the held-out check LoRA+ at rank 16 is not a free win but a
   rejected recipe. The stored N=1000 adapters use the baseline recipe (held-out D mean
   0.37 in A), so no reported store result changes; Table delta_recipes reports in-sample
   KL and must say so.
2. Held-out D falls with rank (q+v, all layers: 1.25, 0.59, 0.45, 0.27 for r = 16, 4, 2,
   1). A smaller update is also a safer one.
3. Restricting layers or freezing A costs efficacy within the 16-pass budget; these
   forms would need more steps, which SS81 does not tune.
4. Paraphrase alone is not monotonic in rank (r4 0.78, r2 0.88); with N=50 a difference
   of 0.10 is about two standard errors.
5. Served with the note (the default mode), paraphrase is 0.94 to 1.00 for 9 of 10
   variants, including the 100 KB r1_v_all. On this set the note alone answers many of
   these facts, so this does not show that a 100 KB adapter is enough for the facts that
   need one (the tiered store's adapter facts); it is not an SS81 criterion.
Decision (SS81 rule). r2_qv_all is the candidate recipe (545 KB in bf16, 16x smaller
than the stored 8.7 MB; about 280 KB if int8 storage holds for it as it does for rank
16). It does not replace the stored recipe before an N=1000 confirmation, to be pinned
next. That confirmation should also separate the recipe from the rank (rank 2 with the
baseline recipe against rank 2 with LoRA+) and read efficacy and paraphrase with the note
on the facts whose note fails alone. The per-fact adoption of LoRA+ (SS46) is withdrawn:
until the confirmation, per-fact adapters keep the baseline recipe.

## 84. Rank-2 adapters on the adapter facts of the N=1000 stores (pinned 2026-10-19, before the run)

Motivation. SS83 left r2_qv_all (545 KB in bf16) as the candidate recipe, measured on 50
facts, many of which the note answers alone. In the default tiered store only facts whose
note fails on the edit prompt carry an adapter (SS68 rule, tiers in cf/zsre_n1000_v4_eval):
162 CounterFact and 206 zsRE facts. These 368 facts are the population on which an
adapter recipe matters, and on which the comparison is run here.

Arms (per fact, trained from a fresh base on the store's pairs facts.pairs, 16 passes, KL
anchor on the two training canaries, lora_alpha = 2r, q_proj + v_proj in all layers,
torch.manual_seed(84) before each adapter):
  ref     the stored rank-16 baseline adapters (no training; verified checkpoints)
  r2_base rank 2, baseline recipe (AdamW, lr 3e-4 for A and B)
  r2_lp   rank 2, LoRA+ recipe (A 3e-4, B 2.4e-3)
and, eval only, r2_base and r2_lp stored in int8 (SS81 rounding, one fp16 scale per row;
about 280 KB per fact).
Readings per arm, each fact served by its own adapter (routing does not read the
adapter), greedy, up to 64 new tokens: efficacy (edit prompt) and paraphrase with the
adapter alone ("Answer briefly.") and with adapter and note (the instruct note system,
the default serving mode); held-out D on the 40 SS69 canaries, share above kappa = 0.968;
wrong-route retention: the 200 SS69 zsRE dev questions that the base answers correctly
(canary_eval.json), each served with the adapter of one of the 368 facts drawn with
seed 84 (the same draw for every arm); for the int8 arms also the share of outputs
identical to the arm in fp32. Paired bootstrap 95% CIs (10000 resamples) for arm - ref.
Readings are reported pooled and per dataset.

Decision rule (fixed now). A rank-2 recipe replaces the stored recipe for new adapters if,
pooled over the 368 facts: (i) efficacy with note >= ref - 0.02 and paraphrase with note
>= ref - 0.03, with the lower CI bound of paraphrase-with-note (arm - ref) >= -0.06;
(ii) at most 2% of its adapters above kappa; (iii) wrong-route retention >= ref - 0.03.
If both rank-2 recipes pass, the one with the higher paraphrase with note is chosen, and
r2_base on a tie within 0.01. If the chosen recipe's int8 arm has >= 97% identical
outputs and also passes (i) to (iii), adapters are stored in int8. If neither rank-2
recipe passes, the stored recipe stays and only int8 storage (SS83 A) is adopted.
The reported N=1000 store results are not recomputed by this entry; a store rebuilt
with the chosen recipe is a later entry.

Predictions.
  P1 r2_base passes the rule.
  P2 r2_lp has a higher mean held-out D than r2_base.
  P3 r2_base has a lower mean held-out D than ref.
  P4 adapter-alone paraphrase is at least as high for r2_lp as for r2_base.
  P5 both int8 arms have >= 97% identical outputs.
  P6 wrong-route retention of both rank-2 arms >= ref.

## 85. SS80 pinned: natural other-relation queries (pinned 2026-10-19, before the run)

This pins the SS80 draft after SS79 was reported (SS82), with these changes to the draft.
Generator: Llama-3.2-3B-Instruct and Gemma-2-2B-it are gated on Hugging Face and not
downloadable here without the author's account (HTTP 401), so the generator is
HuggingFaceTB/SmolLM3-3B (not Qwen, not gated, bf16 on the shared GPU), in non-reasoning
mode ("/no_think"), greedy, at most 48 new tokens; the first line of the output, stripped
of quotes, is the question. Prompt (fixed): system "/no_think You write exactly one
natural English question and nothing else."; user "Write one natural question about
{subject} that asks for its {label} ({description}). Mention {subject} by name. Reply
with the question only." The description is the Wikidata English description cut at its
first ';' or '. '.
Store and queries: the CounterFact N=1000 store (stored rank-16 adapters, SS68 tiers),
one query per fact about its subject. R' != R: 500 facts drawn with seed 80 from the 675
whose relation has a listed near neighbour get R' drawn from its neighbours (P495/P740,
P176/P178, P17/P30/P131, P364/P407, P103/P1412/P37, P276/P131, P19/P20/P937; P131's
neighbours are P17, P30, P276); the other 500 get R' drawn uniformly from the CounterFact
relations other than R and R's neighbours. Questions in which the subject is not found
(router v4 find_subject) are dropped and counted.
Routers (each at its existing dev threshold, chosen now): the configured default (SS76
relation encoder, theta 0.7501), the SS74 default (MiniLM, theta 0.757) and the frozen
all-mpnet-base-v2 (theta 0.6694, SS75 dev).
Readings per router, pooled and for the near-neighbour and uniform halves: routed share;
leak under the default serving mode (adapter and note) and under the tiered policy, with
the automatic rule: the response contains the queried fact's target and the base response
does not (the SS71 rule with any stored answer is also reported). Before any rate is
computed, 120 routed queries are sampled with seed 80 (40 per router, or all if fewer)
into a label file with question, stored fact, response and base response; the author
labels whether the stored target is a correct answer to the question under the
counterfactual. Corrected leak rates are computed after the labels exist, in a later
entry, as the draft specifies.
Predictions.
  P1 the default router routes at least 7% of the natural queries (its dev FPR 0.05 +
     0.02; SS79 inflation off the template space).
  P2 the default router routes a larger share of the natural queries than the frozen
     mpnet, paired over the same queries.
  P3 for the default router, near-neighbour queries are routed at least 0.05 more often
     than uniform ones.
  P4 the default router's automatic leak with adapter and note on natural queries is at
     least its E3 template leak (0.043, xCfEthreeLeakBoth).
  P5 (after labels) the correction removes less than half of the default router's
     automatic leaks.
Decision rule. P1 holds: the paper reports the default configuration's locality on
natural queries next to the template figure and states that 4.3% holds for CounterFact
templates. P1 fails: the paper states that the template E3 figure carries over to natural
queries written by another model family.

Deviation (2026-10-19, before any SS85 run). The pinned description rule cuts P20's
description inside "(e.g. city ...)", giving "most specific known (e.g". A '. ' that ends
"e.g." or "i.e." no longer cuts; this changes only P20 among the 34 relations.

## 86. SS84 results: rank-2 adapters on the adapter facts (2026-10-20)

Run as pinned (rank2_confirm.py; rank2_confirm.json, trained adapters in
rank2_confirm_r2_base.pt / rank2_confirm_r2_lp.pt). 368 adapter facts (162 CF, 206 zsRE);
training 2396 s (r2_base) and 2424 s (r2_lp), about 6.5 s per fact.

Pooled over the 368 facts (each fact served by its own adapter):

| arm | size per fact | efficacy | eff. + note | paraphrase | para. + note | held-out D mean / max | above kappa | wrong-route retained |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| ref (rank 16, fp32) | 8.7 MB | 0.997 | 0.997 | 0.924 | 0.981 | 0.476 / 1.647 | 0.046 | 0.635 |
| r2_base | 545 KB | 0.723 | 0.973 | 0.543 | 0.957 | 0.152 / 0.618 | 0 | 0.795 |
| r2_base int8 | ~280 KB | 0.720 | 0.976 | 0.543 | 0.959 | 0.151 / 0.585 | 0 | 0.795 |
| r2_lp | 545 KB | 0.997 | 0.986 | 0.924 | 0.978 | 0.484 / 1.905 | 0.060 | 0.670 |
| r2_lp int8 | ~280 KB | 0.997 | 0.984 | 0.927 | 0.984 | 0.484 / 1.930 | 0.060 | 0.670 |

Arm - ref, paired 95% CIs: r2_base efficacy with note -0.025 [-0.044, -0.008], paraphrase
with note -0.025 [-0.049, 0.000]; r2_lp efficacy with note -0.011 [-0.022, -0.003],
paraphrase with note -0.003 [-0.016, 0.008]. Identical outputs to fp32: r2_base int8
0.959, r2_lp int8 0.987. Per dataset: CF ref paraphrase alone 0.833, r2_base 0.284, r2_lp
0.840; above kappa CF ref 0.012, r2_lp 0.012; zsRE ref 0.073, r2_lp 0.097, r2_base 0.

Decision rule. r2_base fails (i): efficacy with note is 0.0245 below ref (limit 0.02);
(ii) and (iii) pass. r2_lp fails (ii): 6.0% of its adapters are above kappa (limit 2%);
(i) and (iii) pass. Neither rank-2 recipe passes, so the stored recipe stays and only
int8 storage (SS83 A) is adopted: 2.3 MB per adapter instead of 8.7 MB.
Predictions: P1 (r2_base passes) fails; P2 (r2_lp held-out D above r2_base) holds (0.484
against 0.152); P3 (r2_base below ref) holds (0.152 against 0.476); P4 (adapter-alone
paraphrase r2_lp >= r2_base) holds (0.924 against 0.543); P5 (both int8 arms >= 97%
identical) fails (r2_base int8 0.959); P6 (wrong-route retention of both rank-2 arms >=
ref) holds (0.795 and 0.670 against 0.635).
Readings.
1. At rank 2, LoRA+ reproduces the rank-16 adapters on every quality reading (efficacy,
   paraphrase alone and with the note within 0.011), at one eighth of the bf16 size and
   about 1/31 of the stored fp32 size in int8. Its held-out D is the same as ref's (0.48
   against 0.48). The rank-16 LoRA+ failure in SS83 (D mean 1.25) is therefore a property
   of the high B learning rate at rank 16, not of LoRA+ at low rank.
2. Criterion (ii) is not met by ref itself (4.6% above kappa): kappa is the 99th
   percentile of D over 100 CounterFact adapters (SS69), and zsRE adapters have a higher
   held-out D (ref 7.3% above kappa on zsRE, 1.2% on CF). The criterion was stricter than
   the stored recipe; this was not checked when SS84 was pinned. It is recorded, not
   changed after the fact.
3. r2_base is the safer adapter: no adapter above kappa, and a wrong route keeps 79.5% of
   correct base answers against 63.5% for ref. It does not learn the fact well alone
   (efficacy 0.72) and relies on the note (0.97 with it).
Follow-up (not run): a new entry on facts outside the N=1000 stores, with an
above-kappa criterion relative to the stored recipe fixed before the run, is needed
before r2_lp in int8 can replace the stored recipe; SS84 cannot be re-scored with a new
criterion.

## 87. Rank-2 LoRA+ in int8 on fresh adapter facts (pinned 2026-10-20, before the run)

Motivation. In SS84 rank-2 LoRA+ matched the stored rank-16 adapters on every quality
reading but failed an above-kappa limit (2%) that the stored adapters also fail (4.6%).
SS84 is not re-scored. This entry tests the candidate on facts it has not touched, with
the divergence criterion stated relative to the stored recipe, fixed before the run.

Facts. CounterFact and zsRE facts (external.py adapters) whose subjects are in neither
N=1000 store nor the SS66 dev split; one fact per subject; drawn with seed 87, 1000 per
dataset, in that order. Tier check as SS68: a fact is an adapter fact if the base model
with the instruct note does not produce the target for the edit prompt (greedy, up to 64
new tokens). All adapter facts of the 2000 are used (expected about 360).
Arms, trained per fact from a fresh base on facts.pairs, 16 passes, KL anchor on the two
training canaries, q_proj + v_proj in all layers, lora_alpha = 2r, torch.manual_seed(87)
before each adapter:
  ref16  rank 16, baseline recipe (lr 3e-4): the stored recipe
  r2_lp  rank 2, LoRA+ (A 3e-4, B 2.4e-3), evaluated as stored in int8 (SS81 rounding)
Readings as SS84 (efficacy and paraphrase alone and with the note, held-out D on the 40
SS69 canaries, share above kappa = 0.968, wrong-route retention on the 200 SS69 zsRE dev
questions with adapters drawn with seed 87 from the adapter facts, the same draw for
both arms), pooled and per dataset; paired bootstrap 95% CIs (10000 resamples).

Decision rule (fixed now). r2_lp in int8 replaces the stored recipe for new adapters if,
pooled: (i) efficacy with note >= ref16 - 0.02 and paraphrase with note >= ref16 - 0.03,
with the lower CI bound of paraphrase-with-note (r2_lp - ref16) >= -0.06; (ii) its share
above kappa <= ref16's share + 0.02 and its mean held-out D <= ref16's + 0.05; (iii)
wrong-route retention >= ref16 - 0.03. Otherwise the stored recipe stays (int8 storage,
SS86). The reported N=1000 store results are not recomputed here.
Predictions. P1 r2_lp int8 passes the rule. P2 its adapter-alone paraphrase is within
0.03 of ref16's. P3 adapter facts are 15% to 25% of the 2000 drawn facts.

## 88. SS85 results, automatic rule (2026-10-20)

Run as pinned with the deviation of 2026-10-19 (natural_e3.py; natural_e3_queries.json,
natural_e3.json, label sample natural_e3_labels.json drawn before the rates were
computed). SmolLM3-3B wrote 1000 questions; 7 do not contain the subject and were
dropped (993 kept: 500 near-neighbour, 493 uniform).

| router (theta) | routed: all / near / uniform | leak, adapter + note: all / near / uniform | leak, tiered: all |
| :-- | :-- | :-- | :-- |
| default, SS76 encoder (0.7501) | 0.157 / 0.310 / 0.002 | 0.146 / 0.288 / 0.002 | 0.106 |
| SS74 default, MiniLM (0.757) | 0.119 / 0.230 / 0.006 | 0.107 / 0.206 / 0.006 | 0.083 |
| frozen mpnet (0.6694) | 0.125 / 0.244 / 0.004 | 0.117 / 0.228 / 0.004 | 0.085 |

Leak is the automatic rule (the response contains the queried fact's target and the base
response does not); the SS71 any-answer rule gives the same figures within 0.003.
Default minus frozen mpnet routed: +0.032 (95% CI 0.006 to 0.058).
Default router, near-neighbour queries by relation pair (n, routed, leak): P495/P740 64,
0.53, 0.53; P176/P178 60, 0, 0; P103/P37 51, 0.27, 0.27; P103/P1412 49, 0.18, 0.16;
P364/P407 45, 0.71, 0.69; P1412/P37 45, 0.22, 0.22; P17/P30 43, 0.40, 0.30; P131/P30 41,
0.51, 0.39; P20/P937 28, 0, 0; P19/P937 22, 0, 0; P131/P276 21, 0.81, 0.81; P131/P17 16,
0.06, 0.06; P19/P20 15, 0, 0. The frozen mpnet errs on other pairs (P176/P178 0.35,
P17/P30 0.63) and rarely on P364/P407 (0.04) or P131/P30 (0).
Predictions: P1 holds (0.157 >= 0.07); P2 holds (+0.032, CI excludes 0); P3 holds (near
minus uniform +0.308); P4 holds (0.146 >= 0.043). P5 waits for the labels.
Readings. Natural questions about another relation of the subject are almost never routed
when that relation is unrelated (0.2% to 0.6%), and often routed when it is a near
neighbour (23% to 31%). The template E3 figure of 4.3% is therefore a property of
CounterFact templates, which pair relations uniformly; for questions about neighbouring
relations the default router routes and answers with the stored target about every third
time. Several of the most-routed pairs are near synonyms for the subjects involved (the
original language of a film and the language of the work; location and administrative
location), where the stored target can also be a correct answer; the automatic rule counts
those as leaks, which is what the labels estimate.
Decision (SS85 rule). P1 holds: the paper reports the default configuration's locality
on natural queries next to the template figure and states that 4.3% holds for
CounterFact templates. Corrected rates follow when the 120 labels exist.

## 89. SS87 results: rank-2 LoRA+ in int8 on fresh adapter facts (2026-10-20)

Run as pinned (rank2_fresh.py; rank2_fresh.json, facts and tiers in
rank2_fresh_facts.json, adapters in rank2_fresh_ref16.pt / rank2_fresh_r2_lp.pt).
Of the 2000 drawn facts, 279 are adapter facts (117 CF, 162 zsRE; 13.95%). Training
1651 s (ref16) and 1732 s (r2_lp).

| arm | size per fact | efficacy | eff. + note | paraphrase | para. + note | held-out D mean / max | above kappa | wrong-route retained |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| ref16 (stored recipe) | 8.7 MB (fp32) | 0.986 | 0.946 | 0.835 | 0.918 | 0.401 / 1.401 | 0.022 | 0.695 |
| r2_lp, int8 | 373 KB | 0.989 | 0.932 | 0.821 | 0.928 | 0.390 / 1.407 | 0.029 | 0.705 |

r2_lp int8 - ref16, paired 95% CIs: efficacy +0.004 [-0.007, 0.018], efficacy with note
-0.014 [-0.039, 0.011], paraphrase -0.014 [-0.043, 0.014], paraphrase with note +0.011
[-0.022, 0.043]. Per dataset, above kappa: CF 0.043 / 0.043, zsRE 0.006 / 0.019.

Decision rule: (i) efficacy with note -0.014 >= -0.02, paraphrase with note +0.011 >=
-0.03, CI lower bound -0.022 >= -0.06; (ii) above kappa 0.029 <= 0.022 + 0.02, mean D
0.390 <= 0.401 + 0.05; (iii) wrong-route retention 0.705 >= 0.695 - 0.03. All hold:
rank-2 LoRA+ stored in int8 replaces the stored recipe for new adapters. The N=1000
stores and the results reported from them keep the rank-16 adapters.
Predictions: P1 holds; P2 holds (adapter-alone paraphrase -0.014, within 0.03); P3 fails
(13.95%, below the predicted 15% to 25%).
Size correction. The int8 size of a rank-2 adapter is 373 KB, not "about 280 KB" as
written in SS83, SS84 and SS86: the one fp16 scale per row of B (out x 2) adds about
100 KB. The adopted adapter is 23x smaller than the stored 8.7 MB, and the N=1000 store's
adapter facts (162 CF) would take 60 MB instead of 1.4 GB.

## 90. SS85 labels: who labels (deviation, 2026-10-20, before any label exists)

SS85 pinned that the author labels the 120 sampled routed answers. The author asked the
analysing model (Claude) to label them instead. Procedure, fixed before labelling: only
the 106 items that the scorer uses (the response with adapter and note contains the
stored target and the base response does not) are labelled; the other 14 cannot change
any figure and stay unlabelled. The items are shown to the labeller without the router
and without their ids, shuffled with seed 90; the labeller sees the question, the stored
fact, the stored target, the served response and the base response. The rule is the
label page's: in the counterfactual world where the stored fact is true, is the stored
target a correct answer to the question? A target of the wrong type (a city or a
continent for a country, a place of death for a place of birth) is a leak; a target
entailed by the stored fact for the asked relation (a country of creation for the country
where a group was formed; the original language of a work for its language) is a correct
answer; "unsure" is used where the asked relation and the stored one can differ for the
subject. The labels are written to natural_e3_labels_model.json before scoring. The
paper calls them model labels, not author labels. P5 is scored on them unchanged.

## 91. SS85 results with the model labels (2026-10-20)

Labelled as fixed in SS90 (natural_e3_labels_model.json; scored with natural_e3.py score,
natural_e3_scored.json). Of the 106 labelled items, 45 were labelled a correct answer, 33
a leak and 28 unsure. Most unsure labels are questions whose relation does not apply to
the subject (the "official language" of a person), a native language asked where the
stored fact gives a language the subject uses or a place's official language, and a
country asked where the question wants the administrative entity.

| router | sampled automatic leaks | correct / leak / unsure | correct share (95% Wilson) | leak, adapter + note: all, automatic / corrected | near-neighbour, automatic / corrected |
| :-- | :-- | :-- | :-- | :-- | :-- |
| default, SS76 encoder | 35 | 19 / 13 / 3 | 0.594 (0.423 to 0.745) | 0.146 / 0.059 | 0.288 / 0.117 |
| SS74 default, MiniLM | 36 | 11 / 9 / 16 | 0.550 (0.342 to 0.742) | 0.107 / 0.048 | 0.206 / 0.093 |
| frozen mpnet | 35 | 15 / 11 / 9 | 0.577 (0.389 to 0.745) | 0.117 / 0.049 | 0.228 / 0.096 |

P5 (the correction removes less than half of the default router's automatic leaks) fails:
it removes 59%. The verdict does not depend on the unsure labels (0.543 if they are all
leaks, 0.629 if they are all correct answers), but the interval includes 0.5.
Readings. The correct answers are mostly entailed pairs: a country of creation asked as
the place where a group was formed (P495/P740), an official language asked as a language
spoken in a place (P37/P1412), a native language asked as a language a person speaks
(P103/P1412), and the original language of a work asked as its language (P364/P407). The
leaks are mostly a target of the wrong type: a continent for a country (P30/P17), a
continent for an administrative entity (P30/P131), a city for a country of origin
(P740/P495). With the labels, the default router serves a wrong stored answer to about
12% of natural questions about neighbouring relations and 6% of all, against 4.3% on
CounterFact templates. Model labels are not independent of the analysis; the author's
labels, if made later, are reported next to them and do not replace them.

## 92. LoReFT against the adopted rank-2 LoRA+ adapter (pinned 2026-10-20, before the run)

Motivation. The adopted per-fact adapter (SS89) is rank-2 LoRA+ on q_proj and v_proj in
all layers, stored in int8: 373 KB. Representation finetuning (LoReFT, Wu et al. 2024)
edits hidden states instead of weights, h + R^T(W h + b - R h) with R an r x d matrix
with orthonormal rows, at chosen layers and token positions; at r = 4 in all 28 layers
it stores 2 x 4 x 1536 + 4 numbers per layer. This entry asks whether it gives a smaller
or a better per-fact adapter in this system.

Intervention (fixed). On the residual stream output of a decoder layer, at the last 8
positions of the prompt (the chat template's 5 closing tokens and the last 3 tokens of
the user turn). Earlier positions are not used: every query starts with the same system
prompt and template, so a prefix intervention could only add a fact-independent change.
Generated tokens are not intervened on (they see the intervened positions through the
cache), as in pyreft. Implemented as a forward hook in fp32 on the bf16 base, without
pyreft. Variants:
  loreft_r4_std   LoReFT, r = 4, all 28 layers; pyreft's initialisation (R orthogonal
                  random, W and b as nn.Linear defaults): not the identity at the start
  loreft_r4_id    as above with W = R and b = 0 at the start (the identity)
  direft_r4       DiReFT, h + W2^T (W1 h + b), r = 4, all layers, W2 = 0 at the start
  loreft_r2_id    LoReFT identity start, r = 2, all layers
  loreft_r4_id_L7 LoReFT identity start, r = 4, layers 3, 7, 11, 15, 19, 23, 27
Size per fact in int8 (one fp16 absmax scale per row of each matrix, b in fp16): 345 KB,
345 KB, 345 KB, 172 KB, 86 KB. R is trained under torch's orthogonal parametrisation and
stored as a plain matrix.
Training: as the per-fact adapters, 16 passes over the two pairs, KL anchor on the two
training canaries against the base (kl_lambda 1), AdamW with one learning rate for all
intervention parameters, torch.manual_seed(92) before each intervention is created.

Stage 1, learning rate (ReFT has no rate validated here). For each variant, lr in
{1e-3, 3e-3, 1e-2} on 20 development facts: CounterFact facts of the SS87 draw whose note
alone passes the tier check, outside the SS44 set, drawn with seed 92. Readings in int8:
efficacy, paraphrase (adapter alone), held-out D on the 40 SS69 canaries. Selection
(fixed now): among the rates whose held-out D max is <= kappa (0.968), the one with the
highest efficacy + paraphrase, ties to the lower rate; if no rate qualifies, the rate
with the lowest share above kappa, ties to the lower rate.

Stage 2, screen (the SS81-B setup). The SS44 set (CounterFact N=50), the SS81-B pairs, each
variant at its stage-1 rate, evaluated as stored in int8 (fp32 also reported): efficacy,
paraphrase alone and with the note, held-out D mean / max and share above kappa,
in-sample D, size, training seconds. Reference: r2_qv_all of SS83 (efficacy 0.98,
paraphrase 0.88, held-out D max 0.923). Rule (SS81's): efficacy >= 0.98, paraphrase
(adapter alone) >= 0.86, held-out D max <= kappa. Among the passing variants smaller than
373 KB in int8, the smallest is the candidate (ties to the higher paraphrase). No
candidate: the adopted adapter stays and stage 3 is not run.

Stage 3, confirmation (only with a candidate). The 279 SS87 adapter facts with facts.pairs,
candidate in int8 against the SS87 r2_lp int8 rows (same facts, same wrong-route draw),
readings and paired bootstrap as SS87. The candidate replaces the adopted adapter for new
adapters if, pooled: (i) efficacy with note >= r2_lp - 0.02 and paraphrase with note >=
r2_lp - 0.03, with the lower CI bound of paraphrase with note (candidate - r2_lp) >=
-0.06; (ii) share above kappa <= r2_lp + 0.02 and mean held-out D <= r2_lp + 0.05;
(iii) wrong-route retention >= r2_lp - 0.03. Adoption also needs the store to serve
interventions (hooks, state hash, removal), which is separate work.

Predictions.
  P1 loreft_r4_std fails the stage-2 rule.
  P2 loreft_r4_id and direft_r4 reach efficacy >= 0.98 in stage 2.
  P3 no variant reaches paraphrase (adapter alone) >= 0.86 in stage 2, so no candidate.
  P4 loreft_r4_id has a higher mean held-out D than SS83's r2_qv_all (0.447).
  P5 loreft_r4_id_L7 has lower stage-2 efficacy than loreft_r4_id.

Deviation (2026-10-20, during stage 1, before any valid reading). The first stage-1 run
trained the LoReFT rotation with AdamW's default weight decay (0.01) on the orthogonal
parametrisation's original tensor; within 32 steps the decay drove R to zero, so every
LoReFT intervention became the identity after training (held-out D 0.0000 for all 20
development facts of loreft_r4_std at 1e-3, the only completed setting). The run was
stopped and its output (reft_lr.json) deleted. The rotation is now trained without
weight decay (W, b and the DiReFT matrices keep AdamW's default, as the LoRA adapters),
and training asserts that R keeps orthonormal rows. pyreft trains with weight decay 0 by
default. Nothing else changes; stage 1 restarts from the beginning.

## 93. SS92 results: LoReFT and DiReFT (2026-10-20)

Run as pinned with the deviation recorded under SS92 (reft_compare.py; reft_lr.json,
reft_screen.json, interventions in reft_screen_<variant>.pt). The resumed run also
collects the trained modules after each fact (a GPU memory leak through the
parametrisation's reference cycles stopped the first full run at loreft_r2_id; the
completed settings were kept, nothing was recomputed).

Stage 1 (20 development facts, int8). No variant has a learning rate with held-out D max
<= kappa, so each takes the rate with the lowest share above kappa; that is 1e-3 for all
five. Held-out D grows with the rate for every variant (at 1e-2 the mean is 10 to 39).

| variant | 1e-3: efficacy / paraphrase / D mean / above kappa | 3e-3 | 1e-2 |
| :-- | :-- | :-- | :-- |
| loreft_r4_std | 1.00 / 0.75 / 5.52 / 1.00 | 0.95 / 0.80 / 13.02 / 1.00 | 0.70 / 0.20 / 17.53 / 1.00 |
| loreft_r4_id | 0.90 / 0.75 / 5.68 / 1.00 | 0.90 / 0.50 / 12.63 / 1.00 | 0.70 / 0.40 / 21.42 / 1.00 |
| direft_r4 | 0.95 / 0.80 / 1.66 / 0.70 | 1.00 / 0.75 / 4.91 / 1.00 | 0.35 / 0.15 / 39.42 / 1.00 |
| loreft_r2_id | 0.95 / 0.65 / 2.74 / 1.00 | 1.00 / 0.75 / 7.04 / 1.00 | 0.80 / 0.40 / 15.03 / 1.00 |
| loreft_r4_id_L7 | 1.00 / 0.50 / 1.45 / 0.85 | 1.00 / 0.60 / 3.87 / 0.95 | 0.90 / 0.45 / 10.15 / 1.00 |

Stage 2 (SS44 set, N=50, lr 1e-3, int8; fp32 gives the same picture: rates within 0.06,
D mean within 0.01).

| variant | size (int8) | efficacy | paraphrase | para. + note | held-out D mean / max | above kappa | in-sample D max | pass |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| r2_qv_all LoRA+ (SS83, reference) | 373 KB | 0.98 | 0.88 | 1.00 | 0.447 / 0.923 | 0 | 0.077 | yes |
| loreft_r4_std | 345 KB | 1.00 | 0.86 | 0.94 | 5.557 / 9.458 | 1.00 | 0.266 | no (D) |
| loreft_r4_id | 345 KB | 1.00 | 0.68 | 0.76 | 4.706 / 13.135 | 1.00 | 0.219 | no |
| direft_r4 | 345 KB | 0.98 | 0.88 | 0.90 | 1.949 / 5.591 | 0.82 | 0.065 | no (D) |
| loreft_r2_id | 172 KB | 1.00 | 0.86 | 0.94 | 3.140 / 7.492 | 1.00 | 0.119 | no (D) |
| loreft_r4_id_L7 | 86 KB | 0.98 | 0.68 | 0.90 | 1.743 / 4.928 | 0.90 | 0.038 | no |

(The reference row is SS83's, trained with seed 81 and evaluated in fp32; its int8 size is
the SS89 figure.) No variant passes, so there is no candidate and stage 3 was not run. The
adopted rank-2 LoRA+ adapter stays.
Predictions: P1 holds (loreft_r4_std fails, on D). P2 holds (loreft_r4_id 1.00, direft_r4
0.98). P3 fails: loreft_r4_std (0.86), direft_r4 (0.88) and loreft_r2_id (0.86) reach the
paraphrase limit; no candidate exists because of the held-out check alone. P4 holds (4.71
against 0.447). P5 holds (0.98 against 1.00).
Readings.
1. On quality, ReFT matches LoRA here: at 172 KB, loreft_r2_id has the paraphrase rate of
   the 373 KB LoRA+ adapter. What fails is divergence on prompts outside training: held-out
   D is 4 to 12 times the LoRA+ adapter's and above kappa for 82% to 100% of the facts,
   while in-sample D on the two anchored canaries stays as low as LoRA's (0.04 to 0.27).
2. The KL anchor on two prompts does not carry to other prompts for an intervention on the
   last prompt positions. Such an intervention acts on the position that produces the
   answer of every query, and the anchor only fixes it on the two queries it sees. This is
   the SS69 lesson (in-sample D understates an adapter's effect) in a stronger form.
3. The identity start does not help (loreft_r4_id is no closer to the base than
   loreft_r4_std); the variants differ in how the edit is parametrised, and the one
   without a projection (DiReFT) and the one with fewer layers diverge least.
4. Untested here and not pinned: a larger or held-out anchor set, a position gate keyed on
   the subject, or interventions on the subject's tokens instead of the last positions.
   Any of them would be a new entry.

## 94. The N=1000 stores with the adopted rank-2 adapter (pinned 2026-10-26, before the run)

Motivation. SS89 adopted rank-2 LoRA+ in int8 (373 KB) for new adapters, but every
end-to-end result in the paper (routing, serving arms, tiered store, removal, locality)
comes from the two N=1000 stores built with rank-16 adapters (8.7 MB). This entry
rebuilds both stores with the adopted recipe and repeats the end-to-end evaluation of the
default router, so that the main results can be reported for the adapter the system now
uses.

Build (fixed). For each of the 1000 CounterFact and 1000 zsRE facts of the stores, in the
stored order: a rank-2 LoRA+ adapter on q_proj and v_proj in all layers (A lr 3e-4, B lr
2.4e-3, 16 passes over facts.pairs, KL anchor on the two training canaries, kl_lambda 1;
adapter_size.train_one with b_mult 8 and seed 94), rounded to int8 with one fp16 scale
per row (SS83's round_to), then written into the rank-16 layout with the 14 unused rows
of A and columns of B set to zero. With lora_alpha = 2r the LoRA scale is 2 at both
ranks, so the padded adapter computes the same update; the existing evaluation code then
runs unchanged. Check before the build continues: on the first 20 CounterFact facts the
padded adapter in the store and the rank-2 adapter in a rank-2 model give the same greedy
answer to the edit prompt and the paraphrase, and the logits of the edit prompt differ by
at most 0.05; otherwise the run stops and this entry is revised. Commit gate as in the
stored build: in-sample D on the two training canaries <= 2.0 (store.propose_and_commit's
rule); a rejected candidate is retrained once with seed 95, and if it is rejected again
the run stops. Keys, notes and fact order are those of the rank-16 stores. Outputs:
cf_n1000_r2.ckpt and zsre_n1000_r2.ckpt (same format as the stored checkpoints), added to
routed_eval_manifest.json with their SHA256 once written.

Held-out divergence. For every adapter of both stores, rank-16 and rank-2: held-out D on
the 40 SS69 canaries against the base, and in-sample D. Reading: mean, p95, max and the
share above kappa (0.968), per dataset and pooled, paired by fact.

End-to-end evaluation (fixed). v7_eval.py with the SS76 default router
(router_v8_select.json: the trained relation encoder, theta 0.750) on each rank-2 store,
the same code and readings as {cf,zsre}_n1000_v8_eval.json: serving arms on the edit
prompt and paraphrase, the tiered store, removal, distractor leak and locality (E3 other
relation and E2 neighbourhood on CounterFact, loc questions on zsRE). Wrong route:
canary_eval.py on the rank-2 CounterFact store with its defaults (the same 200 zsRE dev
questions and the same random draw of adapters as SS69). Paired differences to the
rank-16 rows with a bootstrap over facts (10000 resamples, seed 94).

Decision rule (fixed now). The paper reports its main end-to-end results from the rank-2
stores, with the rank-16 results kept as the earlier build, if all of the following hold;
otherwise it reports both and keeps rank 16 as the main build:
  (i)   paraphrase with adapter and note (both) >= rank 16 - 0.03 on each dataset, and the
        lower bound of the paired 95% CI of the difference >= -0.06;
  (ii)  efficacy with adapter alone >= rank 16 - 0.02 on each dataset;
  (iii) E3 leak (both) <= rank 16 + 0.01 on CounterFact;
  (iv)  share of adapters above kappa (pooled over 2000) <= rank 16 + 0.02, and mean
        held-out D <= rank 16 + 0.05;
  (v)   wrong-route retention >= rank 16 - 0.03 (SS69: 0.72).

Predictions.
  P1 All 2000 candidates pass the commit gate on the first training.
  P2 (i) and (ii) hold on both datasets (SS84: rank-2 LoRA+ matched rank 16 on the
     adapter facts of these stores).
  P3 Adapter-alone paraphrase is within 0.03 of rank 16 on each dataset.
  P4 (iv) holds pooled, with a gap in the share above kappa of at most 0.02; on zsRE
     alone the rank-2 share is higher than the rank-16 share (SS84 on the adapter facts:
     9.7% against 7.3%).
  P5 (iii) and (v) hold.
  P6 Removal on CounterFact stays complete (gone 1.0, retention >= 0.99).
Storage of the two stores is computed, not predicted: 2 x 1000 x 373 KB, about 0.75 GB
instead of 17.4 GB.

Deviation (2026-10-26, after the padding check, before any evaluation reading). The
pinned padding check failed on its logit criterion: on the first 20 CounterFact facts
the padded adapter in the store and the rank-2 adapter in a rank-2 model gave identical
greedy answers to all 20 edit prompts and to 19 of 20 paraphrases, but the edit-prompt
logits differed by up to 0.42 (0 for two facts, 0.19 to 0.42 for the others). The run
stopped as pinned. Diagnosis (scratch script, six facts): B @ A of the padded and the
rank-2 adapter are equal in fp32 (difference 0.0); the base logits of the two models are
equal (0.0); moving the rank-2 rows to slots 14 and 15 of the rank-16 layout changes the
store's logits by 0.0. The difference therefore comes from the bf16 matrix product with
an inner size of 2 against 16 (a different reduction order in the kernel), not from the
padding; at logit magnitudes of 23 to 29 one bf16 step is 0.125, so 0.19 to 0.42 is one
to three steps. The 0.05 limit was set without this in mind. The paraphrase that differs
(x0002) is a miss in both models. Revised check (applied to the same 20 facts): B @ A
equal in fp32 within 1e-6, and the hit on the target (edit prompt and paraphrase) the
same in both models for all 40 answers; the logit difference is reported. What SS94
evaluates is the padded store, which computes the rank-2 update up to bf16 rounding.
Nothing else changes; the 20 trained adapters are kept and the build resumes.

## 95. SS94 results: the N=1000 stores with the rank-2 adapter (2026-10-27)

Source: results/live_learning/confirm/store_r2.json (store_r2.py report), from
store_r2_{cf,zsre}.json, {cf,zsre}_n1000_r2_v8_eval.json and canary_eval_r2.json. The
two rank-2 stores are pinned in routed_eval_manifest.json (counterfact_n1000_r2,
zsre_n1000_r2). Training took 6.9 s per adapter on CounterFact and 6.6 s on zsRE.
Storage of the two stores as int8 rank-2 adapters: 2 x 1000 x 373 KB, about 0.75 GB
instead of 17.4 GB.

Serving (rank 16 against rank 2, paired difference with its 95% bootstrap CI).
CounterFact: efficacy with adapter alone 0.997 against 0.998 (+0.001, [-0.002, 0.005]);
paraphrase with adapter and note 0.891 against 0.885 (-0.006, [-0.016, 0.004]);
paraphrase with adapter alone 0.730 against 0.732 (+0.002, [-0.015, 0.019]); tiered
store: 0.162 against 0.161 of facts need an adapter, tiered paraphrase 0.723 against
0.724. zsRE: efficacy 0.999 against 1.000 (+0.001, [0.000, 0.003]); paraphrase with
adapter and note 0.982 against 0.975 (-0.007, [-0.015, 0.000]); adapter alone 0.967
against 0.969 (+0.002, [-0.003, 0.008]); tier fraction 0.206 in both.

Locality and removal. CounterFact E3 leak (both) 0.043 against 0.044 (CI [0.000,
0.003]); E2 neighbourhood leak (both) 0.017 in both; zsRE loc questions 0 in both.
Removal on CounterFact: gone 1.0 in both, retention 0.9967 against 0.9989; on zsRE the
removal readings are those of rank 16 (gone 0.4, retention 0.9989 against 1.000).

Held-out divergence (40 SS69 canaries, kappa 0.968). Share above kappa, rank 16 against
rank 2: CounterFact 0.011 against 0.019, zsRE 0.078 against 0.089, pooled 0.0445 against
0.054. Mean held-out D: CounterFact 0.354 against 0.396, zsRE 0.545 against 0.551,
pooled 0.450 against 0.474. No candidate failed the commit gate (in-sample D max 0.12 on
CounterFact).

Wrong route (canary_eval.py, the same 200 zsRE dev questions and the same adapters as
SS69, paired by question): base answers retained 0.72 with rank 16 and 0.675 with rank
2 (difference -0.045, bootstrap CI [-0.095, 0.005]; 17 answers lost and 8 gained). The
p99 of benign held-out D over the 100 SS69 benign adapters is 1.11 with rank 2 against
0.968 with rank 16.

Decision rule. (i) holds on both datasets, (ii) holds on both, (iii) holds, (iv) holds
(gap in the share above kappa 0.0095, in mean D 0.024), (v) fails: 0.675 is below the
limit of 0.69. The rule fails, so as pinned the paper reports both builds and keeps rank
16 as the main build. The difference in (v) is within the sampling error of 200
questions, but the limit was fixed as a point value.

Predictions. P1 held: all 2000 candidates passed the gate on the first training. P2
held. P3 held (+0.002 on both datasets). P4 held: the pooled gap is 0.0095, and on zsRE
the rank-2 share (8.9%) is higher than the rank-16 share (7.8%). P5 failed: (iii) held,
(v) did not. P6 held (gone 1.0, retention 0.9989).

Reading. On the 2000 stored facts the rank-2 adapter answers as the rank-16 adapter does
(every serving difference is under 0.01) and leaks as little. It changes the base model
slightly more: on held-out prompts, and when it is applied to a question it was not
trained for, where it keeps 4.5 points fewer of the base model's correct answers. New
facts continue to be stored with the rank-2 adapter (SS89); a larger wrong-route sample
would tell whether the difference in (v) is real.
