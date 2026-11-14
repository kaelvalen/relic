# Living model - confirmatory summary (updated 2026-11-14)

One-page record of the confirmatory phase. Sources: `LIVING_MODEL_CONFIRMATORY_PREREG.md`
SS33-78, then entries to 159 at the end; all numbers at N=1000 with Qwen2.5-1.5B-Instruct unless noted. Nothing here is
new - it consolidates. The paper (`paper/main.tex`) reports the same numbers.

## The claim

A frozen base model plus a transactional store. Each fact is an entry: a note (its
statement, served in the system prompt with an instruction to follow it) and, only if
the note fails on the edit prompt at write time, a LoRA adapter. A router whose
parameters do not depend on the stored facts applies at most one entry per query.
Exact removal and attribution follow from the data structure; answering, locality,
removal behavior and the divergence check are measured.

## Current system (default since SS76)

Router: complete-subject match, clause scoping, subject-masked relation texts compared by
a relation encoder (all-mpnet-base-v2 trained offline on 7500 triplets from facts
disjoint from the dev split and the test stores, then frozen), threshold 0.750 (the
lowest with CounterFact dev FPR <= 0.05). Served with adapter and note together.

| reading | CounterFact | zsRE |
| :-- | :-- | :-- |
| edit prompts answered | 0.996 [0.992, 0.999] | 0.989 [0.982, 0.995] |
| paraphrases answered (adapter and note) | 0.891 [0.872, 0.910] | 0.982 [0.973, 0.990] |
| paraphrases, adapter only / note only | 0.730 / 0.684 | 0.967 / 0.815 |
| other relation of a stored subject: leakage | 0.043 [0.031, 0.056] | - |
| same relation, other subject: routed / leakage | 0.018 / 0.017 | - |
| synthetic distractors, NQ loc questions routed | 0 | 0 |
| removal of 100: return to base answer | 1.00 | 0.84 (1.00 removing by content) |
| retention of the other 900 | 0.997 | 0.999 |
| facts needing an adapter (tiered store) | 0.162 | 0.206 |

Divergence check (SS69): 40 held-out canaries, kappa 0.968 (99th percentile over stored
adapters). The two training prompts understate an adapter's effect ~17x (mean D 0.022 vs
0.364); 10/10 harmful candidates rejected, the closest 3.6x above kappa. A wrong route
keeps only 72% of the base model's correct answers.

Held-out relations (11 of 34 CounterFact relations never seen by the encoder):
paraphrase routing 0.747 (seen relations 0.994), leakage unchanged from router v4.

## How the numbers moved

| stage | CF paraphrase | other-relation leakage | prereg |
| :-- | :-- | :-- | :-- |
| first version (router v3, adapter) | 0.323 | 0.346 | SS64/65 |
| router v4, tau 0.631, adapter / adapter and note | 0.669 / 0.803 | 0.186 / 0.237 | SS66-68, SS71 |
| router v4, tau 0.757 (locality-oriented), adapter and note | 0.589 | 0.063 | SS74 |
| mpnet with generated keys, tau 0.693 | 0.620 | 0.050 | SS75 |
| trained relation encoder, tau 0.750 (default) | 0.891 | 0.043 | SS76 |

Attempts that did not improve on the router-v4 threshold curve: subject-local KL anchors
(SS70), notes instead of adapters (SS71), store-key contrast (SS72), base-model relevance
check (SS73), answer-type score (SS74, eliminated on dev). Attempts that did not transfer
locality to held-out relations: an encoder trained on 1785 Wikidata relations (SS77),
write-time negative keys (SS78).

## First version (frozen 2026-10-09), with corrections

The 2026-10-09 summary reported efficacy 0.999 / 0.996; that evaluation served each edit
prompt with its own adapter whenever any route fired. Routed end to end (SS64) the first
version answers 0.997 / 0.994. Its canary check used the two prompts the KL term trains
on and never bound (max 0.092 vs kappa 2.0). Other first-version readings (SS56-62):
distractor no-leak 0.992 / 0.975, return to base after removal 0.96 / 0.80, retention
0.9989 / 0.9944, provenance 1.00 with a stable hash (100-sample).

## Gates and baselines

- strong-RAG gate: instructed retrieval 0.837 on CounterFact edit prompts (SS33-47).
- official GRACE (EasyEdit): plain prompt 1.00, chat 0.025 (SS54).
- official WISE (EasyEdit): per-step 1.00, final 0.07; with its merge lifecycle 0.125,
  no-leak collapses to 0.27 (SS54/63).
- capacity: one grouped adapter carries 256 facts (efficacy 1.00, KLD 0.007; leakage
  0.1-0.3, corrected in SS60/61).
- storage 8.7 MB per adapter (fp32); add ~8.5 s, serve 0.04 s, remove ~1.4-2.5 ms.

## Carried caveats

- One base model, one GPU, single-token targets, substring scoring.
- zsRE uses reference answers, which the base model often knows.
- The trained relation encoder learned CounterFact's relation inventory; on new relations
  it transfers part of the recall and none of the locality. Retraining it changes every
  key at once, so Proposition 2 (locality of adds and removals) holds between retrainings.
- Remaining leakage comes mostly from near-neighbor relations (country of origin vs
  location of formation, original language vs language of the work, manufacturer vs
  developer).
- WISE/GRACE comparisons are serving-format sensitive and use 200 facts.

## Open

- Separating unseen near-neighbor relations (needs examples of the pair).
- A second base model to replicate the main results.
- Releasing the saved stores, the trained encoders and the generated keys/templates.

The 2026-10-09 open decision (paper identity) is settled: the paper is a systems paper
about the transactional store and its measured guarantees.

## Since 2026-10-11 (entries to 146)

Each line is a reading from the pre-registration record; the README's table of checks has them with their entries.

- Adapter size. Rank 2 LoRA+ in int8 (373 KB) matches rank 16 (8.7 MB) on 279 fresh facts and is what `Ledger.write`
  trains. The N=1000 stores keep rank 16 as the main build, as a rank-2 adapter on a wrong route keeps fewer answers:
  0.668 against 0.705 on 1000 questions, lower end -0.058 against a margin of -0.03.
- Wrong route as served. Adapter and note keep 40% of the base model's correct answers, the note alone 51%, the adapter alone 71%.
- Other models. The share of facts that need an adapter is 3.4% at 0.5B, 21.2% at 1.5B, 11.0% at 3B, 8.8% at 7B and 5.0% on
  SmolLM3-3B. Size does not explain it. The divergence limits do not transfer. Leakage, removal and wrong route were read on 1.5B only.
- Repeated writes. A same-answer write joins the entry, and zsRE return to base rises from 0.84 to 1.00.
- Rejected designs. Shards, a soft prompt, hard negatives for the router, LoReFT and DiReFT.
- A store with no note (entries 126 to 137). Training on rewrites lifts CounterFact paraphrases alone from 0.73 to 0.95, and
  the router absorbs the leak that brings. A retry ladder removes the cost of the gate. On 800 fresh facts the store still fails
  its rule on CounterFact paraphrases, 0.895 against 0.915, lower bound -0.045 against a margin of -0.03, and is ahead on edit
  prompts, 0.986 against 0.965. By the author's decision (entry 143) it is now the default, and `Ledger` reproduces it (entry 144).
  The current system is therefore an adapter per fact with no note; the tables above describe the earlier stores.
- Labels (entries 140, 142). The author's labels agree with the model's, kappa 0.967 on legit against leak, and give a corrected
  near-relation leak of 12.8% against 11.7%. Per-relation thresholds remove correct answers and leaks at similar rates.
- Prior work. The architecture is published in SoLA and RRDA, both before this repository. See the paper's related work.
- Thresholds per relation (entries 138, 139). On 400 fresh CounterFact facts as served, the automatic near-relation leak falls from
  0.296 to 0.171 at 0.25 points of paraphrase, and the write path takes them as `relation_tau`. zsRE has no relation identifier.
- The main evaluation, CounterFact half (entry 146). On the 1000 main-store facts the store with no note answers 0.995 of edit
  prompts and 0.844 of paraphrases against 0.885 with adapter and note; removal returns 1.00 and retention is 0.998; a wrong route
  keeps 0.558 of correct answers against 0.369 at the same length. 18% of facts needed a retry, and a write took 21 s at the median.

## Since 2026-11-12 (entries 147 to 159)

- An entailment signal (entry 148). A frozen NLI model ranks correct near-relation answers above leaks better than the router's
  score, AUROC 0.756 against 0.685, but at its operating point removes 17% of leaks and 5% of correct answers, short of the rule.
- The hypernetwork (entries 150, 152). Generated adapters fit their training facts and answer 0.10 of unseen facts, 0.77 with the
  answer's identity as input, and none stays under the divergence limit. The line closed at the author's threshold of 1.00.
- A null-space projection (entry 155). A wrong route keeps 0.91 of base answers against 0.68, and paraphrases fall from 0.78 to 0.32.
- The metric contract (entry 157). The headline set is edit accuracy, paraphrase accuracy, leak rate, wrong-route retention,
  removal, capability cost and cost. "Efficacy" is retired, as ours is free generation and not a teacher-forced probability.
- Another language (entries 156, 159). On BMIKE-53 the router's AUROC is 0.975 in English and 0.532 in Turkish. A multilingual
  encoder on the same triplets gives 0.939 in Turkish and misses the English margin by 0.0001; adding translated Turkish triplets
  gives 0.930 and 0.948. The router stays. English rephrasings written as questions route right 0.552 of the time.
- Open. The zsRE half of the main evaluation (stopped at 550 of 1000), a signal that separates near-relation leaks from correct
  answers, that store beyond 1.5B, routing outside English, time to first token, and the state hash, which names the model
  without its revision. The findings are tabulated in `paper/findings/`.
