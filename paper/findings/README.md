# Findings for the paper

The current findings of RELIC, as tables to write the paper from. Every number is copied from a results entry of
`docs/LIVING_MODEL_CONFIRMATORY_PREREG.md`, and that entry is the record; if a number here and the entry differ, the entry wins.
State as of entry 159 (2026-11-14).

| File | One row is |
|---|---|
| `system.csv` | a component of the default system, with the entry that adopted it |
| `results.csv` | one reading: a metric of one arm on one dataset |
| `decisions.csv` | one pinned question, its rule outcome and what was decided |
| `open.csv` | a limitation or an unread part the paper must state |

## results.csv

- `entry`: the prereg entry that holds the result (not the SS id that pinned it; `decisions.csv` gives both).
- `section`: `default_fresh` and `default_seen` are the default store (no note, retry ladder, rank 2 int8);
  `closed_lines` are ideas that failed their rule and stay as negative results; the rest group the evidence.
- `arm`: `default` is the store with no note; `adapter_and_note` is the store before entry 143, the main comparator.
- `metric`: the names of entry 157 (edit accuracy, paraphrase accuracy, leak rate, wrong-route retention, removal,
  capability cost, cost). Earlier entries' "efficacy" is `edit_accuracy` here, with `scoring` saying how it was scored.
- `value`, `ci_low`, `ci_high`: rates as fractions; the 95% interval when the entry gives one. Counts and paired
  differences that have no column of their own are in `note`.
- `facts`: `fresh` (no decision has seen them), `seen` (the N=1000 stores or a set read before), `dev`, `n/a`.
- `scoring`: `word` (target at a word boundary, `word_hit`), `substring` (`hit`), `auroc`, `count`, `kappa`, `accuracy`, `bytes`, `seconds`.
- `source`: the file under `results/live_learning/confirm/`, which is not in git.

## decisions.csv

`status` is `current` (part of the default system), `evidence` (a reading the paper reports, no default changed),
`closed` (a line that failed its rule), `superseded` or `open`.
