# Contributing

## Setup

```bash
uv sync                                     # full local environment: dev, quant, experiments
uv sync --no-default-groups                 # lean: the library only
uv sync --no-default-groups --group dev     # + test and lint tools (what CI runs)
```

The optional extras remain available to pip users, for example `pip install -e ".[dev,quant]"`.

`pyproject.toml` is the only dependency source; `uv.lock` pins it. Python >= 3.10.

## Checks (what CI runs)

```bash
.venv/bin/python -m pytest
ruff check relic experiments tests && black --check relic tests
```

Tests need no GPU, model download or `results/` directory.

## Rules the codebase depends on

1. **A refactor that changes a number is a bug, not a finding.** `relic/` must reproduce
   the stored results: compare routes and responses against a saved evaluation under
   `results/` before committing a change to the store, the router or the serving path.
2. **One variable per experiment; a failing arm is reported, not tuned.**
3. **Pre-register, then commit the runner, then run.** Every run is pinned in
   `docs/LIVING_MODEL_CONFIRMATORY_PREREG.md` with its predictions and decision rule, and
   tagged `prereg-ss<entry>`. Anything decided after data exists is a dated deviation that
   says so. Pinned entries are never rewritten. New decision rules are written over
   confidence intervals (a one-sided non-inferiority margin), not point thresholds.
4. **Guards are measurements.** A guard failure stops a run and is reported as an
   implementation failure, not as a scientific outcome.

## Commits

Conventional prefixes with a scope, as in the history: `feat(ledger): ...`,
`fix(tests): ...`, `docs(prereg): ...`, `results(ss96): ...`, `refactor(...)`,
`build(ci): ...`, `chore: ...`, `style: ...`. A `results(...)` commit records a run
that already happened; its runner and pre-registration are earlier commits.
