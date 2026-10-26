"""SS94: the N=1000 stores rebuilt with the adopted rank-2 LoRA+ int8 adapter.

build: per dataset, train a rank-2 LoRA+ adapter for every fact of the stored order,
round it to int8, pad it to the rank-16 layout (zero rows of A, zero columns of B; the
LoRA scale is 2 at both ranks), gate it on in-sample D <= 2.0, and record in-sample and
held-out D for the rank-2 and the stored rank-16 adapter of each fact. Writes
{tag}_n1000_r2.ckpt in the stored checkpoint format and store_r2_{tag}.json.
report: paired comparison of the v8 end-to-end evaluations and the divergence readings.

    .venv/bin/python experiments/sandbox/live_learning/confirm/store_r2.py build --dataset counterfact
    .venv/bin/python experiments/sandbox/live_learning/confirm/store_r2.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

import torch  # noqa: E402
from adapter_size import (  # noqa: E402
    KAPPA,
    make_variant,
    round_to,
    set_delta,
    train_one,
)
from canary_eval import HELDOUT  # noqa: E402
from facts import pairs as fact_pairs  # noqa: E402
from harness import chat, load_model  # noqa: E402
from ledger import sym_kl  # noqa: E402
from pilot import CANARIES, SYSTEM, PilotStore, trim  # noqa: E402
from rank2_confirm import R2  # noqa: E402
from routed_eval import (  # noqa: E402
    MANIFEST_PATH,
    REPO_ROOT,
    _repo_path,
    load_verified_checkpoint,
    verify_manifest_inputs,
)

SEED, RETRY_SEED = 94, 95
GATE = 2.0
R_STORE = 16
BOOT = 10000
CONFIRM = Path("results/live_learning/confirm")
FILES = {
    "counterfact": ("cf_n1000.ckpt", "external_counterfact_n1000.json", "cf"),
    "zsre": ("zsre_n1000.ckpt", "external_zsre_n1000.json", "zsre"),
}


def int8_one(delta):
    return {
        n: {"A": round_to(ab["A"], "int8"), "B": round_to(ab["B"], "int8")}
        for n, ab in delta.items()
    }


def pad(delta, r=R_STORE):
    """A rank-2 adapter in the rank-16 layout; B @ A and the LoRA scale are unchanged."""
    out = {}
    for n, ab in delta.items():
        a, b = ab["A"], ab["B"]
        pa = torch.zeros(r, a.shape[1], dtype=a.dtype)
        pb = torch.zeros(b.shape[0], r, dtype=b.dtype)
        pa[: a.shape[0]] = a
        pb[:, : b.shape[1]] = b
        out[n] = {"A": pa, "B": pb}
    return out


class Divergence:
    """In-sample and held-out D of one adapter against the zeroed store, base cached."""

    def __init__(self, store):
        self.store = store
        store.deltas = {}
        store.materialize()
        with torch.no_grad():
            self.base = {
                q: store._logits(store.model, q) for q in list(CANARIES) + HELDOUT
            }

    def __call__(self, delta):
        s = self.store
        s.deltas = {"__tmp__": delta}
        s.materialize()
        with torch.no_grad():
            cur = {q: s._logits(s.model, q) for q in self.base}
        s.deltas = {}
        s.materialize()

        def mean(qs):
            return float(np.mean([sym_kl(self.base[q], cur[q]) for q in qs]))

        return mean(CANARIES), mean(HELDOUT)


def verified_inputs(dataset):
    ckpt, facts_file, tag = FILES[dataset]
    manifest = json.loads(MANIFEST_PATH.read_bytes())
    model_id = manifest["models"]["base"]["id"]
    verified = verify_manifest_inputs(
        manifest,
        _repo_path(CONFIRM / ckpt, REPO_ROOT),
        _repo_path(CONFIRM / facts_file, REPO_ROOT),
        model_id,
        repo_root=REPO_ROOT,
    )
    checkpoint, _ = load_verified_checkpoint(
        verified["checkpoint_path"],
        verified["checkpoint_sha256"],
        lambda source: torch.load(source, map_location="cpu", weights_only=False),
    )
    return model_id, verified["facts"], checkpoint, tag


def identity_check(store, model_id, facts, r2, n=20):
    """The padded adapter in the store answers as the rank-2 adapter in a rank-2 model."""
    m = make_variant(load_model(model_id)[1], R2)
    m.eval()
    rows = []
    for f in facts[:n]:
        set_delta(m, r2[f["id"]])
        store.deltas = {f["id"]: pad(r2[f["id"]])}
        store.materialize()
        row = {"fact_id": f["id"]}
        for k in ("probe", "paraphrase"):
            msgs = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f[k]},
            ]
            row[k] = chat(store.tok, m, msgs) == chat(store.tok, store.model, msgs)
        with torch.no_grad():
            diff = store._logits(m, f["probe"]) - store._logits(store.model, f["probe"])
        row["logit_diff"] = float(diff.abs().max())
        rows.append(row)
    store.deltas = {}
    store.materialize()
    del m
    torch.cuda.empty_cache()
    ok = all(r["probe"] and r["paraphrase"] and r["logit_diff"] <= 0.05 for r in rows)
    return ok, rows


def build(args):
    model_id, facts, checkpoint, tag = verified_inputs(args.dataset)
    ref = checkpoint["deltas"]
    keys = checkpoint["keys"]
    out_ckpt = CONFIRM / f"{tag}_n1000_r2.ckpt"
    report_path = CONFIRM / f"store_r2_{tag}.json"
    train_path = CONFIRM / f"store_r2_{tag}_train.pt"
    if out_ckpt.exists():
        raise SystemExit(f"refusing to overwrite {out_ckpt}")
    report = (
        json.loads(report_path.read_text())
        if report_path.exists()
        else {"prereg": "SS94", "dataset": args.dataset, "kappa": KAPPA}
    )

    store = PilotStore(model_id, steps=16)
    store.kl_prompts = list(CANARIES)

    # Phase A: train (the store model waits on the CPU; train_one loads its own base).
    state = (
        torch.load(train_path, map_location="cpu", weights_only=False)
        if train_path.exists()
        else {"deltas": {}, "seconds": {}}
    )

    def check_padding():
        # Pinned on the first 20 CounterFact facts, before the build continues.
        store.model.to("cuda:0")
        ok, rows = identity_check(store, model_id, facts, state["deltas"])
        report["identity"] = {"ok": ok, "rows": rows}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
        print(f"identity check: {'ok' if ok else 'FAILED'}", flush=True)
        if not ok:
            raise SystemExit("padding check failed; the run stops (SS94)")

    needs_check = args.dataset == "counterfact" and "identity" not in report
    todo = [f for f in facts if f["id"] not in state["deltas"]]
    t0 = time.time()
    for n, f in enumerate(todo):
        if needs_check and len(state["deltas"]) >= 20:
            check_padding()
            needs_check = False
        store.model.to("cpu")
        torch.cuda.empty_cache()
        t = time.time()
        state["deltas"][f["id"]] = int8_one(
            train_one(store, R2, fact_pairs(f), b_mult=8, seed=SEED)
        )
        state["seconds"][f["id"]] = round(time.time() - t, 2)
        trim()
        done = len(state["deltas"])
        if done % 20 == 0 or n + 1 == len(todo):
            torch.save(state, train_path)
        if done % 50 == 0:
            print(
                f"{tag}: trained {done}/{len(facts)} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    if needs_check:
        check_padding()
    store.model.to("cuda:0")
    r2 = state["deltas"]

    # Phase B: gate and divergence of both arms, paired by fact.
    div = Divergence(store)
    rows, adds, padded = [], [], {}
    t0 = time.time()
    for n, f in enumerate(facts):
        fid = f["id"]
        p = pad(r2[fid])
        d_in, d_held = div(p)
        seed, retrained = SEED, False
        if not d_in <= GATE:
            store.model.to("cpu")
            torch.cuda.empty_cache()
            r2[fid] = int8_one(
                train_one(store, R2, fact_pairs(f), b_mult=8, seed=RETRY_SEED)
            )
            store.model.to("cuda:0")
            p = pad(r2[fid])
            d_in, d_held = div(p)
            seed, retrained = RETRY_SEED, True
            if not d_in <= GATE:
                raise SystemExit(f"{fid} rejected twice by the commit gate (SS94)")
        ref_in, ref_held = div(ref[fid])
        padded[fid] = p
        adds.append(
            {
                "fid": fid,
                "committed": True,
                "kld": round(d_in, 3),
                "seconds": state["seconds"][fid],
                "seed": seed,
            }
        )
        rows.append(
            {
                "fact_id": fid,
                "retrained": retrained,
                "r2_in_sample": d_in,
                "r2_heldout": d_held,
                "r16_in_sample": ref_in,
                "r16_heldout": ref_held,
            }
        )
        if (n + 1) % 100 == 0:
            print(
                f"{tag}: divergence {n + 1}/{len(facts)} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    report["rows"] = rows
    report["retrained"] = sum(r["retrained"] for r in rows)
    report["train_seconds_mean"] = round(
        float(np.mean(list(state["seconds"].values()))), 2
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))

    # Phase C: the store checkpoint, in the stored format.
    torch.save({"deltas": padded, "keys": keys, "adds": adds}, out_ckpt)
    print(f"saved: {out_ckpt}", flush=True)


def stats(xs):
    xs = np.asarray(xs, dtype=float)
    return {
        "n": int(xs.size),
        "mean": round(float(xs.mean()), 4),
        "p95": round(float(np.percentile(xs, 95)), 4),
        "max": round(float(xs.max()), 4),
        "above_kappa": round(float((xs > KAPPA).mean()), 4),
    }


def paired(a, b, seed=SEED):
    """Mean of b - a with a bootstrap 95% CI over the paired items."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, size=(BOOT, d.size))].mean(axis=1)
    return {
        "n": int(d.size),
        "diff": round(float(d.mean()), 4),
        "ci95": [
            round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4),
        ],
    }


def serving_rows(path):
    rows = json.loads(path.read_text())["examples"]["serving"]
    return {r["fact_id"]: r for r in rows}


def locality_rows(path, key):
    rows = json.loads(path.read_text())["examples"]["locality"][key]
    return {r["fact_id"]: r for r in rows}


def report(args):
    out = {"prereg": "SS94", "kappa": KAPPA, "datasets": {}}
    div_all = {"r2": [], "r16": []}
    for dataset, (_, _, tag) in FILES.items():
        ds = {}
        dv = json.loads((CONFIRM / f"store_r2_{tag}.json").read_text())
        for arm in ("r2", "r16"):
            ds[f"{arm}_heldout"] = stats([r[f"{arm}_heldout"] for r in dv["rows"]])
            ds[f"{arm}_in_sample"] = stats([r[f"{arm}_in_sample"] for r in dv["rows"]])
            div_all[arm] += [r[f"{arm}_heldout"] for r in dv["rows"]]
        ds["retrained"] = dv["retrained"]
        ds["train_seconds_mean"] = dv["train_seconds_mean"]
        v16 = CONFIRM / f"{tag}_n1000_v8_eval.json"
        v2 = CONFIRM / f"{tag}_n1000_r2_v8_eval.json"
        a, b = serving_rows(v16), serving_rows(v2)
        ids = sorted(a)
        assert sorted(b) == ids
        for prompt in ("probe", "paraphrase"):
            for arm in ("adapter", "retrieval", "both"):
                xa = [a[i][prompt][f"{arm}_hit"] for i in ids]
                xb = [b[i][prompt][f"{arm}_hit"] for i in ids]
                ds[f"{prompt}_{arm}"] = {
                    "r16": round(float(np.mean(xa)), 4),
                    "r2": round(float(np.mean(xb)), 4),
                    "r2_minus_r16": paired(xa, xb),
                }
        e16, e2 = json.loads(v16.read_text()), json.loads(v2.read_text())
        for k in ("tiered", "removal"):
            ds[k] = {"r16": e16[k], "r2": e2[k]}
        ds["locality"] = {}
        for key in e2["locality"]:
            la, lb = locality_rows(v16, key), locality_rows(v2, key)
            ds["locality"][key] = {}
            for pol in ("adapter", "note", "both", "tiered"):
                xa = [la[i][pol]["leak"] for i in sorted(la)]
                xb = [lb[i][pol]["leak"] for i in sorted(la)]
                ds["locality"][key][pol] = {
                    "r16": round(float(np.mean(xa)), 4),
                    "r2": round(float(np.mean(xb)), 4),
                    "r2_minus_r16": paired(xa, xb),
                }
        out["datasets"][dataset] = ds
    out["pooled_heldout"] = {arm: stats(xs) for arm, xs in div_all.items()}
    c16 = json.loads((CONFIRM / "canary_eval.json").read_text())
    c2 = json.loads((CONFIRM / "canary_eval_r2.json").read_text())
    out["wrong_route_retained"] = {
        "r16": c16["wrong_route_retained"],
        "r2": c2["wrong_route_retained"],
    }

    cf, zs = out["datasets"]["counterfact"], out["datasets"]["zsre"]
    ph = out["pooled_heldout"]
    rule = {}
    for name, ds in (("cf", cf), ("zs", zs)):
        pb = ds["paraphrase_both"]
        rule[f"i_{name}_paraphrase_both"] = pb["r2"] >= pb["r16"] - 0.03
        rule[f"i_{name}_paraphrase_both_ci"] = pb["r2_minus_r16"]["ci95"][0] >= -0.06
        pa = ds["probe_adapter"]
        rule[f"ii_{name}_efficacy"] = pa["r2"] >= pa["r16"] - 0.02
    e3 = cf["locality"]["E3_other_relation"]["both"]
    rule["iii_e3_leak"] = e3["r2"] <= e3["r16"] + 0.01
    rule["iv_above_kappa"] = ph["r2"]["above_kappa"] <= ph["r16"]["above_kappa"] + 0.02
    rule["iv_d_mean"] = ph["r2"]["mean"] <= ph["r16"]["mean"] + 0.05
    wr = out["wrong_route_retained"]
    rule["v_wrong_route"] = wr["r2"] >= wr["r16"] - 0.03
    rule["passes"] = all(rule.values())
    out["rule"] = rule
    path = CONFIRM / "store_r2.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: out[k] for k in ("rule", "pooled_heldout")}, indent=1))
    print(f"saved: {path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--dataset", choices=sorted(FILES), required=True)
    sub.add_parser("report")
    args = ap.parse_args(argv)
    torch.manual_seed(0)
    {"build": build, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
