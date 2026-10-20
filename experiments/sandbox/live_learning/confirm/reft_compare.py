"""SS92: LoReFT and DiReFT per-fact interventions against the rank-2 LoRA+ adapter.

An intervention edits the residual stream output of chosen decoder layers at the last
8 prompt positions (forward hook, fp32 on the bf16 base, no pyreft). Stage 1 picks a
learning rate per variant on 20 development facts, stage 2 screens the variants on the
SS44 set with SS81's rule, stage 3 confirms a candidate on the SS87 adapter facts against
the SS87 r2_lp int8 rows.

    .venv/bin/python experiments/sandbox/live_learning/confirm/reft_compare.py lr
    .venv/bin/python experiments/sandbox/live_learning/confirm/reft_compare.py screen
    .venv/bin/python experiments/sandbox/live_learning/confirm/reft_compare.py confirm
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text"))

import torch  # noqa: E402
from adapter_size import KAPPA, _int8_rows, mean_d  # noqa: E402
from canary_eval import HELDOUT  # noqa: E402
from facts import pairs as fact_pairs  # noqa: E402
from harness import chat, load_model  # noqa: E402
from pilot import CANARIES, SYSTEM, hit, rag_system, trim  # noqa: E402
from rank2_confirm import READS, store_logits, summarize  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn.utils.parametrizations import orthogonal  # noqa: E402

SEED = 92
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CONFIRM = Path("results/live_learning/confirm")
POSITIONS = 8  # last prompt positions intervened on
STEPS = 16
LRS = [1e-3, 3e-3, 1e-2]
ALL = None
L7 = [3, 7, 11, 15, 19, 23, 27]
VARIANTS = {
    "loreft_r4_std": dict(kind="loreft", r=4, layers=ALL, init="std"),
    "loreft_r4_id": dict(kind="loreft", r=4, layers=ALL, init="id"),
    "direft_r4": dict(kind="direft", r=4, layers=ALL, init="id"),
    "loreft_r2_id": dict(kind="loreft", r=2, layers=ALL, init="id"),
    "loreft_r4_id_L7": dict(kind="loreft", r=4, layers=L7, init="id"),
}
R2_LP_BYTES = 373_000  # SS89: rank-2 LoRA+ in int8


# -- interventions ----------------------------------------------------------
class LoReFT(nn.Module):
    """h + R^T (W h + b - R h), R with orthonormal rows."""

    def __init__(self, d: int, r: int, init: str):
        super().__init__()
        rot = nn.Linear(d, r, bias=False)
        nn.init.orthogonal_(rot.weight)
        self.rot = orthogonal(rot)
        self.src = nn.Linear(d, r)
        if init == "id":
            with torch.no_grad():
                self.src.weight.copy_(self.rot.weight)
                self.src.bias.zero_()

    def forward(self, h):
        return h + (self.src(h) - self.rot(h)) @ self.rot.weight

    def export(self) -> dict:
        return {
            "R": self.rot.weight.detach().float().cpu().clone(),
            "W": self.src.weight.detach().float().cpu().clone(),
            "b": self.src.bias.detach().float().cpu().clone(),
        }


class DiReFT(nn.Module):
    """h + W2^T (W1 h + b), W2 = 0 at the start."""

    def __init__(self, d: int, r: int, init: str):
        super().__init__()
        self.src = nn.Linear(d, r)
        self.out = nn.Parameter(torch.zeros(r, d))

    def forward(self, h):
        return h + self.src(h) @ self.out

    def export(self) -> dict:
        return {
            "W1": self.src.weight.detach().float().cpu().clone(),
            "b": self.src.bias.detach().float().cpu().clone(),
            "W2": self.out.detach().float().cpu().clone(),
        }


def apply_stored(kind: str, t: dict, h: torch.Tensor) -> torch.Tensor:
    if kind == "loreft":
        return h + (h @ t["W"].T + t["b"] - h @ t["R"].T) @ t["R"]
    return h + (h @ t["W1"].T + t["b"]) @ t["W2"]


def int8_intervention(stored: dict) -> dict:
    """Matrices as int8 rows with an fp16 scale each, the bias in fp16."""
    return {
        layer: {
            k: (v.half().float() if k == "b" else _int8_rows(v)) for k, v in t.items()
        }
        for layer, t in stored.items()
    }


def stored_bytes(stored: dict) -> int:
    total = 0
    for t in stored.values():
        for k, v in t.items():
            total += v.numel() * 2 if k == "b" else v.numel() + v.shape[0] * 2
    return total


class Runner:
    """Forward hooks on the decoder layers; `fn(layer, h)` edits the intervened slice.

    end = None: the last POSITIONS positions of a pass with more than one position (the
    prompt pass of generate, or a prompt-only forward); single-position passes (cached
    generation steps) are left alone. end = n: positions n - POSITIONS .. n - 1 (training
    on prompt + answer).
    """

    def __init__(self, model):
        self.fn = None
        self.end = None
        self.layers = model.model.layers
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(self._hook(i))

    def _hook(self, i):
        def hook(module, args, output):
            if self.fn is None:
                return None
            hs = output[0] if isinstance(output, tuple) else output
            t = hs.shape[1]
            if self.end is None and t < 2:
                return None
            end = t if self.end is None else self.end
            start = max(0, end - POSITIONS)
            new = self.fn(i, hs[:, start:end].float())
            if new is None:
                return None
            hs = torch.cat([hs[:, :start], new.to(hs.dtype), hs[:, end:]], dim=1)
            return (hs, *output[1:]) if isinstance(output, tuple) else hs

        return hook

    def serve(self, kind: str, stored: dict | None, device) -> None:
        if stored is None:
            self.fn = None
            return
        on = {
            int(layer): {k: v.to(device) for k, v in t.items()}
            for layer, t in stored.items()
        }
        self.fn = lambda i, h: apply_stored(kind, on[i], h) if i in on else None

    def train_with(self, modules: dict) -> None:
        self.fn = lambda i, h: modules[str(i)](h) if str(i) in modules else None


# -- training ---------------------------------------------------------------
def make_modules(spec, d, n_layers, device):
    torch.manual_seed(SEED)
    cls = LoReFT if spec["kind"] == "loreft" else DiReFT
    layers = range(n_layers) if spec["layers"] is None else spec["layers"]
    return nn.ModuleDict({str(i): cls(d, spec["r"], spec["init"]) for i in layers}).to(
        device
    )


def pair_step(tok, model, runner, modules, opt, user, assistant, base_logits):
    pair = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    prompt = tok.apply_chat_template(
        pair[:-1], add_generation_prompt=True, return_tensors="pt"
    )
    full = tok.apply_chat_template(pair, return_tensors="pt")
    pid = prompt["input_ids"] if hasattr(prompt, "keys") else prompt
    fid = (full["input_ids"] if hasattr(full, "keys") else full).to(model.device)
    labels = fid.clone()
    labels[:, : pid.shape[1]] = -100
    runner.train_with(modules)
    runner.end = pid.shape[1]
    loss = model(input_ids=fid, labels=labels).loss
    runner.end = None
    kl = 0.0
    for q, b in zip(CANARIES, base_logits):
        pl = torch.log_softmax(store_logits(tok, model, q), -1)
        pb = torch.softmax(b, -1)
        kl = kl + (pb * (pb.log() - pl)).sum()
    loss = loss + kl / len(CANARIES)
    loss.backward()
    opt.step()
    opt.zero_grad()


def train_one(tok, model, runner, spec, lr, pairs, base_logits) -> dict:
    cfg = model.config
    modules = make_modules(spec, cfg.hidden_size, cfg.num_hidden_layers, model.device)
    # weight decay on the orthogonal parametrisation's original tensor drives R to zero
    # (seen in the first SS92 run); the rotation is trained without it
    rot = [p for n, p in modules.named_parameters() if "parametrizations" in n]
    rest = [p for n, p in modules.named_parameters() if "parametrizations" not in n]
    groups = [{"params": rest}]
    if rot:
        groups.append({"params": rot, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=lr)
    for _ in range(STEPS):
        for user, assistant in pairs:
            pair_step(tok, model, runner, modules, opt, user, assistant, base_logits)
    runner.fn = None
    out = {i: m.export() for i, m in modules.items()}
    for t in out.values():
        if "R" in t:
            gram = t["R"] @ t["R"].T
            assert torch.allclose(gram, torch.eye(len(gram)), atol=1e-3), "R lost rank"
    del modules, opt
    # parametrised modules sit in reference cycles; without a collection their GPU
    # tensors accumulate over hundreds of trainings (OOM in the first full SS92 run)
    gc.collect()
    torch.cuda.empty_cache()
    return out


def sweep_pairs(f):
    """The SS81-B pairs."""
    return [(f["teach"], f"Not aldım: {f['teach']}"), (f["probe"], f["answer"])]


# -- evaluation -------------------------------------------------------------
class Bench:
    def __init__(self):
        self.tok, self.model = load_model(MODEL)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.runner = Runner(self.model)
        with torch.no_grad():
            self.base_held = [store_logits(self.tok, self.model, q) for q in HELDOUT]
            self.base_in = [store_logits(self.tok, self.model, q) for q in CANARIES]

    def ask(self, q, system, max_new_tokens):
        return chat(
            self.tok,
            self.model,
            [{"role": "system", "content": system}, {"role": "user", "content": q}],
            max_new_tokens=max_new_tokens,
        )

    @torch.no_grad()
    def d(self, prompts, base):
        return mean_d(base, [store_logits(self.tok, self.model, q) for q in prompts])

    def train_all(self, spec, lr, facts, pairs_of):
        out, t0 = {}, time.time()
        for f in facts:
            out[f["id"]] = train_one(
                self.tok, self.model, self.runner, spec, lr, pairs_of(f), self.base_in
            )
            trim()
        return out, round(time.time() - t0, 1)

    def screen_eval(self, spec, stored, facts, notes=True):
        rows = []
        for i, f in enumerate(facts):
            self.runner.serve(spec["kind"], stored[f["id"]], self.model.device)
            row = {
                "fact_id": f["id"],
                "efficacy": hit(self.ask(f["probe"], SYSTEM, 24), f["answer"]),
                "paraphrase": hit(self.ask(f["paraphrase"], SYSTEM, 24), f["answer"]),
                "d_heldout": self.d(HELDOUT, self.base_held),
                "d_in_sample": self.d(CANARIES, self.base_in),
            }
            if notes:
                row["paraphrase_note"] = hit(
                    self.ask(f["paraphrase"], rag_system("instruct", facts, i), 24),
                    f["answer"],
                )
            rows.append(row)
        self.runner.serve(spec["kind"], None, None)
        dh = [r["d_heldout"] for r in rows]
        di = [r["d_in_sample"] for r in rows]
        keys = ["efficacy", "paraphrase"] + (["paraphrase_note"] if notes else [])
        return {
            **{k: round(float(np.mean([r[k] for r in rows])), 3) for k in keys},
            "d_heldout_mean": round(float(np.mean(dh)), 4),
            "d_heldout_max": round(float(np.max(dh)), 4),
            "above_kappa": round(float(np.mean([x > KAPPA for x in dh])), 3),
            "d_in_sample_mean": round(float(np.mean(di)), 4),
            "d_in_sample_max": round(float(np.max(di)), 4),
            "rows": rows,
        }


def check_identity(bench) -> None:
    """Identity-start variants leave the base logits unchanged before training."""
    cfg = bench.model.config
    for name, spec in VARIANTS.items():
        if spec["init"] != "id":
            continue
        mods = make_modules(spec, cfg.hidden_size, cfg.num_hidden_layers, "cuda:0")
        bench.runner.train_with(mods)
        with torch.no_grad():
            d = bench.d(HELDOUT[:4], bench.base_held[:4])
        bench.runner.fn = None
        assert d == 0.0, (name, d)
        print(f"identity at start: {name} ok", flush=True)


def dev_facts():
    n50 = {
        f["subject"]
        for f in json.loads((CONFIRM / "external_counterfact_n50.json").read_text())[
            "facts"
        ]
    }
    fr = json.loads((CONFIRM / "rank2_fresh_facts.json").read_text())
    pool = [
        f
        for f in fr["sets"]["cf"]
        if fr["tiers"][f"cf:{f['id']}"] == "note" and f["subject"] not in n50
    ]
    return random.Random(SEED).sample(pool, 20)


def select_lr(res: dict) -> float:
    ok = [lr for lr in LRS if res[str(lr)]["d_heldout_max"] <= KAPPA]
    if ok:
        return max(
            ok,
            key=lambda lr: (
                res[str(lr)]["efficacy"] + res[str(lr)]["paraphrase"],
                -lr,
            ),
        )
    return min(LRS, key=lambda lr: (res[str(lr)]["above_kappa"], lr))


def save(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1))


def stage_lr(args) -> None:
    out = Path(args.out)
    report = json.loads(out.read_text()) if out.exists() else {"variants": {}}
    report.update({"prereg": "SS92-1", "kappa": KAPPA, "lrs": LRS})
    facts = dev_facts()
    report["dev_ids"] = [f["id"] for f in facts]
    bench = Bench()
    check_identity(bench)
    for name in args.variants.split(","):
        spec = VARIANTS[name]
        res = report["variants"].setdefault(name, {})
        for lr in LRS:
            if str(lr) in res:
                continue
            stored, secs = bench.train_all(spec, lr, facts, sweep_pairs)
            ev = bench.screen_eval(spec, int8_intervention_all(stored), facts, False)
            ev["train_seconds"] = secs
            res[str(lr)] = ev
            print(
                name,
                lr,
                json.dumps({k: v for k, v in ev.items() if k != "rows"}),
                flush=True,
            )
            save(out, report)
        report["selected"] = report.get("selected", {})
        report["selected"][name] = select_lr(res)
        print(name, "selected lr", report["selected"][name], flush=True)
        save(out, report)
    print("saved:", out)


def int8_intervention_all(stored: dict) -> dict:
    return {k: int8_intervention(v) for k, v in stored.items()}


def stage_screen(args) -> None:
    lr_report = json.loads(Path(args.lr_report).read_text())
    out = Path(args.out)
    report = json.loads(out.read_text()) if out.exists() else {"variants": {}}
    report.update({"prereg": "SS92-2", "kappa": KAPPA, "facts": args.facts})
    data = json.loads(Path(args.facts).read_text())
    facts = data["facts"] if isinstance(data, dict) else data
    bench = Bench()
    for name in args.variants.split(","):
        if name in report["variants"]:
            print(name, "done, skipped", flush=True)
            continue
        spec, lr = VARIANTS[name], lr_report["selected"][name]
        stored, secs = bench.train_all(spec, lr, facts, sweep_pairs)
        torch.save(stored, out.with_name(f"{out.stem}_{name}.pt"))
        example = next(iter(stored.values()))
        res = {
            "spec": spec,
            "lr": lr,
            "train_seconds": secs,
            "bytes_per_fact_int8": stored_bytes(example),
            "bytes_per_fact_fp32": sum(
                v.numel() * 4 for t in example.values() for v in t.values()
            ),
            "int8": bench.screen_eval(spec, int8_intervention_all(stored), facts),
            "fp32": bench.screen_eval(spec, stored, facts),
        }
        q = res["int8"]
        res["passes"] = (
            q["efficacy"] >= 0.98
            and q["paraphrase"] >= 0.86
            and q["d_heldout_max"] <= KAPPA
        )
        report["variants"][name] = res
        print(
            name,
            json.dumps(
                {
                    "lr": lr,
                    "bytes": res["bytes_per_fact_int8"],
                    **{k: v for k, v in q.items() if k != "rows"},
                    "passes": res["passes"],
                }
            ),
            flush=True,
        )
        save(out, report)
    eligible = [
        (v["bytes_per_fact_int8"], -v["int8"]["paraphrase"], n)
        for n, v in report["variants"].items()
        if v["passes"] and v["bytes_per_fact_int8"] < R2_LP_BYTES
    ]
    report["candidate"] = min(eligible)[2] if eligible else None
    save(out, report)
    print("candidate:", report["candidate"])
    print("saved:", out)


def stage_confirm(args) -> None:
    screen = json.loads(Path(args.screen).read_text())
    name = screen["candidate"]
    if name is None:
        print("no candidate in stage 2; stage 3 is not run (SS92)")
        return
    spec, lr = VARIANTS[name], screen["variants"][name]["lr"]
    fresh = json.loads((CONFIRM / "rank2_fresh.json").read_text())
    saved = json.loads((CONFIRM / "rank2_fresh_facts.json").read_text())
    sets, tiers = saved["sets"], saved["tiers"]
    items = [
        (f"{ds}:{f['id']}", f, facts, i)
        for ds, facts in sets.items()
        for i, f in enumerate(facts)
        if tiers[f"{ds}:{f['id']}"] == "adapter"
    ]
    keys = [k for k, *_ in items]
    damage = json.loads((CONFIRM / "canary_eval.json").read_text())["damage"]
    draw = fresh["draw"]
    out = Path(args.out)
    bench = Bench()
    pt = out.with_name(f"{out.stem}_{name}.pt")
    if pt.exists():
        stored = torch.load(pt, map_location="cpu", weights_only=False)
        secs = None
    else:
        t0, stored = time.time(), {}
        for n, (key, f, _, _) in enumerate(items):
            stored[key] = train_one(
                bench.tok,
                bench.model,
                bench.runner,
                spec,
                lr,
                fact_pairs(f),
                bench.base_in,
            )
            trim()
            if (n + 1) % 50 == 0:
                print(f"trained {n + 1}/{len(items)} ({time.time() - t0:.0f}s)")
        secs = round(time.time() - t0, 1)
        torch.save(stored, pt)
    q = int8_intervention_all(stored)
    rows = {}
    for n, (key, f, facts, i) in enumerate(items):
        bench.runner.serve(spec["kind"], q[key], bench.model.device)
        note = rag_system("instruct", facts, i)
        outp = {
            "probe": bench.ask(f["probe"], SYSTEM, 64),
            "probe_note": bench.ask(f["probe"], note, 64),
            "paraphrase": bench.ask(f["paraphrase"], SYSTEM, 64),
            "paraphrase_note": bench.ask(f["paraphrase"], note, 64),
        }
        rows[key] = {
            "outputs": outp,
            **{k: hit(v, f["answer"]) for k, v in outp.items()},
            "d_heldout": bench.d(HELDOUT, bench.base_held),
        }
        trim()
        if (n + 1) % 50 == 0:
            print(f"eval {n + 1}/{len(items)}", flush=True)
    wrong = []
    for dmg, key in zip(damage, draw):
        bench.runner.serve(spec["kind"], q[key], bench.model.device)
        resp = bench.ask(dmg["question"], SYSTEM, 64)
        wrong.append(
            {
                "question": dmg["question"],
                "adapter": key,
                "retained": hit(resp, dmg["answer"]),
                "response": resp,
            }
        )
    bench.runner.serve(spec["kind"], None, None)
    res = {"rows": rows, "wrong_route": wrong}
    ref = fresh["arms"]["r2_lp_int8"]
    subsets = {
        "pooled": keys,
        "cf": [k for k in keys if k.startswith("cf:")],
        "zs": [k for k in keys if k.startswith("zs:")],
    }
    summary = {
        "r2_lp_int8": {n: summarize(ref, ks) for n, ks in subsets.items()},
        name: {n: summarize(res, ks, ref) for n, ks in subsets.items()},
    }
    for arm, r in (("r2_lp_int8", ref), (name, res)):
        summary[arm]["wrong_route_retained"] = round(
            float(np.mean([w["retained"] for w in r["wrong_route"]])), 4
        )
    r, c = summary["r2_lp_int8"]["pooled"], summary[name]["pooled"]
    rule = {
        "i_efficacy_note": c["probe_note"] >= r["probe_note"] - 0.02,
        "i_paraphrase_note": c["paraphrase_note"] >= r["paraphrase_note"] - 0.03,
        "i_paraphrase_note_ci": c["minus_ref"]["paraphrase_note"]["ci95"][0] >= -0.06,
        "ii_above_kappa": c["above_kappa"] <= r["above_kappa"] + 0.02,
        "ii_d_mean": c["d_heldout_mean"] <= r["d_heldout_mean"] + 0.05,
        "iii_wrong_route": summary[name]["wrong_route_retained"]
        >= summary["r2_lp_int8"]["wrong_route_retained"] - 0.03,
    }
    rule["passes"] = all(rule.values())
    report = {
        "prereg": "SS92-3",
        "candidate": name,
        "lr": lr,
        "train_seconds": secs,
        "bytes_per_fact_int8": stored_bytes(next(iter(stored.values()))),
        "kappa": KAPPA,
        "arms": {name: res},
        "summary": summary,
        "rule": rule,
    }
    save(out, report)
    for arm in summary:
        p = summary[arm]["pooled"]
        print(
            arm,
            json.dumps(
                {k: p[k] for k in ("n", *READS, "d_heldout_mean", "above_kappa")}
            ),
            "wrong-route",
            summary[arm]["wrong_route_retained"],
            flush=True,
        )
    print("rule", json.dumps(rule))
    print("saved:", out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("lr")
    a.add_argument("--variants", default=",".join(VARIANTS))
    a.add_argument("--out", default=str(CONFIRM / "reft_lr.json"))
    b = sub.add_parser("screen")
    b.add_argument("--facts", default=str(CONFIRM / "external_counterfact_n50.json"))
    b.add_argument("--variants", default=",".join(VARIANTS))
    b.add_argument("--lr-report", default=str(CONFIRM / "reft_lr.json"))
    b.add_argument("--out", default=str(CONFIRM / "reft_screen.json"))
    c = sub.add_parser("confirm")
    c.add_argument("--screen", default=str(CONFIRM / "reft_screen.json"))
    c.add_argument("--out", default=str(CONFIRM / "reft_confirm.json"))
    args = ap.parse_args(argv)
    {"lr": stage_lr, "screen": stage_screen, "confirm": stage_confirm}[args.cmd](args)


if __name__ == "__main__":
    main()
