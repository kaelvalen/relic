"""SS113: a per-fact soft prompt against the rank-2 LoRA+ adapter.

Ten trained vectors are spliced into the input embeddings just before the user turn.

    .venv/bin/python experiments/sandbox/live_learning/confirm/soft_prompt.py lr
    .venv/bin/python experiments/sandbox/live_learning/confirm/soft_prompt.py confirm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from adapter_size import mean_d  # noqa: E402
from canary_eval import HELDOUT  # noqa: E402
from rank2_confirm import READS, summarize  # noqa: E402
from reft_compare import dev_facts, select_lr  # noqa: E402

from relic.adapters import fact_pairs  # noqa: E402
from relic.ledger import CANARIES  # noqa: E402
from relic.metrics import hit  # noqa: E402
from relic.model import load_model  # noqa: E402
from relic.serving import SYSTEM, rag_system, trim  # noqa: E402

SEED = 113
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TOKENS = 10
PASSES = 32
LRS = [3e-3, 1e-2, 3e-2]
KAPPA = 0.968
CONFIRM = Path("results/live_learning/confirm")


class Bench:
    def __init__(self):
        self.tok, self.model = load_model(MODEL)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.emb = self.model.get_input_embeddings()
        self.im_start = self.tok.convert_tokens_to_ids("<|im_start|>")
        self.std = float(self.emb.weight.float().std())
        self.dim = self.emb.weight.shape[1]
        with torch.no_grad():
            self.base_in = [self.logits(None, q) for q in CANARIES]
            self.base_held = [self.logits(None, q) for q in HELDOUT]

    def _ids(self, messages, generation=True):
        ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=generation, return_tensors="pt"
        )
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        return ids.to(self.model.device)

    def embeds(self, soft, ids):
        """The prompt's embeddings, with the soft prompt before the last user turn's start."""
        e = self.emb(ids)
        if soft is None:
            return e
        starts = (ids[0] == self.im_start).nonzero().flatten()
        # 0: system turn, 1: user turn (the template adds a system turn if none is given)
        at = int(starts[1])
        return torch.cat([e[:, :at], soft.to(e.dtype).unsqueeze(0), e[:, at:]], dim=1)

    def logits(self, soft, q):
        # the same messages as DeltaStore._logits, so D is comparable with the stored rows
        ids = self._ids(
            [
                {"role": "system", "content": "Kısa ve net cevap ver."},
                {"role": "user", "content": q},
            ]
        )
        return self.model(inputs_embeds=self.embeds(soft, ids)).logits[0, -1].float()

    @torch.no_grad()
    def ask(self, soft, q, system, max_new_tokens):
        ids = self._ids(
            [{"role": "system", "content": system}, {"role": "user", "content": q}]
        )
        e = self.embeds(soft, ids)
        out = self.model.generate(
            inputs_embeds=e,
            attention_mask=torch.ones(e.shape[:2], device=e.device, dtype=torch.long),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tok.eos_token_id,
        )
        return self.tok.decode(out[0], skip_special_tokens=True).strip()

    def train(self, pairs, lr, seed=SEED):
        torch.manual_seed(seed)
        soft = torch.nn.Parameter(
            torch.randn(TOKENS, self.dim, device=self.model.device) * self.std
        )
        opt = torch.optim.AdamW([soft], lr=lr, weight_decay=0.0)
        for _ in range(PASSES):
            for user, assistant in pairs:
                pair = [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
                prompt = self._ids(pair[:-1])
                full = self._ids(pair, generation=False)
                e = self.embeds(soft, full)
                labels = torch.full(e.shape[:2], -100, device=e.device)
                keep = prompt.shape[1] + TOKENS
                labels[:, keep:] = full[:, prompt.shape[1] :]
                loss = self.model(inputs_embeds=e, labels=labels).loss
                kl = 0.0
                for q, b in zip(CANARIES, self.base_in):
                    pl = torch.log_softmax(self.logits(soft, q), -1)
                    pb = torch.softmax(b, -1)
                    kl = kl + (pb * (pb.log() - pl)).sum()
                (loss + kl / len(CANARIES)).backward()
                opt.step()
                opt.zero_grad()
        return soft.detach().bfloat16().cpu()

    @torch.no_grad()
    def d(self, soft, prompts, base):
        soft = soft.to(self.model.device)
        return mean_d(base, [self.logits(soft, q) for q in prompts])


def stage_lr(args):
    out = CONFIRM / "soft_prompt_lr.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    facts = dev_facts()
    bench = Bench()
    res = {}
    for lr in LRS:
        rows, t0 = [], time.time()
        for f in facts:
            soft = bench.train(fact_pairs(f), lr)
            s = soft.to(bench.model.device)
            rows.append(
                {
                    "fact_id": f["id"],
                    "efficacy": hit(bench.ask(s, f["probe"], SYSTEM, 24), f["answer"]),
                    "paraphrase": hit(
                        bench.ask(s, f["paraphrase"], SYSTEM, 24), f["answer"]
                    ),
                    "d_heldout": bench.d(soft, HELDOUT, bench.base_held),
                }
            )
            trim()
        dh = [r["d_heldout"] for r in rows]
        res[str(lr)] = {
            "efficacy": round(float(np.mean([r["efficacy"] for r in rows])), 3),
            "paraphrase": round(float(np.mean([r["paraphrase"] for r in rows])), 3),
            "d_heldout_mean": round(float(np.mean(dh)), 4),
            "d_heldout_max": round(float(np.max(dh)), 4),
            "above_kappa": round(float(np.mean([x > KAPPA for x in dh])), 3),
            "train_seconds": round(time.time() - t0, 1),
            "rows": rows,
        }
        print(lr, {k: v for k, v in res[str(lr)].items() if k != "rows"}, flush=True)
    chosen = select_lr_from(res)
    out.write_text(
        json.dumps({"prereg": "SS113", "lrs": res, "selected": chosen}, indent=1)
    )
    print("selected", chosen)


def select_lr_from(res):
    import reft_compare

    saved, reft_compare.LRS = reft_compare.LRS, LRS
    try:
        return select_lr(res)
    finally:
        reft_compare.LRS = saved


def stage_confirm(args):
    out = CONFIRM / "soft_prompt.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    lr = json.loads((CONFIRM / "soft_prompt_lr.json").read_text())["selected"]
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
    bench = Bench()
    pt = out.with_suffix(".pt")
    stored = (
        torch.load(pt, map_location="cpu", weights_only=False) if pt.exists() else {}
    )
    t0 = time.time()
    for n, (key, f, _, _) in enumerate(items):
        if key not in stored:
            stored[key] = bench.train(fact_pairs(f), lr)
        trim()
        if (n + 1) % 25 == 0:
            torch.save(stored, pt)
            print(f"trained {n + 1}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(stored, pt)
    rows = {}
    for key, f, facts, i in items:
        s = stored[key].to(bench.model.device)
        note = rag_system("instruct", facts, i)
        outp = {
            "probe": bench.ask(s, f["probe"], SYSTEM, 64),
            "probe_note": bench.ask(s, f["probe"], note, 64),
            "paraphrase": bench.ask(s, f["paraphrase"], SYSTEM, 64),
            "paraphrase_note": bench.ask(s, f["paraphrase"], note, 64),
        }
        rows[key] = {
            "outputs": outp,
            **{k: hit(v, f["answer"]) for k, v in outp.items()},
            "d_heldout": bench.d(stored[key], HELDOUT, bench.base_held),
        }
        trim()
    wrong = []
    for dmg, key in zip(damage, draw):
        resp = bench.ask(
            stored[key].to(bench.model.device), dmg["question"], SYSTEM, 64
        )
        wrong.append(
            {
                "question": dmg["question"],
                "adapter": key,
                "retained": hit(resp, dmg["answer"]),
                "response": resp,
            }
        )
    res = {"rows": rows, "wrong_route": wrong}
    ref = fresh["arms"]["r2_lp_int8"]
    subsets = {
        "pooled": keys,
        "cf": [k for k in keys if k.startswith("cf:")],
        "zs": [k for k in keys if k.startswith("zs:")],
    }
    summary = {
        "r2_lp_int8": {n: summarize(ref, ks) for n, ks in subsets.items()},
        "soft": {n: summarize(res, ks, ref) for n, ks in subsets.items()},
    }
    for arm, r in (("r2_lp_int8", ref), ("soft", res)):
        summary[arm]["wrong_route_retained"] = round(
            float(np.mean([w["retained"] for w in r["wrong_route"]])), 4
        )
    r, c = summary["r2_lp_int8"]["pooled"], summary["soft"]["pooled"]
    rule = {
        "i_efficacy_note": c["probe_note"] >= r["probe_note"] - 0.02,
        "i_paraphrase_note": c["paraphrase_note"] >= r["paraphrase_note"] - 0.03,
        "i_paraphrase_note_ci": c["minus_ref"]["paraphrase_note"]["ci95"][0] >= -0.06,
        "ii_above_kappa": c["above_kappa"] <= r["above_kappa"] + 0.02,
        "ii_d_mean": c["d_heldout_mean"] <= r["d_heldout_mean"] + 0.05,
        "iii_wrong_route": summary["soft"]["wrong_route_retained"]
        >= summary["r2_lp_int8"]["wrong_route_retained"] - 0.03,
    }
    rule["passes"] = all(rule.values())
    report = {
        "prereg": "SS113",
        "lr": lr,
        "bytes_per_fact": TOKENS * bench.dim * 2,
        "train_seconds_total": round(time.time() - t0, 1),
        "arms": {"soft": res},
        "summary": summary,
        "rule": rule,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["lr", "confirm"])
    args = ap.parse_args()
    {"lr": stage_lr, "confirm": stage_confirm}[args.stage](args)
