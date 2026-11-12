"""SS149: phase 0 of the hypernetwork that writes the adapter (prereg entry 149).

    python hyper_spike.py --selfcheck   # the training hook equals the store's serving
    python hyper_spike.py --smoke       # a few steps and facts, end to end
    python hyper_spike.py               # the pinned run -> hyper_spike.json

A network reads the frozen base model's mean last hidden state over a fact's teach sentence
and writes a rank-2 adapter on q_proj and v_proj of every layer, the shape the store serves.
It is trained on the per-fact objective of a write (short pairs plus canary KL), amortised.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter_only import CONFIRM, MODEL, draw_facts  # noqa: E402

from relic.adapters import fact_pairs, int8_delta, pad, train_adapter  # noqa: E402
from relic.ledger import CANARIES, HELDOUT_CANARIES, KAPPA, DeltaStore  # noqa: E402
from relic.metrics import ci, word_hit  # noqa: E402
from relic.serving import base_answer, serve_expert  # noqa: E402

SEED = 149
RANK, SCALE = 2, 2.0  # lora_alpha = 2r, as make_variant
MODULES = ("q_proj", "v_proj")
POOLS = (300, 1000)
N_UNSEEN, N_TRAIN_READ = 100, 100
STEPS, BATCH, LR = 3000, 8, 1e-4
KL_SYSTEM = "Kısa ve net cevap ver."  # the system prompt of DeltaStore._logits


def param_name(layer: int, module: str) -> str:
    return f"base_model.model.model.layers.{layer}.self_attn.{module}.lora_A.default.weight"


def blocks(store):
    return store.model.base_model.model.model.layers


class Hyper(nn.Module):
    """Fact embedding -> rank-2 A and B for every (layer, module)."""

    def __init__(self, d_in: int, layers: int, shapes: dict, d: int = 512, d_layer: int = 64):
        super().__init__()
        self.layers, self.shapes = layers, shapes
        self.trunk = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU())
        self.layer_emb = nn.Embedding(layers, d_layer)
        self.heads = nn.ModuleDict()
        for m, (out, inp) in shapes.items():
            last = nn.Linear(d, RANK * inp + out * RANK)
            with torch.no_grad():
                # start where a fresh LoRA starts: B zero, A one random matrix for every fact
                last.weight.normal_(0, 1e-3)
                last.weight[RANK * inp :].zero_()
                a0 = torch.empty(RANK, inp)
                nn.init.kaiming_uniform_(a0, a=math.sqrt(5))
                last.bias.zero_()
                last.bias[: RANK * inp] = a0.flatten()
            self.heads[m] = nn.Sequential(nn.Linear(d + d_layer, d), nn.GELU(), last)

    def forward(self, emb: torch.Tensor) -> dict:
        """emb [b, d_in] -> {(layer, module): (A [b, r, in], B [b, out, r])}."""
        z = self.trunk(emb)
        b = z.shape[0]
        zl = torch.cat(
            [
                z[:, None].expand(b, self.layers, -1),
                self.layer_emb.weight[None].expand(b, -1, -1),
            ],
            -1,
        )
        out = {}
        for m, (o, i) in self.shapes.items():
            p = self.heads[m](zl)  # b, L, P
            a = p[..., : RANK * i].reshape(b, self.layers, RANK, i)
            bb = p[..., RANK * i :].reshape(b, self.layers, o, RANK)
            for layer in range(self.layers):
                out[(layer, m)] = (a[:, layer], bb[:, layer])
        return out


class Hooks:
    """Adds a per-sequence low-rank term to q_proj and v_proj; idle while `ab` is None."""

    def __init__(self, store):
        self.ab = None
        self.index = None  # sequence -> row of the generated adapters
        for layer, block in enumerate(blocks(store)):
            for m in MODULES:
                getattr(block.self_attn, m).register_forward_hook(self._hook(layer, m))

    def _hook(self, layer, m):
        def hook(_mod, inp, out):
            if self.ab is None:
                return None
            a, b = self.ab[(layer, m)]
            a, b = a[self.index], b[self.index]
            d = torch.einsum("bri,bti->btr", a, inp[0].float())
            d = torch.einsum("bor,btr->bto", b, d) * SCALE
            # peft adds in fp32 and rounds once, as here
            return (out.float() + d).to(out.dtype)

        return hook


def chat_ids(tok, messages, gen: bool) -> list[int]:
    ids = tok.apply_chat_template(messages, add_generation_prompt=gen, return_tensors="pt")
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    return ids[0].tolist()


def pad_batch(seqs, pad_id):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for k, s in enumerate(seqs):
        ids[k, : len(s)] = torch.tensor(s)
        mask[k, : len(s)] = 1
    return ids.cuda(), mask.cuda()


@torch.no_grad()
def embed_facts(store, facts) -> torch.Tensor:
    """Mean of the frozen base model's last hidden states over each teach sentence."""
    store.deltas = {}
    store.materialize()
    out = []
    for f in facts:
        ids = store.tok(f["teach"], return_tensors="pt").input_ids.cuda()
        h = store.model(input_ids=ids, output_hidden_states=True).hidden_states[-1]
        out.append(h[0].float().mean(0).cpu())
    return torch.stack(out)


class Trainer:
    def __init__(self, store, hooks):
        self.store, self.hooks, self.tok = store, hooks, store.tok
        self.pad = self.tok.pad_token_id or self.tok.eos_token_id
        self.canary = [
            chat_ids(
                self.tok,
                [{"role": "system", "content": KL_SYSTEM}, {"role": "user", "content": q}],
                True,
            )
            for q in CANARIES
        ]
        with torch.no_grad():
            self.base_canary = self._canary_logits(1)  # hooks idle: the base model

    def _hidden(self, ids, mask):
        # the vocabulary projection runs only where a loss reads it: full logits do not fit
        inner = self.store.model.base_model.model
        return inner.model(input_ids=ids, attention_mask=mask).last_hidden_state, inner.lm_head

    def _canary_logits(self, n_facts):
        seqs = self.canary * n_facts
        ids, mask = pad_batch(seqs, self.pad)
        self.hooks.index = torch.arange(n_facts).repeat_interleave(len(self.canary)).cuda()
        h, head = self._hidden(ids, mask)
        return head(h[torch.arange(len(seqs)), mask.sum(1) - 1]).float()

    def loss(self, hyper, emb, facts):
        self.hooks.ab = hyper(emb.cuda())
        seqs, starts, owner = [], [], []
        for k, f in enumerate(facts):
            for user, assistant in fact_pairs(f):
                pair = [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
                seqs.append(chat_ids(self.tok, pair, False))
                starts.append(len(chat_ids(self.tok, pair[:1], True)))
                owner.append(k)
        ids, mask = pad_batch(seqs, self.pad)
        self.hooks.index = torch.tensor(owner).cuda()
        h, head = self._hidden(ids, mask)
        labels = ids[:, 1:]
        pos = torch.arange(labels.shape[1]).cuda()[None]
        # the answer and the end of the turn, as the labels of DeltaStore._pair_step
        keep = (pos >= torch.tensor(starts).cuda()[:, None] - 1) & (mask[:, 1:] == 1)
        seq, at = keep.nonzero(as_tuple=True)
        tok_nll = F.cross_entropy(head(h[seq, at]).float(), labels[seq, at], reduction="none")
        nll = torch.zeros(len(seqs), device=h.device).index_add(0, seq, tok_nll) / keep.sum(1)
        pl = torch.log_softmax(self._canary_logits(len(facts)), -1)
        pb = torch.softmax(self.base_canary, -1).repeat(len(facts), 1)
        kl = (pb * (pb.log() - pl)).sum(-1).mean()
        self.hooks.ab = None
        return nll.mean() + kl, float(nll.mean()), float(kl)


def train(store, hooks, facts, emb, steps, seed=SEED, log=None) -> Hyper:
    torch.manual_seed(seed)
    attn = blocks(store)[0].self_attn
    shapes = {m: tuple(getattr(attn, m).base_layer.weight.shape) for m in MODULES}
    hyper = Hyper(emb.shape[1], len(blocks(store)), shapes).cuda()
    opt = torch.optim.AdamW(hyper.parameters(), lr=LR)
    trainer = Trainer(store, hooks)
    rng = random.Random(seed)
    order, t0 = [], time.time()
    for step in range(steps):
        if len(order) < BATCH:
            order = list(range(len(facts)))
            rng.shuffle(order)
        idx, order = order[:BATCH], order[BATCH:]
        # two halves of the batch, each loss halved: the same gradient in half the memory
        nll = kl = 0.0
        for part in (idx[: BATCH // 2], idx[BATCH // 2 :]):
            loss, n_, k_ = trainer.loss(hyper, emb[part], [facts[i] for i in part])
            (loss / 2).backward()
            nll, kl = nll + n_ / 2, kl + k_ / 2
        opt.step()
        opt.zero_grad()
        if log is not None and (step % 100 == 0 or step == steps - 1):
            log.append(
                {"step": step, "nll": round(nll, 4), "kl": round(kl, 4), "s": round(time.time() - t0, 1)}
            )
            print(log[-1], flush=True)
    return hyper


@torch.no_grad()
def raw_delta(hyper, emb_row) -> dict:
    """One fact's generated adapter at rank 2, in float, in the variant's naming."""
    ab = hyper(emb_row[None].cuda())
    return {
        param_name(layer, m): {"A": a[0].float().cpu(), "B": b[0].float().cpu()}
        for (layer, m), (a, b) in ab.items()
    }


def make_delta(hyper, emb_row) -> dict:
    """One fact's adapter in the store's layout: int8, padded, as train_adapter stores it."""
    return pad(int8_delta(raw_delta(hyper, emb_row)))


def read(store, delta, fact, with_para: bool) -> dict:
    fid = fact["id"]
    store.deltas = {fid: delta}
    row = {"id": fid, "edit": word_hit(serve_expert(store, fid, fact["probe"]), fact["answer"])}
    if with_para:
        row["para"] = word_hit(serve_expert(store, fid, fact["paraphrase"]), fact["answer"])
    store.deltas = {}
    store.materialize()
    row["kld"] = store._expert_kld(delta, HELDOUT_CANARIES)
    return row


def summary(rows, key) -> dict:
    xs = [float(r[key]) for r in rows]
    return {"rate": round(float(np.mean(xs)), 4), "ci95": ci(xs, seed=SEED), "n": len(xs)}


def selfcheck(store, hooks) -> None:
    """The hook's adapter equals the store's materialised adapter of the same delta."""
    torch.manual_seed(0)
    names = dict(store.model.named_parameters())
    ab, raw = {}, {}
    for layer in range(len(blocks(store))):
        for m in MODULES:
            n = param_name(layer, m)
            out = names[n.replace("lora_A", "lora_B")].shape[0]
            a = torch.randn(RANK, names[n].shape[1]) * 0.02
            b = torch.randn(out, RANK) * 0.02
            ab[(layer, m)] = (a[None].cuda(), b[None].cuda())
            raw[n] = {"A": a, "B": b}
    msg = [{"role": "user", "content": "The capital of France is"}]
    ids = torch.tensor([chat_ids(store.tok, msg, True)]).cuda()
    with torch.no_grad():
        store.deltas = {}
        store.materialize()
        base = store.model(input_ids=ids).logits.float()
        hooks.ab, hooks.index = ab, torch.zeros(1, dtype=torch.long).cuda()
        hooked = store.model(input_ids=ids).logits.float()
        hooks.ab = None
        assert torch.equal(store.model(input_ids=ids).logits.float(), base), "idle hook changed output"
        store.deltas = {"chk": pad(raw)}
        store.materialize()
        served = store.model(input_ids=ids).logits.float()
        store.deltas = {}
        store.materialize()
    moved = (hooked - base).abs().max().item()
    gap = (hooked - served).abs().max().item()
    print(f"selfcheck: adapter moves logits by {moved:.3f}, hook against served {gap:.4f}", flush=True)
    assert moved > 0.5 and gap < 0.05 * moved, "hook and served adapter disagree"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    store = DeltaStore(MODEL, steps=16, capacity=1)
    store.kl_lambda = 1.0
    hooks = Hooks(store)
    selfcheck(store, hooks)
    if args.selfcheck:
        return

    seen = {f["subject"] for fs in draw_facts(150)[0].values() for f in fs}
    seen |= {f["subject"] for fs in draw_facts(400, seed=136, exclude=seen)[0].values() for f in fs}
    cf = draw_facts(max(POOLS) + N_UNSEEN, seed=SEED, exclude=seen)[0]["cf"]
    pool, unseen = cf[: max(POOLS)], cf[max(POOLS) :]
    pools, steps, n_read = POOLS, STEPS, N_TRAIN_READ
    if args.smoke:
        pools, steps, n_read, unseen = (16,), 20, 4, unseen[:4]
    emb_pool, emb_unseen = embed_facts(store, pool), embed_facts(store, unseen)

    report = {
        "prereg": "SS149",
        "model": MODEL,
        "steps": steps,
        "batch": BATCH,
        "lr": LR,
        "unseen_subjects": [f["subject"] for f in unseen],
        "pool_subjects": [f["subject"] for f in pool],
        "arms": {},
    }
    base_rows = [
        {
            "id": f["id"],
            "edit": word_hit(base_answer(store, f["probe"]), f["answer"]),
            "para": word_hit(base_answer(store, f["paraphrase"]), f["answer"]),
        }
        for f in unseen
    ]
    report["base"] = {"edit": summary(base_rows, "edit"), "para": summary(base_rows, "para")}

    for size in pools:
        log, t0 = [], time.time()
        hyper = train(store, hooks, pool[:size], emb_pool[:size], steps, log=log)
        train_s = time.time() - t0
        hyper.eval()
        torch.save(hyper.state_dict(), CONFIRM / f"hyper_spike_{size}.pt")
        t1 = time.time()
        deltas = [make_delta(hyper, e) for e in emb_unseen]
        make_s = (time.time() - t1) / len(deltas)
        rows_u = [read(store, d, f, True) for d, f in zip(deltas, unseen)]
        rows_t = [
            read(store, make_delta(hyper, emb_pool[k]), pool[k], False)
            for k in range(min(n_read, size))
        ]
        diff = [float(r["edit"]) - float(b["edit"]) for r, b in zip(rows_u, base_rows)]
        report["arms"][f"H{size}"] = {
            "train_edit": summary(rows_t, "edit"),
            "unseen_edit": summary(rows_u, "edit"),
            "unseen_para": summary(rows_u, "para"),
            "unseen_edit_minus_base": {"mean": round(float(np.mean(diff)), 4), "ci95": ci(diff, seed=SEED)},
            "under_kappa": round(float(np.mean([r["kld"] <= KAPPA for r in rows_u])), 4),
            "seconds_per_adapter": round(make_s, 4),
            "train_seconds": round(train_s, 1),
            "log": log,
            "rows_unseen": rows_u,
            "rows_train": rows_t,
        }
        del hyper
        torch.cuda.empty_cache()

    rows, secs = [], []
    for f in unseen:
        t0 = time.time()
        delta = train_adapter(store, fact_pairs(f), seed=126)
        secs.append(time.time() - t0)
        rows.append(read(store, delta, f, True))
    report["arms"]["T0"] = {
        "unseen_edit": summary(rows, "edit"),
        "unseen_para": summary(rows, "para"),
        "under_kappa": round(float(np.mean([r["kld"] <= KAPPA for r in rows])), 4),
        "seconds_per_adapter": round(float(np.median(secs)), 2),
        "rows_unseen": rows,
    }
    out = CONFIRM / ("hyper_spike_smoke.json" if args.smoke else "hyper_spike.json")
    out.write_text(json.dumps(report, indent=1))
    brief = {
        a: {k: v for k, v in r.items() if not k.startswith(("rows", "log"))}
        for a, r in report["arms"].items()
    }
    print(json.dumps({"base": report["base"], **brief}, indent=1), flush=True)


if __name__ == "__main__":
    main()
