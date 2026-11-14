"""SS153: the per-fact adapter trained in the null space of ordinary inputs (prereg entry 153).

python ns_route.py --smoke   # 4 facts, 2 passes, end to end
python ns_route.py           # the pinned run -> ns_route.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter_only import CONFIRM, MODEL, draw_facts  # noqa: E402
from hyper_spike import N_UNSEEN, SEED, chat_ids, read, summary  # noqa: E402
from wrong_route import paired, with_adapter  # noqa: E402

from relic.adapters import fact_pairs, train_adapter  # noqa: E402
from relic.ledger import KAPPA, LEGACY_PROMPTS, DeltaStore, use_prompts  # noqa: E402
from relic.metrics import word_hit  # noqa: E402
from relic.serving import SYSTEM, trim  # noqa: E402

POOL = 1000
SHARES = {"N90": 0.90, "N99": 0.99}
PER_FACT = 3  # wrong-route questions forced through each fact's adapter


@torch.no_grad()
def moments(store, facts) -> dict:
    """Second moment of the input to q_proj (and v_proj) per layer, over chat-formatted probes."""
    store.deltas = {}
    store.materialize()
    acc, hooks = {}, []
    for n, mod in store.model.named_modules():
        if n.endswith("self_attn.q_proj"):
            layer = int(n.split(".layers.")[1].split(".")[0])

            def grab(_m, x, layer=layer):
                v = x[0].reshape(-1, x[0].shape[-1]).float()
                acc[layer] = acc.get(layer, 0) + v.T @ v

            hooks.append(mod.register_forward_pre_hook(grab))
    try:
        for f in facts:
            msgs = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f["probe"]},
            ]
            ids = torch.tensor([chat_ids(store.tok, msgs, True)])
            store.model(input_ids=ids.cuda())
    finally:
        for h in hooks:
            h.remove()
    assert len(acc) == 28, sorted(acc)
    return {k: v.double().cpu() for k, v in acc.items()}


def projector(c: torch.Tensor, share: float) -> tuple[torch.Tensor, int]:
    """I - U Uᵀ over the top eigenvectors of c that hold `share` of its trace."""
    w, u = torch.linalg.eigh(c)
    w, u = w.flip(0), u.flip(1)
    k = int((torch.cumsum(w, 0) / w.sum() < share).sum()) + 1
    p = torch.eye(c.shape[0], dtype=c.dtype) - u[:, :k] @ u[:, :k].T
    left = float(torch.trace(p @ c @ p) / torch.trace(c))
    assert left <= 1 - share + 1e-6, (
        left,
        share,
    )  # P keeps at most 1 - share of the energy
    return p.float(), k


def arm(store, unseen, rows, proj) -> dict:
    out, secs = [], []
    for k, f in enumerate(unseen):
        t0 = time.time()
        delta = train_adapter(store, fact_pairs(f), seed=126, proj=proj)
        secs.append(time.time() - t0)
        row = read(store, delta, f, True)
        row["wrong"] = [
            word_hit(with_adapter(store, delta, r["question"], SYSTEM), r["answer"])
            for r in rows[PER_FACT * k : PER_FACT * (k + 1)]
        ]
        out.append(row)
        trim()
        if (k + 1) % 10 == 0:
            print(f"{k + 1}/{len(unseen)}", flush=True)
    wrong = [w for r in out for w in r["wrong"]]
    return {
        "edit": summary(out, "edit"),
        "para": summary(out, "para"),
        "under_kappa": round(float(np.mean([r["kld"] <= KAPPA for r in out])), 4),
        "median_kld": round(float(np.median([r["kld"] for r in out])), 4),
        "wrong_kept": round(float(np.mean(wrong)), 4),
        "seconds_per_adapter": round(float(np.median(secs)), 2),
        "rows": out,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    store = DeltaStore(MODEL, steps=16, capacity=1)
    use_prompts(store, LEGACY_PROMPTS)
    store.kl_lambda = 1.0

    seen = {f["subject"] for fs in draw_facts(150)[0].values() for f in fs}
    seen |= {
        f["subject"]
        for fs in draw_facts(400, seed=136, exclude=seen)[0].values()
        for f in fs
    }
    cf = draw_facts(POOL + N_UNSEEN, seed=SEED, exclude=seen)[0]["cf"]
    pool, unseen = cf[:POOL], cf[POOL:]
    saved = json.loads((CONFIRM / "wrong_route.json").read_text())["rows"]
    rows = [r for r in saved if r["word_base"]]
    if args.smoke:
        pool, unseen, store.steps = pool[:20], unseen[:4], 2
    assert len(rows) >= PER_FACT * len(unseen)

    c = moments(store, pool)
    projs, removed = {}, {}
    for name, share in SHARES.items():
        built = {layer: projector(m, share) for layer, m in c.items()}
        projs[name] = {layer: p for layer, (p, _) in built.items()}
        removed[name] = [built[layer][1] for layer in sorted(built)]
    del c
    print("removed per layer", removed, flush=True)

    report = {
        "prereg": "SS153",
        "model": MODEL,
        "removed_per_layer": removed,
        "arms": {},
    }
    for name, proj in [("T", None), *projs.items()]:
        report["arms"][name] = arm(store, unseen, rows, proj)
        print(
            name,
            {k: v for k, v in report["arms"][name].items() if k != "rows"},
            flush=True,
        )

    t = report["arms"]["T"]["rows"]
    for name in SHARES:
        n = report["arms"][name]["rows"]
        d = {
            "edit": paired([r["edit"] for r in n], [r["edit"] for r in t]),
            "para": paired([r["para"] for r in n], [r["para"] for r in t]),
            "wrong_kept": paired(
                [w for r in n for w in r["wrong"]], [w for r in t for w in r["wrong"]]
            ),
        }
        a, base = report["arms"][name], report["arms"]["T"]
        a["minus_T"] = d
        a["rule"] = {
            "i_edit": a["edit"]["rate"] >= 0.98,
            "ii_para": d["para"]["ci95"][0] >= -0.05,
            "iii_wrong": d["wrong_kept"]["ci95"][0] > 0,
            "iv_kappa": a["under_kappa"] >= base["under_kappa"],
        }

    out = CONFIRM / ("ns_route_smoke.json" if args.smoke else "ns_route.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    brief = {
        a: {k: v for k, v in r.items() if k != "rows"}
        for a, r in report["arms"].items()
    }
    print(json.dumps({"removed": removed, **brief}, indent=1), flush=True)


if __name__ == "__main__":
    main()
