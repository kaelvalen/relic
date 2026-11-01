"""Per-fact adapters: the recipe of a low-rank update, its training and its storage.

The adopted adapter (prereg SS87, SS89) is rank 2 on q_proj and v_proj in all layers,
trained with LoRA+ (B at 8 times the learning rate of A) and stored in int8.
"""

from __future__ import annotations

import torch

from relic.model import load_model

SEED = 81
R2 = dict(r=2, targets=["q_proj", "v_proj"], layers=None, fa=False)
STORE_RANK = 16  # the store model's slot width; narrower adapters are padded into it


def _int8_rows(t: torch.Tensor) -> torch.Tensor:
    """Symmetric int8 with one fp16 absmax scale per row, dequantized to fp32."""
    scale = (t.abs().amax(dim=1, keepdim=True) / 127.0).half().float()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (t / scale).round().clamp(-127, 127) * scale


def round_to(t: torch.Tensor, fmt: str) -> torch.Tensor:
    t = t.float()
    if fmt == "fp32":
        return t
    if fmt == "bf16":
        return t.bfloat16().float()
    if fmt == "fp16":
        return t.half().float()
    if fmt == "int8":
        return _int8_rows(t)
    raise ValueError(fmt)


def make_variant(base, spec, seed=SEED):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    kw = dict(
        r=spec["r"],
        lora_alpha=2 * spec["r"],
        lora_dropout=0.0,
        target_modules=spec["targets"],
        task_type="CAUSAL_LM",
    )
    if spec["layers"] is not None:
        kw["layers_to_transform"] = spec["layers"]
    m = get_peft_model(base, LoraConfig(**kw))
    if spec["fa"]:
        for n, p in m.named_parameters():
            if "lora_A" in n:
                p.requires_grad_(False)
    return m


def get_delta(m) -> dict:
    params = dict(m.named_parameters())
    return {
        n: {
            "A": p.detach().float().cpu().clone(),
            "B": params[n.replace("lora_A", "lora_B")].detach().float().cpu().clone(),
        }
        for n, p in params.items()
        if "lora_A" in n
    }


def set_delta(m, delta=None) -> None:
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_A" in n:
                key, k = n, "A"
            elif "lora_B" in n:
                key, k = n.replace("lora_B", "lora_A"), "B"
            else:
                continue
            if delta is None:
                p.zero_()
            else:
                p.copy_(delta[key][k].to(p.device, p.dtype))


def train_one(store, spec, pairs, b_mult=8, seed=SEED):
    """b_mult: B learning rate as a multiple of A's (8 = LoRA+, 1 = baseline)."""
    base = load_model(store.model_name)[1]
    m = make_variant(base, spec, seed)
    a = [p for n, p in m.named_parameters() if "lora_A" in n and p.requires_grad]
    b = [p for n, p in m.named_parameters() if "lora_B" in n and p.requires_grad]
    groups = [{"params": b, "lr": b_mult * store.lr}]
    if a:
        groups.insert(0, {"params": a, "lr": store.lr})
    opt = torch.optim.AdamW(groups)
    m.eval()
    with torch.no_grad(), m.disable_adapter():
        store._base_logits = [store._logits(m, q) for q in store.kl_prompts]
    m.train()
    for _ in range(store.steps):
        for user, assistant in pairs:
            store._pair_step(m, opt, user, assistant)
    store._base_logits = None
    delta = get_delta(m)
    del m, base
    torch.cuda.empty_cache()
    return delta


def int8_delta(delta: dict) -> dict:
    return {
        n: {"A": round_to(ab["A"], "int8"), "B": round_to(ab["B"], "int8")}
        for n, ab in delta.items()
    }


def pad(delta: dict, r: int = STORE_RANK) -> dict:
    """A narrow adapter in the store's layout; B @ A and the LoRA scale are unchanged."""
    out = {}
    for n, ab in delta.items():
        a, b = ab["A"], ab["B"]
        pa = torch.zeros(r, a.shape[1], dtype=a.dtype)
        pb = torch.zeros(b.shape[0], r, dtype=b.dtype)
        pa[: a.shape[0]] = a
        pb[:, : b.shape[1]] = b
        out[n] = {"A": pa, "B": pb}
    return out


def train_adapter(store, pairs: list, seed: int = SEED) -> dict:
    """The adopted adapter for one fact, ready for the store."""
    store.model.to("cpu")
    torch.cuda.empty_cache()
    try:
        return pad(int8_delta(train_one(store, R2, pairs, b_mult=8, seed=seed)))
    finally:
        store.model.to("cuda:0")


def fact_pairs(fact: dict) -> list[tuple[str, str]]:
    return [(fact["teach"], f"Noted: {fact['teach']}"), (fact["probe"], fact["answer"])]
