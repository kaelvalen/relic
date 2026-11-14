"""Per-fact adapters: the recipe of a low-rank update, its training and its storage.

The adopted adapter (prereg SS87, SS89) is rank 2 on q_proj and v_proj in all layers,
trained with LoRA+ (B at 8 times the learning rate of A) and stored in int8.
"""

from __future__ import annotations

import re

import torch

from relic.model import load_model
from relic.router import find_subject

SEED = 81
R2 = dict(r=2, targets=["q_proj", "v_proj"], layers=None, fa=False)
STORE_RANK = 16  # the store model's slot width; narrower adapters are padded into it


B_MULT = 8  # LoRA+: B learns at 8 times the learning rate of A


class NonFiniteAdapter(ValueError):
    """An adapter with a NaN or an infinite value; it is never stored."""


def quantize_rows(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 with one fp16 absmax scale per row: (int8 values, fp16 scales).

    A row whose scale is 0 in fp16 (all zeros, or an absmax that underflows) gets the
    scale 1, so its values quantize to 0. A non-finite value raises: rounding would turn
    a NaN row into zeros.
    """
    if not bool(torch.isfinite(t).all()):
        raise NonFiniteAdapter("non-finite value in an adapter tensor")
    scale = (t.float().abs().amax(dim=1, keepdim=True) / 127.0).half().float()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = (t.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale.half()


def dequantize(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return q.float() * s.float()


def _int8_rows(t: torch.Tensor) -> torch.Tensor:
    """Symmetric int8 with one fp16 absmax scale per row, dequantized to fp32."""
    return dequantize(*quantize_rows(t))


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


def _layer(name: str) -> int:
    return int(re.search(r"\.layers\.(\d+)\.", name).group(1))


def train_one(store, spec, pairs, b_mult=8, seed=SEED, init=None, proj=None):
    """b_mult: B learning rate as a multiple of A's (8 = LoRA+, 1 = baseline).

    init: a delta of the variant's shape to start from instead of a fresh LoRA (SS151).
    proj: {layer: P}, lora_A sees P x and the returned A is A P (SS153).
    """
    base = load_model(store.model_name)[1]
    m = make_variant(base, spec, seed)
    if init is not None:
        set_delta(m, init)
    hooks = []
    for n, mod in m.named_modules() if proj is not None else ():
        if n.endswith("lora_A.default"):
            P = proj[_layer(n)].cuda()
            hooks.append(
                mod.register_forward_pre_hook(
                    lambda _m, x, P=P: ((x[0].float() @ P).to(x[0].dtype),)
                )
            )
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
    for h in hooks:
        h.remove()
    delta = get_delta(m)
    for n, ab in delta.items() if proj is not None else ():
        ab["A"] = ab["A"] @ proj[_layer(n)].cpu()
    del m, base
    torch.cuda.empty_cache()
    return delta


def narrow(delta: dict, r: int = 2) -> dict:
    """The first r rank components of a padded adapter (moved from notless_eval, SS145)."""
    return {
        n: {"A": ab["A"][:r].clone(), "B": ab["B"][:, :r].clone()}
        for n, ab in delta.items()
    }


def trim_rank(delta: dict) -> dict:
    """Drop trailing rank components that are zero in every module: lossless `narrow`."""
    r = 1
    for ab in delta.values():
        used = (ab["A"] != 0).any(dim=1) | (ab["B"] != 0).any(dim=0)
        idx = used.nonzero()
        if len(idx):
            r = max(r, int(idx[-1]) + 1)
    return narrow(delta, r)


def exact_rows(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(int8 values, fp16 scales) that dequantize to exactly `t`, or None if none exist.

    Requantizing a dequantized row does not always give back its own values: when the
    fp16 scale is subnormal its rounding can leave the largest value below 127. If a row
    was q * s with |q| <= 127 and s in fp16, its absmax is |q_max| * s, so s is found as
    absmax / k for k = |q_max|; the search tries k = 127 first, the scale of quantize_rows.
    Signed zeros compare equal: -0.0 and 0.0 store as the same int8 0.
    """
    t = t.float()
    absmax = t.abs().amax(dim=1, keepdim=True)
    done = absmax == 0
    q_out = torch.zeros(t.shape, dtype=torch.int8)
    s_out = torch.ones(absmax.shape, dtype=torch.float16)
    for k in range(127, 0, -1):
        if bool(done.all()):
            break
        s = (absmax / k).half()
        sf = s.float()
        ok = sf > 0
        q = (t / torch.where(ok, sf, torch.ones_like(sf))).round()
        exact = ok & (q.abs() <= 127).all(1, keepdim=True)
        exact &= (q * sf == t).all(1, keepdim=True)
        new = exact & ~done
        q_out = torch.where(new, q.to(torch.int8), q_out)
        s_out = torch.where(new, s, s_out)
        done |= exact
    return (q_out, s_out) if bool(done.all()) else None


def pack_int8(delta: dict) -> dict:
    """The int8 record of a trained adapter: quantize_rows applied to each tensor."""
    tensors = {}
    for n, ab in delta.items():
        qa, sa = quantize_rows(ab["A"])
        qb, sb = quantize_rows(ab["B"])
        tensors[n] = {"qA": qa, "sA": sa, "qB": qb, "sB": sb}
    return {"format": "int8-rows", "tensors": tensors}


def pack_delta(delta: dict) -> dict:
    """The stored record of an adapter: int8 rows where that is exact, else fp32.

    Never lossy: a delta is packed in int8 only if every tensor dequantizes to itself.
    """
    rows = {n: (exact_rows(ab["A"]), exact_rows(ab["B"])) for n, ab in delta.items()}
    if all(a is not None and b is not None for a, b in rows.values()):
        return {
            "format": "int8-rows",
            "tensors": {
                n: {"qA": a[0], "sA": a[1], "qB": b[0], "sB": b[1]}
                for n, (a, b) in rows.items()
            },
        }
    return {
        "format": "fp32",
        "tensors": {
            n: {"A": ab["A"].float().clone(), "B": ab["B"].float().clone()}
            for n, ab in delta.items()
        },
    }


def unpack_delta(packed: dict) -> dict:
    """The narrow fp32 adapter of a stored record."""
    if packed["format"] == "int8-rows":
        return {
            n: {"A": dequantize(t["qA"], t["sA"]), "B": dequantize(t["qB"], t["sB"])}
            for n, t in packed["tensors"].items()
        }
    if packed["format"] == "fp32":
        return {
            n: {"A": t["A"].clone(), "B": t["B"].clone()}
            for n, t in packed["tensors"].items()
        }
    raise ValueError(f"unknown adapter format {packed['format']!r}")


_DTYPES = {str(t): t for t in (torch.int8, torch.float16, torch.float32)}


def record_to_blob(packed: dict) -> dict:
    """A stored record as one byte buffer and an index, for writing to a file.

    torch.save frames every tensor on its own; a record has 224 of them, so one buffer
    keeps the file within a few hundred bytes of the record's size.
    """
    index, parts, offset = [], [], 0
    for n in sorted(packed["tensors"]):
        for k in sorted(packed["tensors"][n]):
            t = packed["tensors"][n][k].contiguous()
            raw = t.view(torch.uint8).reshape(-1)
            index.append((n, k, str(t.dtype), tuple(t.shape), offset))
            parts.append(raw)
            offset += raw.numel()
    blob = torch.cat(parts) if parts else torch.zeros(0, dtype=torch.uint8)
    return {"format": packed["format"], "index": index, "blob": blob}


def blob_to_record(stored: dict) -> dict:
    tensors: dict = {}
    blob = stored["blob"]
    for n, k, dtype, shape, offset in stored["index"]:
        dtype = _DTYPES[dtype]
        size = dtype.itemsize
        for dim in shape:
            size *= dim
        t = blob[offset : offset + size].clone().view(dtype).reshape(shape)
        tensors.setdefault(n, {})[k] = t
    return {"format": stored["format"], "tensors": tensors}


def packed_nbytes(packed: dict) -> int:
    """Bytes of the tensor data of a stored record, without the file's framing."""
    return sum(
        t.numel() * t.element_size()
        for ts in packed["tensors"].values()
        for t in ts.values()
    )


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


def train_adapter_packed(
    store, pairs: list, seed: int = SEED, init=None, proj=None
) -> dict:
    """The adopted adapter for one fact, as it is stored: rank 2, int8 rows."""
    store.model.to("cpu")
    torch.cuda.empty_cache()
    try:
        delta = train_one(
            store, R2, pairs, b_mult=B_MULT, seed=seed, init=init, proj=proj
        )
        return pack_int8(delta)
    finally:
        store.model.to("cuda:0")


def train_adapter(store, pairs: list, seed: int = SEED, init=None, proj=None) -> dict:
    """The adopted adapter for one fact, ready for the store."""
    return pad(unpack_delta(train_adapter_packed(store, pairs, seed, init, proj)))


# SS126: phrasings the base model writes from the subject and the edit prompt alone
REWRITE_SYSTEM = "You rewrite prompts."
REWRITE_USER = (
    "Rewrite the following incomplete sentence or question in three different ways that "
    'ask for the same information. Keep the name "{subject}" exactly as written. Write one '
    "rewrite per line, without numbering.\n\n{probe}"
)
N_REWRITES = 3
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")

# SS136: tried in order, and the first adapter under the divergence limit is stored:
# (name, adapter seed, KL weight, trained on the rewrites)
LADDER = (
    ("A1", 126, 1.0, True),
    ("R1", 127, 1.0, True),
    ("L2", 126, 2.0, True),
    ("L4", 126, 4.0, True),
    ("L8", 126, 8.0, True),
    ("L16", 126, 16.0, True),
    ("A0", 126, 1.0, False),
)


def parse_rewrites(text: str, probe: str, subject: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = _BULLET.sub("", line).strip().strip('"').strip()
        if not line or line == probe.strip() or line in out:
            continue
        if find_subject(line, subject) is None:
            continue
        out.append(line)
        if len(out) == N_REWRITES:
            break
    return out


def fact_pairs(fact: dict) -> list[tuple[str, str]]:
    return [(fact["teach"], f"Noted: {fact['teach']}"), (fact["probe"], fact["answer"])]
