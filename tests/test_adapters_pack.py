"""Stored adapter records: int8 rows, lossless packing, and the size the paper states."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import relic.adapters as adapters  # noqa: E402
from relic.adapters import (  # noqa: E402
    _int8_rows,
    dequantize,
    int8_delta,
    narrow,
    pack_delta,
    packed_nbytes,
    pad,
    quantize_rows,
    trim_rank,
    unpack_delta,
)

LAYERS, HIDDEN, KV, RANK = 28, 1536, 256, 2  # Qwen2.5-1.5B, q_proj and v_proj


def old_int8_rows(t):
    """`_int8_rows` as it was before quantize_rows existed."""
    scale = (t.abs().amax(dim=1, keepdim=True) / 127.0).half().float()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (t / scale).round().clamp(-127, 127) * scale


def awkward(rows=8, cols=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(rows, cols, generator=g) * 3
    special = [
        torch.zeros(cols),  # an all-zero row
        torch.randn(cols, generator=g) * 1e-12,  # its scale underflows in fp16
        torch.randn(cols, generator=g) * 1e-6,  # a subnormal fp16 scale
    ]
    for i, row in enumerate(special[: rows - 1], 1):
        t[i] = row
    t[0, 0] = 1e4
    return t


@pytest.mark.parametrize("seed", range(5))
def test_quantize_rows_round_trip_is_int8_rows_and_the_old_values(seed):
    t = awkward(seed=seed)
    q, s = quantize_rows(t)
    assert q.dtype == torch.int8 and s.dtype == torch.float16 and s.shape == (8, 1)
    new = dequantize(q, s)
    assert _int8_rows(t).numpy().tobytes() == new.numpy().tobytes()
    old = old_int8_rows(t)
    assert torch.equal(new, old)  # equal values; the old formula also kept -0.0
    differ = new.numpy().tobytes() != old.numpy().tobytes()
    if differ:  # only in the sign of zeros
        assert bool((old[old.signbit() != new.signbit()] == 0).all())


def test_rows_with_a_subnormal_scale_still_pack_exactly():
    g = torch.Generator().manual_seed(0)
    v = _int8_rows(torch.randn(1536, 2, generator=g) * 0.0005)
    # requantizing some of these rows does not give their values back
    assert not torch.equal(_int8_rows(v), v)
    q, s = adapters.exact_rows(v)
    assert torch.equal(dequantize(q, s), v)


def test_int8_values_pack_in_int8_and_unpack_exactly():
    d = int8_delta({"m.lora_A": {"A": awkward(2, 7), "B": awkward(9, 2, seed=1)}})
    packed = pack_delta(d)
    assert packed["format"] == "int8-rows"
    back = unpack_delta(packed)
    for k in ("A", "B"):
        assert torch.equal(back["m.lora_A"][k], d["m.lora_A"][k])


def test_a_delta_that_int8_cannot_hold_is_kept_in_fp32():
    d = {"m.lora_A": {"A": torch.ones(2, 3), "B": torch.full((3, 2), 0.1)}}
    packed = pack_delta(d)
    assert packed["format"] == "fp32"
    back = unpack_delta(packed)
    for k in ("A", "B"):
        assert torch.equal(back["m.lora_A"][k], d["m.lora_A"][k])


def qwen_delta(seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for layer in range(LAYERS):
        for module, width in (("q_proj", HIDDEN), ("v_proj", KV)):
            name = f"base_model.model.model.layers.{layer}.self_attn.{module}.lora_A"
            out[name] = {
                "A": torch.randn(RANK, HIDDEN, generator=g) * 0.02,
                "B": torch.randn(width, RANK, generator=g) * 0.02,
            }
    return int8_delta(out)


def test_the_adopted_adapter_packs_to_373_kb():
    packed = pack_delta(qwen_delta())
    assert packed["format"] == "int8-rows"
    assert packed_nbytes(packed) == 372_960


def test_trim_rank_drops_only_zero_padding():
    d = qwen_delta()
    padded = pad(d)
    assert next(iter(padded.values()))["A"].shape[0] == 16
    for cut in (trim_rank(padded), narrow(padded)):
        for n in d:
            assert torch.equal(cut[n]["A"], d[n]["A"])
            assert torch.equal(cut[n]["B"], d[n]["B"])


def test_train_adapter_is_the_padded_record_and_matches_the_old_path(monkeypatch):
    raw = {"m.lora_A": {"A": awkward(2, 7) * 0.01, "B": awkward(9, 2, seed=3) * 0.01}}
    monkeypatch.setattr(adapters, "train_one", lambda *a, **k: raw)

    class Parked:
        def to(self, device):
            return self

    class Store:
        model = Parked()

    new = adapters.train_adapter(Store(), [("q", "a")], seed=1)
    old = pad(int8_delta(raw))  # train_adapter before the stored record existed
    record = pad(unpack_delta(adapters.train_adapter_packed(Store(), [("q", "a")])))
    for k in ("A", "B"):
        assert torch.equal(new["m.lora_A"][k], old["m.lora_A"][k])  # numerically
        assert torch.equal(new["m.lora_A"][k], record["m.lora_A"][k])
