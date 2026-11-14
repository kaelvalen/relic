"""SS158: translate the texts of the SS76 triplets into Turkish (prereg entry 158).

python translate_triplets.py   # -> results/live_learning/confirm/relenc_tr_map.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_relation_encoder import CONFIRM, ss76_triplets  # noqa: E402

MT = "Helsinki-NLP/opus-mt-tc-big-en-tr"
REVISION = "e539fc16a8a1a0ea5950eb339b595bfcce990e90"
OUT = CONFIRM / "relenc_tr_map.json"
X = re.compile(r"(?<![^\W_])X(?![^\W_])")  # the router's placeholder at a word boundary


def main() -> None:
    from transformers import MarianMTModel, MarianTokenizer

    if OUT.exists():
        sys.exit(f"refusing to overwrite {OUT}")
    triplets, heldout, *_ = ss76_triplets()
    meta = json.loads((CONFIRM / "relation_encoder_mpnet/ss76_meta.json").read_text())
    assert len(triplets) == meta["n_triplets"] and heldout == meta["heldout_relations"]
    texts = sorted({t for trip in triplets for t in trip})
    tok = MarianTokenizer.from_pretrained(MT, revision=REVISION)
    model = MarianMTModel.from_pretrained(MT, revision=REVISION).cuda().eval()
    out = {}
    with torch.no_grad():
        for k in range(0, len(texts), 64):
            batch = texts[k : k + 64]
            enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
            gen = model.generate(**enc, num_beams=4, max_new_tokens=64)
            for src, tr in zip(batch, tok.batch_decode(gen, skip_special_tokens=True)):
                out[src] = tr
            if k % 1280 == 0:
                print(f"{k}/{len(texts)}", flush=True)
    kept = {s: t for s, t in out.items() if len(X.findall(t)) == 1}
    report = {
        "prereg": "SS158",
        "model": MT,
        "revision": REVISION,
        "n_texts": len(texts),
        "n_kept": len(kept),
        "kept_share": round(len(kept) / len(texts), 4),
        "map": kept,
        "dropped": {s: t for s, t in out.items() if s not in kept},
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("map", "dropped")}))


if __name__ == "__main__":
    main()
