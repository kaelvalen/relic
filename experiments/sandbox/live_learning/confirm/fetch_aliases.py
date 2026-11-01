"""SS103 input: Wikidata aliases of the subjects of the N=1000 stores (no model).

Entities are looked up by English Wikipedia title, the subject string itself, then by
the top search hit if its English label is that string. Anything looser could pick the
wrong item; a subject found neither way has no aliases.

    .venv/bin/python experiments/sandbox/live_learning/confirm/fetch_aliases.py
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import certifi

CONFIRM = Path("results/live_learning/confirm")
OUT = CONFIRM / "subject_aliases.json"
PARTIAL = CONFIRM / "subject_aliases.partial.json"
API = "https://www.wikidata.org/w/api.php"
BATCH = 50


def call(params, ctx):
    req = urllib.request.Request(
        f"{API}?{urllib.parse.urlencode({**params, 'format': 'json'})}",
        headers={"User-Agent": "relic-research/0.3 (github.com/kaelvalen/relic)"},
    )
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 7:
                raise
            time.sleep(int(e.headers.get("Retry-After") or 30))


def by_title(titles, ctx):
    data = call(
        {
            "action": "wbgetentities",
            "sites": "enwiki",
            "titles": "|".join(titles),
            "props": "aliases|sitelinks",
            "languages": "en",
            "sitefilter": "enwiki",
        },
        ctx,
    )
    return {
        e["sitelinks"]["enwiki"]["title"]: {
            "qid": qid,
            "aliases": [a["value"] for a in e.get("aliases", {}).get("en", [])],
            "via": "enwiki",
        }
        for qid, e in data["entities"].items()
        if qid.startswith("Q") and "enwiki" in e.get("sitelinks", {})
    }


def by_label(subject, ctx):
    """The top search hit, only if its English label is the subject string."""
    hits = call(
        {
            "action": "wbsearchentities",
            "search": subject,
            "language": "en",
            "type": "item",
            "limit": 1,
        },
        ctx,
    )["search"]
    if not hits or hits[0].get("label", "").lower() != subject.lower():
        return None
    return {"qid": hits[0]["id"], "aliases": hits[0].get("aliases", []), "via": "label"}


def main() -> None:
    subjects = sorted(
        {
            f["subject"]
            for name in ("external_counterfact_n1000", "external_zsre_n1000")
            for f in json.loads((CONFIRM / f"{name}.json").read_text())["facts"]
        }
    )
    ctx = ssl.create_default_context(cafile=certifi.where())
    # the fetch can be resumed; the file is the input of SS103, not one of its results
    found = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {}
    tried = set(found.pop("__tried__", []))
    if not tried:
        for i in range(0, len(subjects), BATCH):
            found.update(by_title(subjects[i : i + BATCH], ctx))
            time.sleep(1.0)
    print(f"by title: {sum(s in found for s in subjects)}/{len(subjects)}", flush=True)
    for n, s in enumerate(subjects):
        if s not in found and s not in tried:
            hit = by_label(s, ctx)
            if hit:
                found[s] = hit
            tried.add(s)
            time.sleep(0.5)
        if n % 100 == 0:
            PARTIAL.write_text(json.dumps({**found, "__tried__": sorted(tried)}))
    found = {s: found[s] for s in subjects if s in found}
    OUT.write_text(json.dumps(found, ensure_ascii=False, indent=1))
    PARTIAL.unlink(missing_ok=True)
    with_alias = sum(bool(v["aliases"]) for v in found.values())
    print(f"subjects {len(subjects)}, entity {len(found)}, with aliases {with_alias}")


if __name__ == "__main__":
    main()
