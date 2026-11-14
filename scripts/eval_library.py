"""Score every retriever on the hand-labeled queries over the real generation library.

    python scripts/eval_library.py            # loads data/library.jsonl as user "ryan"
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assetsearch import db, search  # noqa: E402
from assetsearch.embed import Embedder  # noqa: E402
from assetsearch.ingest import insert_assets, search_text  # noqa: E402

USER = "eval-ryan"


def load_queries(path: Path) -> list[tuple[str, list[str]]]:
    out = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            q, rel = line.split("\t")
            out.append((q, rel.split(",")))
    return out


def expand(assets: list[dict], keys: list[str]) -> set[str]:
    """A labeled asset stands for every asset with the same text plus everything derived from it."""
    by_text = defaultdict(set)
    children = defaultdict(set)
    for a in assets:
        by_text[search_text(a)].add(a["key"])
        for p in a.get("parents", []):
            children[p].add(a["key"])
    text_of = {a["key"]: search_text(a) for a in assets}
    rel = set()
    frontier = [k for k in keys]
    for k in frontier:
        if k not in text_of:
            raise KeyError(f"unknown asset key in labels: {k}")
    while frontier:
        k = frontier.pop()
        for s in by_text[text_of[k]] | {k}:
            if s not in rel:
                rel.add(s)
                frontier.extend(children[s])
    return rel


def metrics(ranked: list[str], rel: set[str]) -> dict[str, float]:
    m = {}
    for k in (1, 5, 10):
        m[f"R@{k}"] = len(set(ranked[:k]) & rel) / min(len(rel), k)  # normalised: 1.0 means the top k is as good as it can be
    m["hit@10"] = float(bool(set(ranked[:10]) & rel))
    m["MRR"] = next((1 / (i + 1) for i, r in enumerate(ranked[:10]) if r in rel), 0.0)
    dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(ranked[:10]) if r in rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), 10)))
    m["nDCG@10"] = dcg / idcg
    m["empty"] = float(not ranked)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=Path, default=Path("data/library.jsonl"))
    ap.add_argument("--queries", type=Path, default=Path("eval/queries.tsv"))
    ap.add_argument("--model", type=Path, default=Path("models/encoder"))
    ap.add_argument("--out", type=Path, default=Path("results/library_eval.json"))
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()

    assets = [json.loads(line) for line in args.library.read_text().splitlines()]
    emb = Embedder(args.model)
    conn = db.connect()
    db.migrate(conn)
    conn.execute("DELETE FROM assets WHERE user_id = %s", (USER,))
    ids = insert_assets(conn, USER, assets, emb)
    key_of = {v: k for k, v in ids.items()}
    queries = load_queries(args.queries)

    systems = {
        "keyword (all words)": lambda q, v: search.keyword(conn, USER, q, 10),
        "keyword (any word)": lambda q, v: search.keyword(conn, USER, q, 10, any_word=True),
        "vector (whole prompt)": lambda q, v: search.vector(conn, USER, v, 10, chunks=False),
        "vector (+ sentence chunks)": lambda q, v: search.vector(conn, USER, v, 10),
        "hybrid (RRF)": lambda q, v: search.hybrid(conn, USER, q, v, 10),
    }
    if args.tune:
        # Fusion weight sweep: run on the dev queries only, then fix the value for the held-out set.
        for w in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
            systems[f"hybrid kw_weight={w}"] = (lambda w: lambda q, v: search.hybrid(conn, USER, q, v, 10, kw_weight=w))(w)
    # Objective "vocabulary mismatch" subset: queries where the keyword-any baseline finds
    # nothing relevant in its top 10 *because no stemmed query word occurs in any relevant asset*.
    def stems(t):
        return set(conn.execute("SELECT coalesce(array_agg(lexeme), '{}') FROM unnest(to_tsvector('english', %s))", (t,)).fetchone()[0])
    text_of = {a["key"]: search_text(a) for a in assets}
    mismatch = {q for q, keys in queries if not (stems(q) & set().union(*(stems(text_of[k]) for k in expand(assets, keys))))}
    report, per_query = {}, defaultdict(dict)
    for name, fn in systems.items():
        rows, lat = [], []
        for q, keys in queries:
            rel = expand(assets, keys)
            t0 = time.perf_counter()
            v = emb.encode([q])[0] if "keyword" not in name else None  # encode time counts toward latency
            hits = fn(q, v)
            lat.append((time.perf_counter() - t0) * 1000)
            ranked = [key_of[h.id] for h in hits]
            m = metrics(ranked, rel)
            rows.append(m)
            per_query[q][name] = {"MRR": m["MRR"], "top3": ranked[:3]}
        report[name] = {k: round(statistics.mean(r[k] for r in rows), 3) for k in rows[0]}
        mm = [r for r, (q, _) in zip(rows, queries) if q in mismatch]
        report[name]["MRR (no shared words)"] = round(statistics.mean(r["MRR"] for r in mm), 3) if mm else None
        report[name]["p50 ms"] = round(statistics.median(lat), 2)
    print(f"{len(assets)} assets, {len(queries)} queries, {len(mismatch)} share no stemmed word with any relevant asset\n")
    cols = ["R@1", "R@5", "R@10", "hit@10", "MRR", "nDCG@10", "empty", "p50 ms", "MRR (no shared words)"]
    print(f"{'system':28s}" + "".join(f"{c[:9]:>10s}" for c in cols))
    for name, m in report.items():
        print(f"{name:28s}" + "".join(f"{(m[c] if m[c] is not None else float('nan')):10.3f}" for c in cols))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n_assets": len(assets), "n_queries": len(queries), "systems": report,
                                    "mismatch_queries": sorted(mismatch), "per_query": per_query}, indent=1))


if __name__ == "__main__":
    main()
