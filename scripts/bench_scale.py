"""Latency and filtered-search recall at 1M assets.

    python scripts/bench_scale.py load      # build 1M synthetic generations across 3,611 users (COPY), then HNSW
    python scripts/bench_scale.py bench     # per-user search: exact vs HNSW vs HNSW + iterative scan vs keyword

The synthetic library is COCO captions (train + dev split, never the test images) turned
into prompts with the same style families the encoder was trained on. Users follow a
heavy-tailed size distribution, because that is what breaks filtered vector search:

    1 user x 100,000   10 x 20,000   100 x 3,000   1,500 x 200   2,000 x 50   = 1,000,000
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assetsearch import db  # noqa: E402
from assetsearch.embed import Embedder  # noqa: E402
from assetsearch.encoder.augment import make_prompt, make_query, random_style  # noqa: E402
from assetsearch.encoder.train import split_coco  # noqa: E402
from assetsearch.ingest import to_halfvec  # noqa: E402

TIERS = [("whale", 1, 100_000), ("large", 10, 20_000), ("medium", 100, 3_000), ("small", 1_500, 200), ("tiny", 2_000, 50)]
PREFIX = "bench-"


def users():
    for tier, count, size in TIERS:
        for i in range(count):
            yield f"{PREFIX}{tier}-{i}", tier, size


def load(args) -> None:
    splits = split_coco(Path("data/captions_val2014.json"))
    caps = [c for part in ("train", "dev") for cs in splits[part].values() for c in cs]
    rng = random.Random(7)
    emb = Embedder(args.model)
    conn = db.connect()
    db.migrate(conn, include_index=False)
    conn.execute("DROP INDEX IF EXISTS assets_embedding_hnsw")
    conn.execute(f"DELETE FROM assets WHERE user_id LIKE '{PREFIX}%'")
    total = sum(c * s for _, c, s in TIERS)
    t0 = time.time()
    done = 0
    batch_users, batch = [], []

    def flush():
        nonlocal done
        texts = [p for _, p, _ in batch]
        vecs = emb.encode(texts, batch=512)
        with conn.cursor().copy("COPY assets (user_id, key, kind, tool, prompt, search_text, embedding, created_at) FROM STDIN") as cp:
            for (u, p, ts), v in zip(batch, vecs):
                done += 1
                cp.write_row((u, f"syn-{done}", "image", "ai_image_generator", p, p, to_halfvec(v), ts))
        batch.clear()
        el = time.time() - t0
        print(f"\r{done:,}/{total:,}  {done / el:,.0f} rows/s", end="", flush=True)

    for u, _, size in users():
        for _ in range(size):
            p = make_prompt(rng.choice(caps), rng, random_style(rng))
            ts = f"2026-{rng.randint(1, 10):02d}-{rng.randint(1, 28):02d}"
            batch.append((u, p, ts))
            if len(batch) >= 20_000:
                flush()
    if batch:
        flush()
    load_s = time.time() - t0
    print(f"\nloaded in {load_s:.0f}s")
    reindex(args, load_s)


def reindex(args, load_s=None) -> None:
    """Drop and rebuild the HNSW index on the full table, timing the build."""
    conn = db.connect()
    conn.execute("DROP INDEX IF EXISTS assets_embedding_hnsw")
    total = conn.execute("SELECT count(*) FROM assets").fetchone()[0]
    print(f"building HNSW over {total:,} rows...", flush=True)
    conn.execute("SET maintenance_work_mem = '6GB'")
    conn.execute("SET max_parallel_maintenance_workers = 3")
    t1 = time.time()
    conn.execute((Path(db.MIGRATIONS) / "002_hnsw.sql").read_text())
    idx_s = time.time() - t1
    conn.execute("VACUUM ANALYZE assets")
    size = conn.execute("SELECT pg_size_pretty(pg_relation_size('assets_embedding_hnsw'))").fetchone()[0]
    print(f"HNSW built in {idx_s:.0f}s, {size}")
    Path("results").mkdir(exist_ok=True)
    prev = json.loads(Path("results/scale_load.json").read_text()) if Path("results/scale_load.json").exists() else {}
    prev.update({"rows": total, "hnsw_build_s": round(idx_s), "hnsw_size": size, "hnsw": "m=16, ef_construction=64, halfvec(256), 4 cores"})
    if load_s is not None:
        prev["load_s"] = round(load_s)
    Path("results/scale_load.json").write_text(json.dumps(prev, indent=1))


SQL_VEC = """SELECT id FROM assets WHERE {filter} ORDER BY embedding <=> %(v)s::halfvec LIMIT 10"""


def run(conn, mode: str, user: str, qtext: str, qv: np.ndarray, ef: int) -> list[int]:
    v = to_halfvec(qv)
    with conn.transaction():
        conn.execute(f"SET LOCAL hnsw.ef_search = {ef}")
        if mode == "exact":
            # Materialise the user's rows first so no ANN index can be used: true top 10.
            sql = """WITH mine AS MATERIALIZED (SELECT id, embedding FROM assets WHERE user_id = %(u)s)
                     SELECT id FROM mine ORDER BY embedding <=> %(v)s::halfvec LIMIT 10"""
        elif mode == "planner default":
            conn.execute("SET LOCAL hnsw.iterative_scan = off")
            sql = SQL_VEC.format(filter="user_id = %(u)s")
        elif mode == "hnsw post-filter":
            # user_id || '' hides the btree from the planner, forcing the ANN index + filter.
            conn.execute("SET LOCAL hnsw.iterative_scan = off")
            sql = SQL_VEC.format(filter="user_id || '' = %(u)s")
        elif mode == "hnsw iterative":
            conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            conn.execute("SET LOCAL hnsw.max_scan_tuples = 200000")
            sql = SQL_VEC.format(filter="user_id || '' = %(u)s")
        elif mode == "shipped (iterative + planner)":
            conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            conn.execute("SET LOCAL hnsw.max_scan_tuples = 200000")
            sql = SQL_VEC.format(filter="user_id = %(u)s")
        elif mode == "keyword (all words)":
            sql = """SELECT id FROM assets WHERE user_id = %(u)s AND tsv @@ websearch_to_tsquery('english', %(q)s)
                     ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', %(q)s)) DESC LIMIT 10"""
        else:
            raise ValueError(mode)
        return [r[0] for r in conn.execute(sql, {"u": user, "v": v, "q": qtext}).fetchall()]


def bench(args) -> None:
    splits = split_coco(Path("data/captions_val2014.json"))
    test_caps = [c for cs in splits["test"].values() for c in cs]
    rng = random.Random(3)
    emb = Embedder(args.model)
    conn = db.connect()
    conn.execute("SET jit = off")
    modes = ["exact", "planner default", "hnsw post-filter", "hnsw iterative", "shipped (iterative + planner)", "keyword (all words)"]
    out = {}
    for tier, count, size in TIERS:
        picks = [f"{PREFIX}{tier}-{rng.randrange(count)}" for _ in range(args.n)]
        qs = [make_query(rng.choice(test_caps), rng, random_style(rng)) for _ in range(args.n)]
        qvs = emb.encode(qs)
        truth = [run(conn, "exact", u, q, v, args.ef) for u, q, v in zip(picks, qs, qvs)]
        row = {}
        for mode in modes:
            lat, rec, empty = [], [], 0
            for u, q, v, t in zip(picks, qs, qvs, truth):
                t0 = time.perf_counter()
                got = run(conn, mode, u, q, v, args.ef)
                lat.append((time.perf_counter() - t0) * 1000)
                rec.append(len(set(got) & set(t)) / max(1, len(t)))
                empty += len(got) < 10
            lat.sort()
            row[mode] = {"p50_ms": round(statistics.median(lat), 2), "p99_ms": round(lat[int(0.99 * (len(lat) - 1))], 2),
                         "recall@10_vs_exact": round(statistics.mean(rec), 3), "short_results": round(empty / len(picks), 3)}
        out[f"{tier} ({size:,} assets)"] = row
        print(f"\n== {tier}: {size:,} assets per user, {args.n} queries, ef_search={args.ef}")
        for mode, m in row.items():
            print(f"  {mode:32s} p50 {m['p50_ms']:7.2f} ms  p99 {m['p99_ms']:7.2f} ms  recall {m['recall@10_vs_exact']:.3f}  <10 results {m['short_results']:.0%}")
    Path("results").mkdir(exist_ok=True)
    Path("results/scale_bench.json").write_text(json.dumps({"ef_search": args.ef, "tiers": out}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["load", "reindex", "bench"])
    ap.add_argument("--model", type=Path, default=Path("models/encoder"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ef", type=int, default=100)
    a = ap.parse_args()
    {"load": load, "reindex": reindex, "bench": bench}[a.cmd](a)
