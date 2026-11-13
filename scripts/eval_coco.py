"""Second, larger benchmark: 3,000 held-out COCO images as one user's library.

Each image becomes one prompt-style asset (caption B + style boilerplate); the query is a
short search built from a different caption (A) of the same image, so query and prompt
were written by different people. Exactly one relevant asset per query.

The encoder never saw these images. It did see this query/prompt *shape* in training,
so treat this as the in-distribution number; eval_library.py (real generations,
hand-written queries) is the out-of-distribution one.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assetsearch import db, search  # noqa: E402
from assetsearch.embed import Embedder  # noqa: E402
from assetsearch.encoder.augment import make_prompt, make_query, random_style  # noqa: E402
from assetsearch.encoder.train import split_coco  # noqa: E402
from assetsearch.ingest import insert_assets  # noqa: E402

USER = "eval-coco"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("models/encoder"))
    ap.add_argument("--out", type=Path, default=Path("results/coco_eval.json"))
    args = ap.parse_args()
    rng = random.Random(42)
    test = split_coco(Path("data/captions_val2014.json"))["test"]
    assets, queries = [], []
    for img, caps in sorted(test.items()):
        a, b = rng.sample(caps, 2)
        style = random_style(rng)
        assets.append({"key": str(img), "kind": "image", "tool": "ai_image_generator", "prompt": make_prompt(b, rng, style)})
        queries.append((make_query(a, rng, style), str(img)))
    emb = Embedder(args.model)
    conn = db.connect()
    db.migrate(conn)
    conn.execute("DELETE FROM assets WHERE user_id = %s", (USER,))
    ids = insert_assets(conn, USER, assets, emb)
    key_of = {v: k for k, v in ids.items()}
    qvs = emb.encode([q for q, _ in queries])
    systems = {
        "keyword (all words)": lambda q, v: search.keyword(conn, USER, q, 10),
        "keyword (any word)": lambda q, v: search.keyword(conn, USER, q, 10, any_word=True),
        "vector": lambda q, v: search.vector(conn, USER, v, 10),
        "hybrid (RRF)": lambda q, v: search.hybrid(conn, USER, q, v, 10),
    }
    report = {}
    for name, fn in systems.items():
        r1, r10, mrr, empty = [], [], [], 0
        for (q, rel), v in zip(queries, qvs):
            ranked = [key_of[h.id] for h in fn(q, v)]
            empty += not ranked
            r1.append(float(ranked[:1] == [rel]))
            r10.append(float(rel in ranked))
            mrr.append(next((1 / (i + 1) for i, k in enumerate(ranked) if k == rel), 0.0))
        report[name] = {"R@1": round(statistics.mean(r1), 3), "R@10": round(statistics.mean(r10), 3),
                        "MRR@10": round(statistics.mean(mrr), 3), "empty": round(empty / len(queries), 3)}
        print(f"{name:22s} " + "  ".join(f"{k} {v:.3f}" for k, v in report[name].items()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n": len(queries), "examples": queries[:10], "systems": report}, indent=1))


if __name__ == "__main__":
    main()
