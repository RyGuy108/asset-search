"""Pick the hybrid fusion parameters on dev data only.

Two dev sets pull in opposite directions, which is the point of tuning on both:
  * library dev queries (eval/queries.tsv): users reuse words from their prompts, keyword is strong
  * COCO dev split (1,000 images): query and prompt written by different people, vector is strong
The objective is the mean MRR@10 of the two. Run eval_library.py and eval_coco.py --split dev first
so both users are loaded.
"""

from __future__ import annotations

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
from scripts.eval_library import USER as LIB_USER, expand, load_queries  # noqa: E402


def mrr(ranked, rel):
    return next((1 / (i + 1) for i, r in enumerate(ranked[:10]) if r in rel), 0.0)


def main() -> None:
    emb = Embedder("models/encoder")
    conn = db.connect()
    sets = []
    assets = [json.loads(line) for line in open("data/library.jsonl")]
    ids = dict(conn.execute("SELECT id, key FROM assets WHERE user_id = %s", (LIB_USER,)).fetchall())
    sets.append(("library dev", LIB_USER, [(q, expand(assets, k)) for q, k in load_queries(Path("eval/queries.tsv"))], ids))
    rng = random.Random(42)
    dev = split_coco(Path("data/captions_val2014.json"))["dev"]
    cq = []
    for img, caps in sorted(dev.items()):
        a, b = rng.sample(caps, 2)
        style = random_style(rng)
        make_prompt(b, rng, style)  # keep the RNG stream identical to eval_coco.py
        cq.append((make_query(a, rng, style), {str(img)}))
    ids = dict(conn.execute("SELECT id, key FROM assets WHERE user_id = 'eval-coco-dev'").fetchall())
    sets.append(("coco dev", "eval-coco-dev", cq, ids))

    cache = {}
    for name, user, qs, key_of in sets:
        qvs = emb.encode([q for q, _ in qs])
        cache[name] = [(search.vector(conn, user, v, 50), search.keyword(conn, user, q, 50, any_word=True), rel, key_of)
                       for (q, rel), v in zip(qs, qvs)]
    results = []
    for w in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0):
        for p in (0.0, 1.0, 2.0, 3.0, 4.0):
            per = {}
            for name, rows in cache.items():
                per[name] = statistics.mean(
                    mrr([key_of[h.id] for h in search.rrf(v, kw, k=10, weights=(1.0, w), coverage_power=p)], rel)
                    for v, kw, rel, key_of in rows)
            results.append((statistics.mean(per.values()), w, p, per))
    results.sort(key=lambda r: -r[0])
    for obj, w, p, per in results[:8]:
        print(f"kw_weight {w:<5} coverage_power {p:<4} mean MRR {obj:.3f}  " + "  ".join(f"{k} {v:.3f}" for k, v in per.items()))
    for name, rows in cache.items():
        vec = statistics.mean(mrr([key_of[h.id] for h in v], rel) for v, kw, rel, key_of in rows)
        kwm = statistics.mean(mrr([key_of[h.id] for h in kw], rel) for v, kw, rel, key_of in rows)
        print(f"  reference {name}: vector {vec:.3f}  keyword-any {kwm:.3f}")
    Path("results/fusion_tuning.json").write_text(json.dumps([{"kw_weight": w, "coverage_power": p, **per, "mean": o} for o, w, p, per in results], indent=1))


if __name__ == "__main__":
    main()
