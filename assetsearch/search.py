"""Search one user's past generations.

Four retrievers over the same rows, so the evaluation compares like with like:

* keyword  - Postgres full-text search, the way most apps ship search: every query
             word must match (websearch_to_tsquery), ranked by ts_rank_cd.
* keyword_any - full-text with OR semantics (any word matches), a stronger baseline.
* vector   - pgvector cosine distance over the trained encoder's embeddings.
* hybrid   - vector + keyword_any fused with reciprocal rank fusion (RRF).

Every query is scoped to one user_id. With an HNSW index that is a filtered
approximate search, which is where pgvector's iterative scan matters: without it the
index returns its ef_search nearest neighbours across *all* users, the filter then
throws most of them away, and a small user gets few or no results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from pgvector.psycopg import register_vector
from psycopg import Connection

from .ingest import to_halfvec as _halfvec

RRF_K = 60


@dataclass
class Hit:
    id: int
    score: float


def _or_query(q: str) -> str:
    words = re.findall(r"[a-z0-9]+", q.lower())
    return " | ".join(words) or "''"


def keyword(conn: Connection, user_id: str, q: str, k: int = 10, any_word: bool = False, kind: str | None = None) -> list[Hit]:
    tsq = "to_tsquery('english', %(q)s)" if any_word else "websearch_to_tsquery('english', %(q)s)"
    rows = conn.execute(
        f"""SELECT id, ts_rank_cd(tsv, {tsq}) AS s FROM assets
            WHERE user_id = %(u)s AND tsv @@ {tsq} AND (%(kind)s::text IS NULL OR kind = %(kind)s)
            ORDER BY s DESC, id LIMIT %(k)s""",
        {"q": _or_query(q) if any_word else q, "u": user_id, "k": k, "kind": kind},
    ).fetchall()
    return [Hit(r[0], float(r[1])) for r in rows]


def vector(conn: Connection, user_id: str, qv: np.ndarray, k: int = 10, kind: str | None = None,
           ef_search: int = 100, iterative: bool = True) -> list[Hit]:
    register_vector(conn)
    with conn.transaction():
        conn.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
        conn.execute(f"SET LOCAL hnsw.iterative_scan = {'relaxed_order' if iterative else 'off'}")
        rows = conn.execute(
            """SELECT id, embedding <=> %(v)s::halfvec AS d FROM assets
               WHERE user_id = %(u)s AND (%(kind)s::text IS NULL OR kind = %(kind)s)
               ORDER BY d LIMIT %(k)s""",
            {"v": _halfvec(qv), "u": user_id, "k": k, "kind": kind},
        ).fetchall()
    # relaxed_order can return rows slightly out of order; re-sort the small result.
    return sorted((Hit(r[0], 1 - float(r[1])) for r in rows), key=lambda h: -h.score)


def rrf(*lists: list[Hit], k: int = 10, weights: tuple[float, ...] | None = None) -> list[Hit]:
    weights = weights or (1.0,) * len(lists)
    score: dict[int, float] = {}
    for w, hits in zip(weights, lists):
        for rank, h in enumerate(hits):
            score[h.id] = score.get(h.id, 0.0) + w / (RRF_K + rank + 1)
    return [Hit(i, s) for i, s in sorted(score.items(), key=lambda x: -x[1])[:k]]


def hybrid(conn: Connection, user_id: str, q: str, qv: np.ndarray, k: int = 10, kind: str | None = None,
           depth: int = 50) -> list[Hit]:
    v = vector(conn, user_id, qv, depth, kind)
    kw = keyword(conn, user_id, q, depth, any_word=True, kind=kind)
    return rrf(v, kw, k=k, weights=(1.0, 0.5))

