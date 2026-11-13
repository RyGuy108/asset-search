"""Insert generations into the asset table with their embeddings."""

from __future__ import annotations

import numpy as np
from psycopg import Connection

from .embed import Embedder


def search_text(a: dict) -> str:
    parts = [a.get("prompt") or "", a.get("derived") or ""]
    return " ".join(p for p in parts if p).strip() or a.get("tool", "")


def to_halfvec(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.5f}" for x in v.tolist()) + "]"


def insert_assets(conn: Connection, user_id: str, assets: list[dict], emb: Embedder) -> dict[str, int]:
    """Insert (or update) assets for one user. Returns key -> row id."""
    texts = [search_text(a) for a in assets]
    vecs = emb.encode(texts)
    ids: dict[str, int] = {}
    with conn.transaction():
        for a, t, v in zip(assets, texts, vecs):
            row = conn.execute(
                """INSERT INTO assets (user_id, key, mh_id, kind, tool, model, prompt, search_text, path, url, credits, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::halfvec)
                   ON CONFLICT (user_id, key) DO UPDATE SET search_text = EXCLUDED.search_text,
                       embedding = EXCLUDED.embedding, prompt = EXCLUDED.prompt
                   RETURNING id""",
                (user_id, a["key"], a.get("mh_id"), a["kind"], a["tool"], a.get("model"), a.get("prompt"), t,
                 a.get("path"), a.get("url"), a.get("credits"), to_halfvec(v)),
            ).fetchone()
            ids[a["key"]] = row[0]
        for a in assets:
            parents = [ids[p] for p in a.get("parents", []) if p in ids]
            if parents:
                conn.execute("UPDATE assets SET parent_ids = %s WHERE id = %s", (parents, ids[a["key"]]))
    return ids
