"""Insert generations into the asset table with their embeddings."""

from __future__ import annotations

import re

import numpy as np
from psycopg import Connection

from .embed import Embedder


def search_text(a: dict) -> str:
    parts = [a.get("prompt") or "", a.get("derived") or ""]
    return " ".join(p for p in parts if p).strip() or a.get("tool", "")


CHUNK_MIN_WORDS = 24


def chunks(text: str) -> list[str]:
    """Sentence-level pieces of a long prompt; short prompts get none (the whole prompt is enough)."""
    if len(text.split()) < CHUNK_MIN_WORDS:
        return []
    out, cur = [], []
    for sent in re.split(r"(?<=[.;:!?])\s+", text):
        cur.append(sent)
        if len(" ".join(cur).split()) >= 6:
            out.append(" ".join(cur))
            cur = []
    if cur:
        if out:
            out[-1] += " " + " ".join(cur)
        else:
            out.append(" ".join(cur))
    # A long single sentence: fall back to windows of comma clauses.
    if len(out) == 1:
        clauses = [c.strip() for c in text.split(",") if c.strip()]
        out = [", ".join(clauses[i : i + 3]) for i in range(0, len(clauses), 3)]
    return out if len(out) > 1 else []


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
                       embedding = EXCLUDED.embedding, prompt = EXCLUDED.prompt, path = EXCLUDED.path
                   RETURNING id""",
                (user_id, a["key"], a.get("mh_id"), a["kind"], a["tool"], a.get("model"), a.get("prompt"), t,
                 a.get("path"), a.get("url"), a.get("credits"), to_halfvec(v)),
            ).fetchone()
            ids[a["key"]] = row[0]
        pieces = [(ids[a["key"]], i, c) for a, t in zip(assets, texts) for i, c in enumerate(chunks(t))]
        if pieces:
            cvecs = emb.encode([c for _, _, c in pieces])
            conn.execute("DELETE FROM asset_chunks WHERE asset_id = ANY(%s)", (list({p[0] for p in pieces}),))
            with conn.cursor().copy("COPY asset_chunks (asset_id, user_id, n, text, embedding) FROM STDIN") as cp:
                for (aid, i, c), v in zip(pieces, cvecs):
                    cp.write_row((aid, user_id, i, c, to_halfvec(v)))
        for a in assets:
            parents = [ids[p] for p in a.get("parents", []) if p in ids]
            if parents:
                conn.execute("UPDATE assets SET parent_ids = %s WHERE id = %s", (parents, ids[a["key"]]))
    return ids
